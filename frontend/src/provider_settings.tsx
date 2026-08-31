import {useMemo,useState} from 'react';
import {ArrowLeft,FlaskConical,Plus,RefreshCw,Wifi} from 'lucide-react';
import type {Provider,ProviderDraft,ProviderInspection,ProviderKind,ProviderModel,ProviderRole,TokenLimits} from './api';
import {ConfirmDialog,Overlay} from './ui';

const KINDS_BY_ROLE:Record<ProviderRole,Array<{value:ProviderKind;label:string}>>={
  main:[{value:'openai',label:'OPENAI-COMPATIBLE'},{value:'ollama',label:'OLLAMA'}],
  vlm:[{value:'openai',label:'OPENAI-COMPATIBLE'},{value:'ollama',label:'OLLAMA'}],
  tts:[{value:'sandevistan_tts',label:'SANDEVISTAN TTS'},{value:'openai_tts',label:'OPENAI TTS'}],
};

type DrawerView={mode:'list'}|{mode:'add'}|{mode:'edit';provider:Provider};
type InspectionMode='catalog'|'deep';
type SettingsProps={
  status:any;
  providers:Provider[];
  onClose:()=>void;
  onSave:(id:string,body:Record<string,any>)=>Promise<void>;
  onCreate:(body:Record<string,any>)=>Promise<void>;
  onInspect:(draft:ProviderDraft,mode:InspectionMode)=>Promise<ProviderInspection>;
};

function newDraft():ProviderDraft{return{name:'',role:'main',kind:'openai',base_url:'',model:'',api_key:'',config:{}}}
function providerDraft(provider:Provider):ProviderDraft{return{provider_id:provider.id,name:provider.name,role:provider.role,kind:provider.kind,base_url:provider.base_url,model:provider.model,api_key:'',config:{...provider.config}}}

function ModelPicker({value,models,disabled,onChange}:{value:string;models:ProviderModel[];disabled?:boolean;onChange:(value:string)=>void}){
  const[open,setOpen]=useState(false);
  const normalized=value.trim().toLowerCase();
  const visible=useMemo(()=>models.filter(model=>!normalized||model.id.toLowerCase().includes(normalized)||model.name.toLowerCase().includes(normalized)).slice(0,80),[models,normalized]);
  return <div className="model-picker" onBlur={event=>{if(!event.currentTarget.contains(event.relatedTarget as Node|null))setOpen(false)}}>
    <input role="combobox" aria-expanded={open} aria-controls="provider-model-options" aria-autocomplete="list" disabled={disabled} value={value} placeholder={models.length?'搜索或手动输入模型 ID':'手动输入模型 ID'} onFocus={()=>setOpen(true)} onChange={event=>{onChange(event.target.value);setOpen(true)}} onKeyDown={event=>{if(event.key==='ArrowDown'){event.preventDefault();setOpen(true)}else if(event.key==='Escape')setOpen(false);else if(event.key==='Enter'&&open&&visible[0]){event.preventDefault();onChange(visible[0].id);setOpen(false)}}}/>
    {open&&visible.length?<ul id="provider-model-options" role="listbox">{visible.map(model=><li key={model.id} role="option" aria-selected={model.id===value}><button type="button" disabled={model.installed===false} onMouseDown={event=>event.preventDefault()} onClick={()=>{onChange(model.id);setOpen(false)}}><b>{model.name}</b><small>{model.id}{model.installed===false?' · 未安装':''}</small></button></li>)}</ul>:null}
  </div>;
}

function InspectionPanel({inspection,mode}:{inspection?:ProviderInspection;mode:InspectionMode}){
  if(!inspection)return <p className="provider-inspection idle">连接检查不会保存配置，也不会发送 Notebook 资料。</p>;
  const message=inspection.error?.message||inspection.warning||(mode==='deep'?'深度验证通过':'连接与模型清单验证通过');
  const hint=inspection.error?.hint||(`${inspection.models.length} 个模型 · ${inspection.latency_ms} ms`);
  return <div className={`provider-inspection ${inspection.status}`} role={inspection.status==='failed'?'alert':'status'}><b>{message}</b><small>{hint}</small></div>;
}

const LIMIT_SOURCE_LABELS:Record<string,string>={manual:'人工覆盖',ollama_runtime:'Ollama 当前运行',ollama_modelfile:'Modelfile',provider_metadata:'Provider 元数据',fallback:'安全回退'};
function tokenLimitText(limits?:Partial<TokenLimits>){
  if(!limits)return '窗口能力尚未读取';
  const maximum=limits.model_context_tokens?`理论最大 ${limits.model_context_tokens.toLocaleString()}`:'理论最大未知';
  const effective=limits.effective_context_tokens?`运行 ${limits.effective_context_tokens.toLocaleString()}`:'运行未知';
  const output=limits.max_output_tokens?`输出 ${limits.max_output_tokens.toLocaleString()}`:'输出未知';
  return `${maximum} · ${effective} · ${output} · ${LIMIT_SOURCE_LABELS[limits.context_source||'']||limits.context_source||'未知来源'}`;
}

function ProviderStatus({provider,status}:{provider:Provider;status:any}){
  const active=Boolean(provider.active),online=active&&Boolean(status?.providers?.[provider.role]?.ok);
  const label=!active?'未启用':online?'活跃在线':'活跃异常';
  const limits=provider.capabilities?.token_limits as Partial<TokenLimits>|undefined;
  return <div className="provider"><i className={!active?'inactive':online?'ok':''}/><span><b>{provider.role.toUpperCase()} · {provider.name}</b><small>{label} · {provider.kind} · {provider.model||'未选择模型'}{provider.config?.compute_device?` · ${String(provider.config.compute_device).toUpperCase()}`:''}{provider.has_api_key?' · KEY SAVED':''}</small>{provider.role!=='tts'?<small>CTX/OUT · {tokenLimitText(limits)}</small>:null}</span></div>;
}

export function SettingsDrawer({status,providers,onClose,onSave,onCreate,onInspect}:SettingsProps){
  const[view,setView]=useState<DrawerView>({mode:'list'});
  const[draft,setDraft]=useState<ProviderDraft>(newDraft);
  const[inspection,setInspection]=useState<ProviderInspection>();
  const[lastMode,setLastMode]=useState<InspectionMode>('catalog');
  const[busy,setBusy]=useState('');
  const[dirty,setDirty]=useState(false);
  const[leaveTarget,setLeaveTarget]=useState<'list'|'close'>();
  const[cardResults,setCardResults]=useState<Record<string,ProviderInspection>>({});

  const selectedModel=inspection?.models.find(model=>model.id===draft.model);
  const devices=selectedModel?.devices||[];
  const voices=Array.isArray(inspection?.capabilities?.voices)?inspection.capabilities.voices:[];
  const controls=selectedModel?.controls||{};
  const supportsInstruct=(controls.instruction_voice_modes||[]).includes('preset');
  const isForm=view.mode!=='list';
  const currentProvider=view.mode==='edit'?view.provider:undefined;
  const tokenLimits=(inspection?.capabilities?.token_limits||(currentProvider?.model===draft.model?currentProvider?.capabilities?.token_limits:undefined)) as Partial<TokenLimits>|undefined;
  const canSave=Boolean(draft.name.trim()&&draft.base_url.trim());

  function mutate(updater:(current:ProviderDraft)=>ProviderDraft,invalidates=true){
    setDraft(current=>updater(current));setDirty(true);if(invalidates)setInspection(undefined);
  }
  function beginAdd(){setView({mode:'add'});setDraft(newDraft());setInspection(undefined);setLastMode('catalog');setDirty(false)}
  function beginEdit(provider:Provider){const next=providerDraft(provider);setView({mode:'edit',provider});setDraft(next);setInspection(undefined);setLastMode('catalog');setDirty(false);void inspect(next,'catalog',true)}
  function requestLeave(target:'list'|'close'){if(isForm&&dirty)setLeaveTarget(target);else if(target==='close')onClose();else setView({mode:'list'})}
  function confirmLeave(){const target=leaveTarget;setLeaveTarget(undefined);setDirty(false);if(target==='close')onClose();else setView({mode:'list'})}

  async function inspect(target:ProviderDraft,mode:InspectionMode,quiet=false){
    if(!target.base_url.trim())return;
    setBusy(mode);setLastMode(mode);
    try{
      const result=await onInspect(target,mode);setInspection(result);
      if(result.recommended?.model&&target.kind==='sandevistan_tts'&&target.config?.auto_select){
        setDraft(current=>({...current,model:result.recommended?.model||current.model,config:{...current.config,compute_device:result.recommended?.compute_device||current.config.compute_device}}));
      }
      if(!quiet)setDirty(true);
    }catch{/* The application-level toast already reports request failures. */}finally{setBusy('')}
  }

  async function inspectCard(provider:Provider){
    setBusy(`card-${provider.id}`);
    try{const result=await onInspect(providerDraft(provider),'catalog');setCardResults(current=>({...current,[provider.id]:result}))}catch{/* Keep the existing card state. */}finally{setBusy('')}
  }

  function payload(active:boolean){
    const validationMode=lastMode==='deep'&&inspection?.catalog_supported?'catalog':lastMode;
    const body:Record<string,any>={name:draft.name.trim(),base_url:draft.base_url.trim(),model:draft.model.trim(),config:draft.config,active,validation_mode:validationMode};
    if(draft.api_key)body.api_key=draft.api_key;
    if(view.mode==='add'){body.role=draft.role;body.kind=draft.kind;body.capabilities={}}
    return body;
  }
  async function save(active:boolean){
    setBusy(active?'activate':'save');
    try{
      if(currentProvider)await onSave(currentProvider.id,payload(active));else await onCreate(payload(active));
      setDirty(false);setInspection(undefined);setView({mode:'list'});
    }catch{/* Keep the draft available for correction. */}finally{setBusy('')}
  }

  function updateRole(role:ProviderRole){const kind=KINDS_BY_ROLE[role][0].value;mutate(current=>({...current,role,kind,model:'',config:kind==='sandevistan_tts'?{auto_select:true}:{}}))}
  function updateModel(model:string){mutate(current=>{const match=inspection?.models.find(item=>item.id===model),available=match?.devices?.find(device=>device.available&&device.id==='gpu')||match?.devices?.find(device=>device.available);return{...current,model,config:{...current.config,...(available?{auto_select:false,compute_device:available.id}:{})}}})}
  function updateTokenOverride(field:'context_window_tokens'|'max_output_tokens',value:string){mutate(current=>{const config={...current.config};if(value)config[field]=Number(value);else delete config[field];return{...current,config}})}

  return <Overlay className={`settings ${isForm?'settings-form-view':''}`} label="Provider 配置" onClose={()=>requestLeave('close')} closeOnBackdrop={false}>
    <button className="drawer-close" data-autofocus onClick={()=>requestLeave('close')}>关闭 ×</button>
    <span>SYSTEM CONFIG</span>
    <h2>{isForm?(view.mode==='add'?'添加 Provider':`编辑 ${currentProvider?.name}`):'Provider 配置'}</h2>
    {view.mode==='list'?<>
      {providers.map(provider=><section className="provider-card" key={provider.id}><ProviderStatus provider={provider} status={status}/><div className="provider-actions"><button onClick={()=>beginEdit(provider)}>编辑</button><button disabled={Boolean(busy)} onClick={()=>void inspectCard(provider)}>{busy===`card-${provider.id}`?'检查中…':'检查'}</button>{cardResults[provider.id]?<small className={cardResults[provider.id].status} role="status">{cardResults[provider.id].error?.message||cardResults[provider.id].warning||`ONLINE · ${cardResults[provider.id].latency_ms} ms`}</small>:null}</div></section>)}
      <button className="add-provider" onClick={beginAdd}><Plus size={14}/> 添加 Provider</button>
      <h3>项目工具</h3>{['ffmpeg','libreoffice'].map(name=>{const tool=status?.tools?.[name];return <div className="provider" key={name}><i className={tool?.available?'ok':''}/><span><b>{name.toUpperCase()}</b><small>{tool?.version||'未安装'} · {tool?.scope||'missing'}</small></span></div>})}
      <h3>检索</h3><div className="provider"><i className={status?.retrieval?.embedding_mode==='sentence-transformers'?'ok':''}/><span><b>{status?.retrieval?.embedding_mode||'正在读取'}</b><small>{status?.retrieval?.model||'状态将在后台更新'}</small></span></div>
      <p>Provider 凭据在项目内加密保存。启用云端 Provider 时，选中的资料上下文会发送给该服务。</p>
    </>:<div className="provider-editor">
      <button className="provider-back" onClick={()=>requestLeave('list')}><ArrowLeft size={15}/> 返回 Provider 列表</button>
      <div className="provider-form">
        <div className="provider-form-section"><h3>基本信息</h3><label>名称<input value={draft.name} onChange={event=>mutate(current=>({...current,name:event.target.value}),false)}/></label><div className="provider-form-grid"><label>角色<select value={draft.role} disabled={view.mode==='edit'} onChange={event=>updateRole(event.target.value as ProviderRole)}><option value="main">MAIN</option><option value="vlm">VLM</option><option value="tts">TTS</option></select></label><label>类型<select value={draft.kind} disabled={view.mode==='edit'} onChange={event=>mutate(current=>({...current,kind:event.target.value as ProviderKind,model:'',config:event.target.value==='sandevistan_tts'?{auto_select:true}:{}}))}>{KINDS_BY_ROLE[draft.role].map(option=><option key={option.value} value={option.value}>{option.label}</option>)}</select></label></div></div>
        <div className="provider-form-section"><h3>连接</h3><label>服务地址<input value={draft.base_url} placeholder={draft.kind==='ollama'?'http://localhost:11434':draft.kind==='sandevistan_tts'?'http://localhost:20810':'https://api.example.com/v1'} onChange={event=>mutate(current=>({...current,base_url:event.target.value}))}/><small>可直接粘贴带 /v1 或 /api 的地址，保存时会自动规范化。</small></label><label>API Key<input type="password" autoComplete="new-password" value={draft.api_key||''} placeholder={currentProvider?.has_api_key?'已安全保存；留空保持不变':'可选'} onChange={event=>mutate(current=>({...current,api_key:event.target.value}))}/></label><div className="provider-check-actions"><button disabled={!draft.base_url.trim()||Boolean(busy)} onClick={()=>void inspect(draft,'catalog')}><Wifi size={14}/>{busy==='catalog'?'正在连接…':'连接并读取模型'}</button><button disabled={!draft.model.trim()||Boolean(busy)} onClick={()=>void inspect(draft,'deep')}><FlaskConical size={14}/>{busy==='deep'?'正在验证…':'深度验证'}</button><button aria-label="刷新能力清单" disabled={!draft.base_url.trim()||Boolean(busy)} onClick={()=>void inspect(draft,'catalog')}><RefreshCw size={14}/></button></div><small className="provider-cost-note">深度验证使用固定测试内容，不含 Notebook 资料；云服务可能产生少量费用。</small><InspectionPanel inspection={inspection} mode={lastMode}/></div>
        <div className="provider-form-section"><h3>模型与能力</h3>{draft.kind==='sandevistan_tts'?<><label className="provider-check"><input type="checkbox" checked={Boolean(draft.config.auto_select)} onChange={event=>mutate(current=>({...current,config:{...current.config,auto_select:event.target.checked}}))}/> 自动选择已安装的最高质量模型</label><label>模型<ModelPicker value={draft.model} models={inspection?.models||[]} disabled={Boolean(draft.config.auto_select)} onChange={updateModel}/></label><label>设备<select value={draft.config.compute_device||''} disabled={Boolean(draft.config.auto_select)||!devices.length} onChange={event=>mutate(current=>({...current,config:{...current.config,auto_select:false,compute_device:event.target.value}}))}><option value="">连接后选择设备</option>{devices.map(device=><option key={device.id} value={device.id} disabled={!device.available}>{device.id.toUpperCase()} · {device.available?device.precision:device.reason||'不可用'}</option>)}</select></label><div className="voice-grid">{['host_a','host_b'].map(field=><label key={field}>{field.toUpperCase()}{voices.length?<select value={draft.config[field]||''} onChange={event=>mutate(current=>({...current,config:{...current.config,[field]:event.target.value}}))}><option value="">选择音色</option>{voices.map((voice:any)=><option key={voice.id} value={voice.id}>{voice.id} · {voice.native_language||'MULTI'}</option>)}</select>:<input value={draft.config[field]||''} placeholder="连接后选择或手填" onChange={event=>mutate(current=>({...current,config:{...current.config,[field]:event.target.value}}))}/>}</label>)}</div>{supportsInstruct?<><label>Host A 语气<textarea value={draft.config.host_a_instruct||''} onChange={event=>mutate(current=>({...current,config:{...current.config,host_a_instruct:event.target.value}}))}/></label><label>Host B 语气<textarea value={draft.config.host_b_instruct||''} onChange={event=>mutate(current=>({...current,config:{...current.config,host_b_instruct:event.target.value}}))}/></label></>:null}<label className="provider-check"><input type="checkbox" checked={draft.config.allow_device_fallback!==false} onChange={event=>mutate(current=>({...current,config:{...current.config,allow_device_fallback:event.target.checked}}))}/> GPU 失败时回退同模型 CPU</label></>:<><label>模型<ModelPicker value={draft.model} models={inspection?.models||[]} onChange={updateModel}/></label>{draft.role!=='tts'?<><label>学习生成档位<select value={draft.config.study_generation_tier||'auto'} onChange={event=>mutate(current=>({...current,config:{...current.config,study_generation_tier:event.target.value}}))}><option value="auto">AUTO · 按模型与窗口判断</option><option value="lite">LITE · 兼容小模型</option><option value="full">FULL · 蓝图与独立审校</option></select><small>自动档会同时考虑参数量、上下文窗口与最大输出；人工覆盖仍受 token 安全预算约束。</small></label><small className="token-limit-summary" role="status">{tokenLimitText(tokenLimits)}</small><div className="provider-form-grid"><label>上下文窗口覆盖（tokens）<input type="number" min="1024" step="1" value={draft.config.context_window_tokens??''} placeholder="留空自动探测" onChange={event=>updateTokenOverride('context_window_tokens',event.target.value)}/><small>控制输入与输出总量；Ollama 不会自动强制使用理论最大值。</small></label><label>最大输出覆盖（tokens）<input type="number" min="128" step="1" value={draft.config.max_output_tokens??''} placeholder="留空自动推导" onChange={event=>updateTokenOverride('max_output_tokens',event.target.value)}/><small>未知时按运行窗口的 25% 推导，最多 4096。</small></label></div></>:null}{draft.role==='vlm'?<small className="capability-note">视觉能力：{lastMode==='deep'&&inspection?.activation_eligible?'已深度验证':inspection?.catalog_supported?'清单未声明，建议深度验证':'未知'}</small>:null}{draft.kind==='openai_tts'?<div className="voice-grid">{['host_a','host_b'].map(field=><label key={field}>{field.toUpperCase()} 音色<input value={draft.config[field]||''} placeholder="例如 alloy" onChange={event=>mutate(current=>({...current,config:{...current.config,[field]:event.target.value}}))}/></label>)}</div>:null}</>}</div>
      </div>
      <div className="provider-form-actions"><button onClick={()=>requestLeave('list')}>取消</button>{!currentProvider?.active?<button disabled={!canSave||Boolean(busy)} onClick={()=>void save(false)}>{busy==='save'?'保存中…':'保存为未启用'}</button>:null}<button className="primary" disabled={!canSave||!inspection?.activation_eligible||Boolean(busy)} onClick={()=>void save(true)}>{busy==='activate'?'验证并保存中…':currentProvider?.active?'验证并保存':'验证并启用'}</button></div>
    </div>}
    {leaveTarget?<ConfirmDialog title="放弃未保存的 Provider 配置？" description="当前表单中的修改尚未保存。" confirmLabel="放弃修改" onCancel={()=>setLeaveTarget(undefined)} onConfirm={async()=>confirmLeave()}/>:null}
  </Overlay>;
}
