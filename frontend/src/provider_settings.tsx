import {useId,useMemo,useState} from 'react';
import {ArrowLeft,FlaskConical,Plus,RefreshCw,Wifi} from 'lucide-react';
import type {ConfigurableProviderRole,HostVoiceMode,ImageProcessingPolicy,Provider,ProviderDraft,ProviderInspection,ProviderKind,ProviderModel,ProviderRoleState,TokenLimits,VoiceprintPersonOption} from './api';
import {ConfirmDialog,Overlay} from './ui';

const KINDS_BY_ROLE:Record<ConfigurableProviderRole,Array<{value:ProviderKind;label:string}>>={
  main:[{value:'openai',label:'OPENAI-COMPATIBLE'},{value:'ollama',label:'OLLAMA'}],
  vlm:[{value:'openai',label:'OPENAI-COMPATIBLE'},{value:'ollama',label:'OLLAMA'}],
  audio:[{value:'sandevistan_audio',label:'SANDEVISTAN AUDIO'}],
};

type DrawerView={mode:'list'}|{mode:'role';role:ConfigurableProviderRole}|{mode:'add'}|{mode:'edit';provider:Provider};
type InspectionMode='catalog'|'deep';
type SettingsProps={
  status:any;
  providers:Provider[];
  roles:ProviderRoleState[];
  imagePolicy:ImageProcessingPolicy;
  onClose:()=>void;
  onSave:(id:string,body:Record<string,any>)=>Promise<void>;
  onCreate:(body:Record<string,any>)=>Promise<void>;
  onInspect:(draft:ProviderDraft,mode:InspectionMode)=>Promise<ProviderInspection>;
  onSaveRole:(role:ConfigurableProviderRole,body:Record<string,any>)=>Promise<void>;
  onSaveImagePolicy:(policy:ImageProcessingPolicy)=>Promise<void>;
};

function newDraft(role:ConfigurableProviderRole='main'):ProviderDraft{const kind=KINDS_BY_ROLE[role][0].value;return{name:'',role,kind,base_url:'',model:'',api_key:'',config:role==='audio'?{auto_select:true,podcast_sequence_tts:true,asr_auto_select:true,asr_allow_device_fallback:true}:{}}}
function providerDraft(provider:Provider):ProviderDraft{return{provider_id:provider.id,name:provider.name,role:provider.role,kind:provider.kind,base_url:provider.base_url,model:provider.model,api_key:'',config:{...provider.config}}}

function audioVoiceError(draft:ProviderDraft,people:VoiceprintPersonOption[],supportsVoiceprint:boolean){
  const config=draft.config;
  const aMode=config.host_a_voice_mode||'preset',bMode=config.host_b_voice_mode||'preset';
  if((aMode==='voiceprint'||bMode==='voiceprint')&&!supportsVoiceprint)return '所选 TTS 模型不支持声纹克隆';
  for(const host of ['host_a','host_b'] as const){
    if((config[`${host}_voice_mode`]||'preset')!=='voiceprint')continue;
    const personId=config[`${host}_voiceprint_person_id`]||'';
    const person=people.find(item=>item.id===personId);
    if(!personId)return `${host.toUpperCase()} 尚未选择声纹人员`;
    if(!person?.latest_sample)return `${host.toUpperCase()} 的声纹人员没有可用样本`;
  }
  if(aMode===bMode&&aMode==='preset'&&config.host_a&&config.host_a===config.host_b)return 'Host A 与 Host B 不能使用同一个预置音色';
  if(aMode===bMode&&aMode==='voiceprint'&&config.host_a_voiceprint_person_id&&config.host_a_voiceprint_person_id===config.host_b_voiceprint_person_id)return 'Host A 与 Host B 不能使用同一个声纹人员';
  return '';
}

function ModelPicker({value,models,disabled,onChange}:{value:string;models:ProviderModel[];disabled?:boolean;onChange:(value:string)=>void}){
  const listId=useId();
  const[open,setOpen]=useState(false);
  const[manual,setManual]=useState(false);
  const[query,setQuery]=useState('');
  const normalized=query.trim().toLowerCase();
  const selected=models.find(model=>model.id===value);
  const visible=useMemo(()=>models.filter(model=>!normalized||model.id.toLowerCase().includes(normalized)||model.name.toLowerCase().includes(normalized)).slice(0,80),[models,normalized]);
  if(!models.length||manual)return <div className="model-picker manual-model-picker"><input disabled={disabled} value={value} placeholder="手动输入模型 ID" onChange={event=>onChange(event.target.value)}/>{models.length?<button type="button" disabled={disabled} onClick={()=>{setManual(false);setOpen(true);setQuery('')}}>返回模型清单</button>:null}</div>;
  return <div className="model-picker" onBlur={event=>{if(!event.currentTarget.contains(event.relatedTarget as Node|null)){setOpen(false);setQuery('')}}}>
    <button type="button" className="model-picker-trigger" role="combobox" aria-expanded={open} aria-controls={listId} disabled={disabled} onClick={()=>{setOpen(current=>!current);setQuery('')}}><span>{selected?.name||value||'选择模型'}</span><small>{selected?.id||(value?`自定义 · ${value}`:`${models.length} 个可用选项`)}</small></button>
    {open?<div className="model-picker-popover" data-escape-boundary><input autoFocus aria-label="搜索模型" value={query} placeholder="搜索模型名称或 ID" onChange={event=>setQuery(event.target.value)} onKeyDown={event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();setOpen(false);setQuery('')}else if(event.key==='Enter'&&visible[0]){event.preventDefault();onChange(visible[0].id);setOpen(false);setQuery('')}}}/><ul id={listId} role="listbox">{visible.map(model=><li key={model.id} role="option" aria-selected={model.id===value}><button type="button" disabled={model.installed===false} onMouseDown={event=>event.preventDefault()} onClick={()=>{onChange(model.id);setOpen(false);setQuery('')}}><span><b>{model.name}</b><small>{model.id}{model.installed===false?' · 未安装':''}</small></span>{model.id===value?<b aria-hidden="true">✓</b>:null}</button></li>)}{!visible.length?<li className="model-picker-empty">没有匹配的模型</li>:null}</ul><button type="button" className="model-picker-manual" onMouseDown={event=>event.preventDefault()} onClick={()=>{setManual(true);setOpen(false)}}>手动输入模型 ID</button></div>:null}
  </div>;
}

function InspectionPanel({inspection,mode,hasCatalog}:{inspection?:ProviderInspection;mode:InspectionMode;hasCatalog:boolean}){
  if(!inspection)return <p className="provider-inspection idle">{hasCatalog?'配置已修改；模型清单仍可用，保存时会重新验证。':'连接检查不会保存配置，也不会发送 Notebook 资料。'}</p>;
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
  const active=Boolean(provider.active),legacy=provider.role==='tts_only',online=active&&Boolean(status?.providers?.[provider.role]?.ok);
  const label=legacy?'兼容保留 · 不用于 Podcast':!active?'未启用':online?'活跃在线':'活跃异常';
  const limits=provider.capabilities?.token_limits as Partial<TokenLimits>|undefined;
  const roleLabel=provider.role==='audio'?'AUDIO':legacy?'TTS ONLY':provider.role.toUpperCase();
  return <div className="provider"><i className={!active?'inactive':online?'ok':''}/><span><b>{roleLabel} · {provider.name}</b><small>{label}{provider.has_api_key?' · KEY SAVED':''}</small>{provider.role==='audio'?<><small>TTS · {provider.model||'未选择模型'} · {String(provider.config?.compute_device||'未配置').toUpperCase()}</small><small>ASR · {provider.config?.asr_model||'未选择模型'} · {String(provider.config?.asr_compute_device||'未配置').toUpperCase()}</small></>:legacy?<small>{provider.kind} · {provider.model||'未选择模型'}</small>:<><small>{provider.kind} · {provider.model||'未选择模型'}</small><small>CTX/OUT · {tokenLimitText(limits)}</small></>}</span></div>;
}

type HostField='host_a'|'host_b';
function HostVoiceEditor({host,draft,voices,people,libraryStatus,libraryMessage,supportsVoiceprint,supportsInstruct,onConfig}:{host:HostField;draft:ProviderDraft;voices:Array<{id:string;native_language?:string}>;people:VoiceprintPersonOption[];libraryStatus?:string;libraryMessage?:string|null;supportsVoiceprint:boolean;supportsInstruct:boolean;onConfig:(updater:(config:Record<string,any>)=>Record<string,any>)=>void}){
  const other:HostField=host==='host_a'?'host_b':'host_a';
  const mode=(draft.config[`${host}_voice_mode`]||'preset') as HostVoiceMode;
  const personId=draft.config[`${host}_voiceprint_person_id`]||'';
  const person=people.find(item=>item.id===personId);
  const otherMode=(draft.config[`${other}_voice_mode`]||'preset') as HostVoiceMode;
  const setMode=(next:HostVoiceMode)=>onConfig(config=>({...config,[`${host}_voice_mode`]:next}));
  return <section className="host-voice-card" aria-label={`${host==='host_a'?'Host A':'Host B'} 音色`}>
    <div className="host-voice-head"><b>{host==='host_a'?'HOST A':'HOST B'}</b><div className="voice-mode-tabs"><button type="button" className={mode==='preset'?'active':''} aria-pressed={mode==='preset'} onClick={()=>setMode('preset')}>预置音色</button><button type="button" className={mode==='voiceprint'?'active':''} aria-pressed={mode==='voiceprint'} disabled={!supportsVoiceprint||libraryStatus!=='ready'} title={!supportsVoiceprint?'所选模型不支持声纹克隆':libraryMessage||undefined} onClick={()=>setMode('voiceprint')}>声纹克隆</button></div></div>
    {mode==='preset'?<label>音色{voices.length?<select value={draft.config[host]||''} onChange={event=>onConfig(config=>({...config,[host]:event.target.value}))}><option value="">选择音色</option>{voices.map(voice=><option key={voice.id} value={voice.id} disabled={otherMode==='preset'&&draft.config[other]===voice.id}>{voice.id} · {voice.native_language||'MULTI'}</option>)}</select>:<input value={draft.config[host]||''} placeholder="连接后选择或手填" onChange={event=>onConfig(config=>({...config,[host]:event.target.value}))}/>}</label>:<><label>声纹人员<select value={personId} onChange={event=>{const selected=people.find(item=>item.id===event.target.value);onConfig(config=>({...config,[`${host}_voiceprint_person_id`]:event.target.value,[`${host}_voiceprint_sample_id`]:selected?.latest_sample?.id||''}))}}><option value="">选择声纹库人员</option>{personId&&!person?<option value={personId}>已失效的人员配置</option>:null}{people.map(item=><option key={item.id} value={item.id} disabled={!item.latest_sample||(otherMode==='voiceprint'&&draft.config[`${other}_voiceprint_person_id`]===item.id)}>{item.name}{item.note?` · ${item.note}`:''}{item.latest_sample?'':` · 无可用样本`}</option>)}</select></label>{person?.latest_sample?<small>已锁定最新样本 · {person.latest_sample.language}{person.latest_sample.duration?` · ${person.latest_sample.duration.toFixed(1)} 秒`:''}{Number(person.latest_sample.duration)>15?' · 合成时截断至 15 秒以内':''}</small>:<small>{libraryMessage||'只能选择声纹库中已有且具备可用样本的人员。'}</small>}</>}
    {mode==='preset'&&supportsInstruct?<label>基础表达风格<textarea value={draft.config[`${host}_instruct`]||''} onChange={event=>onConfig(config=>({...config,[`${host}_instruct`]:event.target.value}))}/><small>整集会自动追加稳定语速、音高和情绪范围约束。</small></label>:mode==='voiceprint'?<small>克隆模式由固定参考样本控制声线，上游不支持额外语气指令。</small>:null}
  </section>;
}

export function SettingsDrawer({status,providers,roles,imagePolicy,onClose,onSave,onCreate,onInspect,onSaveRole,onSaveImagePolicy}:SettingsProps){
  const[view,setView]=useState<DrawerView>({mode:'list'});
  const[draft,setDraft]=useState<ProviderDraft>(newDraft);
  const[catalog,setCatalog]=useState<ProviderInspection>();
  const[inspection,setInspection]=useState<ProviderInspection>();
  const[lastMode,setLastMode]=useState<InspectionMode>('catalog');
  const[busy,setBusy]=useState('');
  const[dirty,setDirty]=useState(false);
  const[leaveTarget,setLeaveTarget]=useState<'list'|'close'>();
  const[cardResults,setCardResults]=useState<Record<string,ProviderInspection>>({});

  const isForm=view.mode==='add'||view.mode==='edit';
  const currentProvider=view.mode==='edit'?view.provider:undefined;
  const selectedModel=catalog?.models.find(model=>model.id===draft.model);
  const devices=selectedModel?.devices||[];
  const asrCapability=catalog?.capabilities?.asr||(currentProvider?.capabilities?.asr||{});
  const asrModels=Array.isArray(asrCapability?.models)?asrCapability.models as ProviderModel[]:[];
  const selectedAsrModel=asrModels.find(model=>model.id===draft.config.asr_model);
  const asrDevices=selectedAsrModel?.devices||[];
  const voices=Array.isArray(catalog?.capabilities?.voices)?catalog.capabilities.voices:[];
  const controls=selectedModel?.controls||{};
  const supportsInstruct=(controls.instruction_voice_modes||[]).includes('preset');
  const supportsVoiceprint=(selectedModel?.voice_modes||[]).includes('voiceprint');
  const voiceprintLibrary=catalog?.voiceprint_library;
  const voiceprintPeople=voiceprintLibrary?.people||[];
  const tokenLimits=(inspection?.capabilities?.token_limits||(currentProvider?.model===draft.model?currentProvider?.capabilities?.token_limits:undefined)) as Partial<TokenLimits>|undefined;
  const canSave=Boolean(draft.name.trim()&&draft.base_url.trim());
  const hostVoiceError=draft.kind==='sandevistan_audio'?audioVoiceError(draft,voiceprintPeople,supportsVoiceprint):'';
  const canActivate=Boolean(canSave&&draft.model.trim()&&!hostVoiceError);

  function mutate(updater:(current:ProviderDraft)=>ProviderDraft,invalidates=true,clearsCatalog=false){
    setDraft(current=>updater(current));setDirty(true);if(invalidates)setInspection(undefined);if(clearsCatalog)setCatalog(undefined);
  }
  function beginAdd(role:ConfigurableProviderRole){setView({mode:'add'});setDraft(newDraft(role));setCatalog(undefined);setInspection(undefined);setLastMode('catalog');setDirty(false)}
  function beginEdit(provider:Provider){const next=providerDraft(provider);setView({mode:'edit',provider});setDraft(next);setCatalog(undefined);setInspection(undefined);setLastMode('catalog');setDirty(false);void inspect(next,'catalog',true)}
  function requestLeave(target:'list'|'close'){if(isForm&&dirty)setLeaveTarget(target);else if(target==='close')onClose();else setView({mode:'role',role:draft.role as ConfigurableProviderRole})}
  function confirmLeave(){const target=leaveTarget;setLeaveTarget(undefined);setDirty(false);if(target==='close')onClose();else setView({mode:'role',role:draft.role as ConfigurableProviderRole})}

  async function inspect(target:ProviderDraft,mode:InspectionMode,quiet=false){
    if(!target.base_url.trim())return;
    setBusy(mode);setLastMode(mode);
    try{
      const result=await onInspect(target,mode);setInspection(result);if(result.connection_ok)setCatalog(result);
      if(target.kind==='sandevistan_audio'){
        const ttsRecommendation=result.recommended;
        const asrRecommendation=result.capabilities?.asr?.recommended;
        setDraft(current=>({
          ...current,
          model:current.config.auto_select&&ttsRecommendation?.model?ttsRecommendation.model:current.model,
          config:{
            ...current.config,
            ...(result.resolved_audio_config||{}),
            ...(current.config.auto_select&&ttsRecommendation?.compute_device?{compute_device:ttsRecommendation.compute_device}:{}),
            ...(current.config.asr_auto_select!==false&&asrRecommendation?.model?{asr_auto_select:true,asr_model:asrRecommendation.model,asr_compute_device:asrRecommendation.compute_device}:{}),
          },
        }));
      }
      if(!quiet)setDirty(true);
    }catch{/* The application-level toast already reports request failures. */}finally{setBusy('')}
  }

  async function inspectCard(provider:Provider){
    setBusy(`card-${provider.id}`);
    try{const result=await onInspect(providerDraft(provider),'catalog');setCardResults(current=>({...current,[provider.id]:result}))}catch{/* Keep the existing card state. */}finally{setBusy('')}
  }

  function payload(active:boolean){
    const listed=Boolean(catalog?.catalog_supported&&catalog.models.some(model=>model.id===draft.model));
    const validationMode=listed?'catalog':'deep';
    const body:Record<string,any>={name:draft.name.trim(),base_url:draft.base_url.trim(),model:draft.model.trim(),config:draft.config,active,validation_mode:validationMode};
    if(draft.api_key)body.api_key=draft.api_key;
    if(view.mode==='add'){body.role=draft.role;body.kind=draft.kind;body.capabilities={}}
    return body;
  }
  async function save(active:boolean){
    setBusy(active?'activate':'save');
    try{
      if(currentProvider)await onSave(currentProvider.id,payload(active));else await onCreate(payload(active));
      setDirty(false);setCatalog(undefined);setInspection(undefined);setView({mode:'role',role:draft.role as ConfigurableProviderRole});
    }catch{/* Keep the draft available for correction. */}finally{setBusy('')}
  }

  function updateRole(role:ConfigurableProviderRole){const kind=KINDS_BY_ROLE[role][0].value;mutate(current=>({...current,role,kind,model:'',config:kind==='sandevistan_audio'?{auto_select:true,podcast_sequence_tts:true,asr_auto_select:true,asr_allow_device_fallback:true}:{}}),true,true)}
  function updateModel(model:string){mutate(current=>{const match=catalog?.models.find(item=>item.id===model),available=match?.devices?.find(device=>device.available&&device.id==='gpu')||match?.devices?.find(device=>device.available);return{...current,model,config:{...current.config,...(available?{auto_select:false,compute_device:available.id}:{})}}})}
  function updateAsrModel(model:string){mutate(current=>{const match=asrModels.find(item=>item.id===model),available=match?.devices?.find(device=>device.available&&device.id==='gpu')||match?.devices?.find(device=>device.available);return{...current,config:{...current.config,asr_auto_select:false,asr_model:model,...(available?{asr_compute_device:available.id}:{})}}})}
  function updateNumericOverride(field:'temperature'|'context_window_tokens'|'max_output_tokens',value:string){mutate(current=>{const config={...current.config};if(value)config[field]=Number(value);else delete config[field];return{...current,config}})}
  function moveImageProcessor(index:number,direction:-1|1){const processors=[...imagePolicy.processors],target=index+direction;if(target<0||target>=processors.length)return;[processors[index],processors[target]]=[processors[target],processors[index]];void onSaveImagePolicy({mode:'process',processors})}

  return <Overlay className={`settings ${isForm?'settings-form-view':''}`} label="Provider 配置" onClose={()=>requestLeave('close')} closeOnBackdrop={false}>
    <button className="drawer-close" data-autofocus onClick={()=>requestLeave('close')}>关闭 ×</button>
    <span>SYSTEM CONFIG</span>
    <h2>{isForm?(view.mode==='add'?'添加 Provider':`编辑 ${currentProvider?.name}`):'Provider 配置'}</h2>
    {view.mode==='list'?<>
      <p>按职责管理能力。MAIN 始终启用；VLM 与 AUDIO 可暂停，已选择的配置会保留。</p>
      <div className="provider-role-grid">{(['main','vlm','audio'] as const).map(role=>{const state=roles.find(item=>item.role===role);const selected=providers.find(item=>item.id===state?.selected_provider_id);return <section className={`provider-role-card ${state?.enabled?'enabled':'disabled'}`} key={role}><div><span>{role.toUpperCase()}</span><h3>{role==='main'?'核心推理':role==='vlm'?'图片理解':'语音生成与验收'}</h3><p>{selected?selected.name:'尚未选择 Provider'}</p></div>{role==='main'?<b className="role-required">必需</b>:<label className="role-switch"><input type="checkbox" checked={Boolean(state?.enabled)} disabled={Boolean(busy)||!selected} onChange={event=>{setBusy(`role-${role}`);void onSaveRole(role,{enabled:event.target.checked,validation_mode:event.target.checked?'deep':'catalog'}).finally(()=>setBusy(''))}}/><span>{state?.enabled?'已启用':'已暂停'}</span></label>}<button onClick={()=>setView({mode:'role',role})}>管理 {role.toUpperCase()}</button></section>})}</div>
      <section className="image-policy-card"><span>IMAGE PIPELINE</span><h3>新资料图片处理</h3><label className="role-switch"><input type="checkbox" checked={imagePolicy.mode==='process'} onChange={event=>void onSaveImagePolicy({...imagePolicy,mode:event.target.checked?'process':'off'})}/><span>{imagePolicy.mode==='process'?'已启用':'已关闭'}</span></label><div className="image-policy-order">{imagePolicy.processors.map((processor,index)=><span key={processor}><b>{processor.toUpperCase()}</b><button aria-label={`上移 ${processor.toUpperCase()}`} disabled={index===0} onClick={()=>moveImageProcessor(index,-1)}>↑</button><button aria-label={`下移 ${processor.toUpperCase()}`} disabled={index===imagePolicy.processors.length-1} onClick={()=>moveImageProcessor(index,1)}>↓</button><button aria-label={`移除 ${processor.toUpperCase()}`} disabled={imagePolicy.processors.length===1} onClick={()=>void onSaveImagePolicy({mode:'process',processors:imagePolicy.processors.filter(item=>item!==processor)})}>×</button></span>)}{(['vlm','main','ocr'] as const).filter(processor=>!imagePolicy.processors.includes(processor)).map(processor=><button key={processor} onClick={()=>void onSaveImagePolicy({mode:'process',processors:[...imagePolicy.processors,processor]})}>+ {processor.toUpperCase()}</button>)}</div><small>{imagePolicy.mode==='off'?'不处理图片':imagePolicy.processors.map(item=>item.toUpperCase()).join(' → ')} · 成功即停止 · 仅影响新上传</small></section>
      <h3>项目工具</h3>{['ffmpeg','libreoffice'].map(name=>{const tool=status?.tools?.[name];return <div className="provider" key={name}><i className={tool?.available?'ok':''}/><span><b>{name.toUpperCase()}</b><small>{tool?.version||'未安装'} · {tool?.scope||'missing'}</small></span></div>})}
      <h3>检索</h3><div className="provider"><i className={status?.retrieval?.embedding_mode==='sentence-transformers'?'ok':''}/><span><b>{status?.retrieval?.embedding_mode||'正在读取'}</b><small>{status?.retrieval?.model||'状态将在后台更新'}</small></span></div>
      <p>Provider 凭据在项目内加密保存。启用云端 Provider 时，选中的资料上下文会发送给该服务。</p>
    </>:view.mode==='role'?<div className="provider-role-view"><button className="provider-back" onClick={()=>setView({mode:'list'})}><ArrowLeft size={15}/> 返回角色概览</button><span>{view.role.toUpperCase()} PROVIDERS</span><h3>{view.role==='main'?'核心推理配置':view.role==='vlm'?'视觉模型配置':'语音服务配置'}</h3>{providers.filter(provider=>provider.role===view.role).map(provider=><section className="provider-card" key={provider.id}><ProviderStatus provider={provider} status={status}/><div className="provider-actions"><button onClick={()=>beginEdit(provider)}>编辑</button><button disabled={Boolean(busy)} onClick={()=>void inspectCard(provider)}>{busy===`card-${provider.id}`?'检查中…':'检查'}</button>{cardResults[provider.id]?<small className={cardResults[provider.id].status} role="status">{cardResults[provider.id].error?.message||cardResults[provider.id].warning||`ONLINE · ${cardResults[provider.id].latency_ms} ms`}</small>:null}</div></section>)}<button className="add-provider" onClick={()=>beginAdd(view.role)}><Plus size={14}/> 添加 {view.role.toUpperCase()} Provider</button></div>:<div className="provider-editor">
      <button className="provider-back" onClick={()=>requestLeave('list')}><ArrowLeft size={15}/> 返回角色配置</button>
      <div className="provider-form">
        <div className="provider-form-section"><h3>基本信息</h3><label>名称<input value={draft.name} onChange={event=>mutate(current=>({...current,name:event.target.value}),false)}/></label><div className="provider-form-grid"><label>角色<select value={draft.role} disabled><option value="main">MAIN</option><option value="vlm">VLM</option><option value="audio">AUDIO</option>{draft.role==='tts_only'?<option value="tts_only">TTS ONLY</option>:null}</select></label><label>类型<select value={draft.kind} disabled={view.mode==='edit'} onChange={event=>mutate(current=>({...current,kind:event.target.value as ProviderKind,model:'',config:event.target.value==='sandevistan_audio'?{auto_select:true,podcast_sequence_tts:true,asr_auto_select:true,asr_allow_device_fallback:true}:{}}),true,true)}>{draft.role==='tts_only'?<option value="openai_tts">OPENAI TTS · LEGACY</option>:KINDS_BY_ROLE[draft.role as ConfigurableProviderRole].map(option=><option key={option.value} value={option.value}>{option.label}</option>)}</select></label></div></div>
        <div className="provider-form-section"><h3>连接</h3><label>服务地址<input value={draft.base_url} placeholder={draft.kind==='ollama'?'http://localhost:11434':draft.kind==='sandevistan_audio'?'http://localhost:20810':'https://api.example.com/v1'} onChange={event=>mutate(current=>({...current,base_url:event.target.value}),true,true)}/><small>可直接粘贴带 /v1 或 /api 的地址，保存时会自动规范化。</small></label><label>API Key<input type="password" autoComplete="new-password" value={draft.api_key||''} placeholder={currentProvider?.has_api_key?'已安全保存；留空保持不变':'可选'} onChange={event=>mutate(current=>({...current,api_key:event.target.value}),true,true)}/></label><div className="provider-check-actions"><button disabled={!draft.base_url.trim()||Boolean(busy)} onClick={()=>void inspect(draft,'catalog')}><Wifi size={14}/>{busy==='catalog'?'正在连接…':'连接并读取模型'}</button><button disabled={!draft.model.trim()||Boolean(busy)||draft.role==='tts_only'} onClick={()=>void inspect(draft,'deep')}><FlaskConical size={14}/>{busy==='deep'?'正在验证…':'深度验证'}</button><button aria-label="刷新能力清单" disabled={!draft.base_url.trim()||Boolean(busy)} onClick={()=>void inspect(draft,'catalog')}><RefreshCw size={14}/></button></div><small className="provider-cost-note">深度验证使用固定测试内容，不含 Notebook 资料；AUDIO 会分别验证两位主持人并完成一次短 TTS→ASR 闭环。</small><InspectionPanel inspection={inspection} mode={lastMode} hasCatalog={Boolean(catalog)}/></div>
        <div className="provider-form-section">
          <h3>模型与能力</h3>
          {draft.kind==='sandevistan_audio'?<>
            <h4>TTS 合成</h4>
            <label className="provider-check"><input type="checkbox" checked={Boolean(draft.config.auto_select)} onChange={event=>mutate(current=>({...current,config:{...current.config,auto_select:event.target.checked}}))}/> 使用服务推荐的 TTS 模型与设备</label>
            {draft.config.auto_select&&catalog?.recommended?<small className="capability-note">当前解析为 {catalog.recommended.model} · {(catalog.recommended.compute_device||'').toUpperCase()}（{catalog.recommended.reason==='service_default'?'服务默认配置':catalog.recommended.reason==='preserve_custom_instructions'?'保留自定义表达指令':'可用模型回退'}）</small>:null}
            <label>TTS 模型<ModelPicker value={draft.model} models={catalog?.models||[]} disabled={Boolean(draft.config.auto_select)} onChange={updateModel}/></label>
            <label>TTS 设备<select value={draft.config.compute_device||''} disabled={Boolean(draft.config.auto_select)||!devices.length} onChange={event=>mutate(current=>({...current,config:{...current.config,auto_select:false,compute_device:event.target.value}}))}><option value="">连接后选择设备</option>{devices.map(device=><option key={device.id} value={device.id} disabled={!device.available}>{device.id.toUpperCase()} · {device.available?device.precision:device.reason||'不可用'}</option>)}</select></label>
            <div className="host-voice-grid">{(['host_a','host_b'] as const).map(host=><HostVoiceEditor key={host} host={host} draft={draft} voices={voices} people={voiceprintPeople} libraryStatus={voiceprintLibrary?.status} libraryMessage={voiceprintLibrary?.message} supportsVoiceprint={supportsVoiceprint} supportsInstruct={supportsInstruct} onConfig={updater=>mutate(current=>({...current,config:updater(current.config)}))}/>)}</div>
            {hostVoiceError?<small className="provider-config-error" role="alert">{hostVoiceError}</small>:null}
            <label className="provider-check"><input type="checkbox" checked={draft.config.allow_device_fallback!==false} onChange={event=>mutate(current=>({...current,config:{...current.config,allow_device_fallback:event.target.checked}}))}/> TTS GPU 失败时回退同模型 CPU</label>
            {catalog?.capabilities?.sequence_jobs?.supported?<label className="provider-check"><input type="checkbox" checked={draft.config.podcast_sequence_tts!==false} onChange={event=>mutate(current=>({...current,config:{...current.config,podcast_sequence_tts:event.target.checked}}))}/> Podcast 使用批量合成与安全并行加速</label>:null}
            <h4>ASR 验收</h4>
            <label className="provider-check"><input type="checkbox" checked={draft.config.asr_auto_select!==false} onChange={event=>mutate(current=>({...current,config:{...current.config,asr_auto_select:event.target.checked}}))}/> 使用服务推荐的 ASR 模型与设备</label>
            <label>ASR 模型<ModelPicker value={draft.config.asr_model||''} models={asrModels} disabled={draft.config.asr_auto_select!==false} onChange={updateAsrModel}/></label>
            <label>ASR 设备<select value={draft.config.asr_compute_device||''} disabled={draft.config.asr_auto_select!==false||!asrDevices.length} onChange={event=>mutate(current=>({...current,config:{...current.config,asr_auto_select:false,asr_compute_device:event.target.value}}))}><option value="">连接后选择设备</option>{asrDevices.map(device=><option key={device.id} value={device.id} disabled={!device.available}>{device.id.toUpperCase()} · {device.available?device.precision:device.reason||'不可用'}</option>)}</select></label>
            <label className="provider-check"><input type="checkbox" checked={draft.config.asr_allow_device_fallback!==false} onChange={event=>mutate(current=>({...current,config:{...current.config,asr_allow_device_fallback:event.target.checked}}))}/> ASR GPU 失败时回退同模型 CPU</label>
            <small className="capability-note">固定验收约束：双说话人分离、时间对齐、语言跟随 Podcast；CER/WER 与静音门槛不可在 Provider 中放宽。</small>
          </>:draft.role==='tts_only'?<>
            <small className="capability-note">这是兼容保留的 TTS-only 配置；因为不提供必需的 ASR，不能启用或用于 Podcast。</small>
            <label>TTS 模型<ModelPicker value={draft.model} models={catalog?.models||[]} onChange={updateModel}/></label>
            <div className="voice-grid">{['host_a','host_b'].map(field=><label key={field}>{field.toUpperCase()} 音色<input value={draft.config[field]||''} placeholder="例如 alloy" onChange={event=>mutate(current=>({...current,config:{...current.config,[field]:event.target.value}}))}/></label>)}</div>
          </>:<>
            <label>模型<ModelPicker value={draft.model} models={catalog?.models||[]} onChange={updateModel}/></label>
            <div className="provider-form-grid"><label>学习生成档位<select value={draft.config.study_generation_tier||'auto'} onChange={event=>mutate(current=>({...current,config:{...current.config,study_generation_tier:event.target.value as 'auto'|'lite'|'full'}}))}><option value="auto">AUTO · 按模型与窗口判断</option><option value="lite">LITE · 兼容小模型</option><option value="full">FULL · 蓝图与独立审校</option></select><small>自动档会同时考虑参数量、上下文窗口与最大输出；人工覆盖仍受 token 安全预算约束。</small></label><label>Temperature 覆盖<input type="number" min="0" max="2" step="0.01" value={draft.config.temperature??''} placeholder="留空按任务设置" onChange={event=>updateNumericOverride('temperature',event.target.value)}/><small>填写后覆盖该 Provider 的所有任务温度；留空时使用各任务默认值。</small></label></div>
            <small className="token-limit-summary" role="status">{tokenLimitText(tokenLimits)}</small>
            <div className="provider-form-grid"><label>上下文窗口覆盖（tokens）<input type="number" min="1024" step="1" value={draft.config.context_window_tokens??''} placeholder="留空自动探测" onChange={event=>updateNumericOverride('context_window_tokens',event.target.value)}/><small>控制输入与输出总量；Ollama 不会自动强制使用理论最大值。</small></label><label>最大输出覆盖（tokens）<input type="number" min="128" step="1" value={draft.config.max_output_tokens??''} placeholder="留空自动推导" onChange={event=>updateNumericOverride('max_output_tokens',event.target.value)}/><small>未知时按运行窗口的 25% 推导，最多 4096。</small></label></div>
            {draft.role==='vlm'?<small className="capability-note">视觉能力：{lastMode==='deep'&&inspection?.activation_eligible?'已深度验证':inspection?.catalog_supported?'清单未声明，建议深度验证':'未知'}</small>:null}
          </>}
        </div>
      </div>
      <div className="provider-form-actions"><button onClick={()=>requestLeave('list')}>取消</button>{!currentProvider?.active?<button disabled={!canSave||Boolean(busy)} onClick={()=>void save(false)}>{busy==='save'?'保存中…':'保存为未启用'}</button>:null}{draft.role!=='tts_only'?<button className="primary" disabled={!canActivate||Boolean(busy)} onClick={()=>void save(true)}>{busy==='activate'?'验证并保存中…':currentProvider?.active?'验证并保存':'验证并启用'}</button>:null}</div>
    </div>}
    {leaveTarget?<ConfirmDialog title="放弃未保存的 Provider 配置？" description="当前表单中的修改尚未保存。" confirmLabel="放弃修改" onCancel={()=>setLeaveTarget(undefined)} onConfirm={async()=>confirmLeave()}/>:null}
  </Overlay>;
}
