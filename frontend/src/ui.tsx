import {type ReactNode,useEffect,useId,useMemo,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {previewTask,recheck,reviewHistory,notebookUsage,type Citation,type Usage,type TaskPreview,type ReviewReport} from './api';

const FOCUSABLE='button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
const overlayStack:string[]=[];

type OverlayProps={
  children:ReactNode;
  className?:string;
  label:string;
  layer?:'base'|'nested';
  onClose:()=>void;
  closeOnBackdrop?:boolean;
  closeOnEscape?:boolean;
};

export function Overlay({children,className='',label,layer='base',onClose,closeOnBackdrop=true,closeOnEscape=true}:OverlayProps){
  const panelRef=useRef<HTMLElement>(null);
  const openerRef=useRef<HTMLElement|null>(null);
  const onCloseRef=useRef(onClose);
  onCloseRef.current=onClose;
  const overlayId=useId();
  useEffect(()=>{
    openerRef.current=document.activeElement instanceof HTMLElement?document.activeElement:null;
    overlayStack.push(overlayId);
    document.body.classList.add('overlay-open');
    const frame=requestAnimationFrame(()=>{
      const preferred=panelRef.current?.querySelector<HTMLElement>('[data-autofocus]');
      (preferred||panelRef.current?.querySelector<HTMLElement>(FOCUSABLE)||panelRef.current)?.focus();
    });
    const onKeyDown=(event:KeyboardEvent)=>{
      if(overlayStack.at(-1)!==overlayId)return;
      if(event.key==='Escape'&&event.target instanceof Element&&event.target.closest('[data-escape-boundary]'))return;
      if(event.key==='Escape'&&closeOnEscape){event.preventDefault();event.stopImmediatePropagation();onCloseRef.current();return}
      if(event.key!=='Tab'||!panelRef.current)return;
      const focusable=[...panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(element=>element.offsetParent!==null);
      if(!focusable.length){event.preventDefault();panelRef.current.focus();return}
      const first=focusable[0],last=focusable.at(-1)!;
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus()}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus()}
    };
    document.addEventListener('keydown',onKeyDown,true);
    return()=>{
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown',onKeyDown,true);
      const index=overlayStack.lastIndexOf(overlayId);if(index>=0)overlayStack.splice(index,1);
      if(!overlayStack.length)document.body.classList.remove('overlay-open');
      const opener=openerRef.current;if(opener?.isConnected)requestAnimationFrame(()=>opener.focus());
    };
  },[closeOnEscape,overlayId]);
  return createPortal(
    <div className={`drawer-backdrop overlay-layer-${layer}`} onMouseDown={event=>{if(closeOnBackdrop&&event.target===event.currentTarget)onClose()}}>
      <aside ref={panelRef} className={`drawer ${className}`} role="dialog" aria-modal="true" aria-label={label} tabIndex={-1}>{children}</aside>
    </div>,document.body,
  );
}

type ConfirmDialogProps={
  title:string;
  description:string;
  confirmLabel:string;
  onCancel:()=>void;
  onConfirm:()=>Promise<void>;
  requireText?:string;
  items?:string[];
  danger?:boolean;
};

export function ConfirmDialog({title,description,confirmLabel,onCancel,onConfirm,requireText,items,danger=true}:ConfirmDialogProps){
  const[value,setValue]=useState('');
  const[busy,setBusy]=useState(false);
  async function submit(){
    if(busy||(requireText!==undefined&&value!==requireText))return;
    setBusy(true);try{await onConfirm()}catch{/* The caller already exposes the error and the dialog stays open. */}finally{setBusy(false)}
  }
  return <Overlay className="confirm-modal" label={title} onClose={onCancel} closeOnBackdrop={false}>
    <span>{danger?'DANGER // CONFIRM ACTION':'CONFIRM ACTION'}</span>
    <h2>{title}</h2>
    <p>{description}</p>
    {items?.length?<ul className="confirm-items">{items.map((item,index)=><li key={`${item}-${index}`}>{item}</li>)}</ul>:null}
    {requireText!==undefined?<label>输入下方内容确认
      <input data-autofocus value={value} onChange={event=>setValue(event.target.value)} placeholder={requireText}/>
    </label>:null}
    <div className="confirm-actions">
      <button onClick={onCancel} disabled={busy}>取消</button>
      <button data-autofocus={requireText===undefined||undefined} className={danger?'danger-action':'primary'} disabled={busy||(requireText!==undefined&&value!==requireText)} onClick={submit}>{busy?'处理中…':confirmLabel}</button>
    </div>
  </Overlay>;
}

function inlineParts(text:string,citations:Map<string,Citation>,onCitation?:(citation:Citation)=>void){
  return text.split(/(\[S\d+\])/g).map((part,index)=>{
    const match=part.match(/^\[(S\d+)\]$/);const citation=match?citations.get(match[1]):undefined;
    return citation&&onCitation?<button className="citation-inline" key={`${part}-${index}`} onClick={()=>onCitation(citation)} aria-label={`查看引用 ${citation.id}`}>{part}</button>:part;
  });
}

type Block={kind:'heading'|'paragraph'|'list';lines:string[]};
function parseBlocks(content:string){
  const blocks:Block[]=[];
  for(const raw of content.replace(/\r/g,'').split('\n')){
    const line=raw.trim();if(!line)continue;
    if(/^#{1,4}\s+/.test(line)){blocks.push({kind:'heading',lines:[line.replace(/^#{1,4}\s+/,'')]});continue}
    if(/^[-*]\s+/.test(line)){
      const text=line.replace(/^[-*]\s+/,'');const last=blocks.at(-1);
      if(last?.kind==='list')last.lines.push(text);else blocks.push({kind:'list',lines:[text]});continue;
    }
    blocks.push({kind:'paragraph',lines:[line]});
  }
  return blocks;
}

export function RichText({content,citations=[],onCitation}:{content:string;citations?:Citation[];onCitation?:(citation:Citation)=>void}){
  const citationMap=useMemo(()=>new Map(citations.map(citation=>[citation.id,citation])),[citations]);
  const blocks=useMemo(()=>parseBlocks(content),[content]);
  return <div className="rich-text">{blocks.map((block,index)=>{
    if(block.kind==='heading')return <h3 key={index}>{inlineParts(block.lines[0],citationMap,onCitation)}</h3>;
    if(block.kind==='list')return <ul key={index}>{block.lines.map((line,lineIndex)=><li key={lineIndex}>{inlineParts(line,citationMap,onCitation)}</li>)}</ul>;
    return <p key={index}>{inlineParts(block.lines[0],citationMap,onCitation)}</p>;
  })}</div>;
}

export function CitationIndex({citations,onCitation}:{citations:Citation[];onCitation:(citation:Citation)=>void}){
  if(!citations.length)return null;
  return <details className="citation-index"><summary>引用索引 · {citations.length}</summary><div className="citations">{citations.map(citation=><button className="citation" key={citation.id+citation.source_id} onClick={()=>onCitation(citation)}>[{citation.id}] <span>{citation.filename}</span></button>)}</div></details>;
}

export function UsageDetails({value,legacy}:{value?:Usage;legacy?:Record<string,any>}){
  const known=value?.recorded;
  const total=known?(value.input_tokens+value.output_tokens):legacy?.actual_total_tokens;
  return <details className="usage-details"><summary>本次用量 · {total!=null?`${Number(total).toLocaleString()} tokens${known&&value.complete?'':'（已知部分）'}`:'无完整计量记录'}</summary>
    {known?<><p>输入 {value.input_tokens.toLocaleString()} · 输出 {value.output_tokens.toLocaleString()} · {value.calls} 次模型请求</p><p>其中推理 {value.reasoning_tokens.toLocaleString()} · 缓存输入 {value.cached_tokens.toLocaleString()}（已计入上述总量；未单列返回时不表示没有发生）</p>{value.unknown_calls>0?<p>{value.unknown_calls} 次请求未返回完整计量；对应输入估算 {value.estimated_input_tokens.toLocaleString()}。失败或超时不代表免费；预算预留值不作为实际用量。</p>:null}<ul>{Object.entries(value.stages).map(([name,stage])=><li key={name}>{({generation:'生成',summary:'摘要',summary_grounding_audit:'原文核查',aggregate_audit:'题卡核查',episode_audit:'播客核查',context_prepare:'资料预读',act_draft:'章节写作'} as Record<string,string>)[name]||name}：{stage.calls} 次 · 已知 {(stage.input_tokens+stage.output_tokens).toLocaleString()} tokens{stage.unknown_calls?` · ${stage.unknown_calls} 次计量不完整`:''}</li>)}</ul>{value.media?.length?<p>语音尝试单列：{value.media.length} 次 · 合成输入 {value.media.reduce((sum,item)=>sum+item.submitted_chars,0).toLocaleString()} 字符 · 已测量识别音频 {Math.round(value.media.reduce((sum,item)=>sum+(item.audio_seconds||0),0))} 秒。请求量不等于服务收费量；重试和失败仍保留。</p>:null}<details><summary>实际请求配置</summary>{value.requests_truncated?<small>仅展示最近 100 次请求；汇总包含全部已记录请求。</small>:null}{value.requests?.map(request=><p key={request.id}>{request.model} · {request.state} · {JSON.stringify(request.requested_controls)}（请求值，不保证服务采用）</p>)}</details><small>{value.scope}</small></>:<p>历史记录可能不完整。{legacy?.actual_total_tokens?`已记录输入 ${legacy.actual_prompt_tokens||0}、输出 ${legacy.actual_completion_tokens||0}；未记录的请求不按零消耗计算。`:'无法追溯的消耗不会补填为零。'}</p>}
  </details>
}

export function TaskEstimate({notebookId,kind,sourceIds,question='',count,minutes,limit,onLimit}:{notebookId?:string;kind:string;sourceIds:string[];question?:string;count?:number;minutes?:number;limit?:number;onLimit:(value:number|undefined)=>void}){
  const[value,setValue]=useState<TaskPreview>();const[error,setError]=useState('');
  const body=JSON.stringify({kind,source_ids:sourceIds,question,count,minutes,token_limit:validTokenLimit(limit)?limit:undefined});
  useEffect(()=>{setValue(undefined);setError('');if(!notebookId||!sourceIds.length)return;const controller=new AbortController();const timer=setTimeout(()=>{void previewTask(notebookId,JSON.parse(body),controller.signal).then(result=>{if(!controller.signal.aborted)setValue(result)}).catch(error=>{if(!controller.signal.aborted)setError(error instanceof Error?error.message:'预估暂不可用')})},400);return()=>{clearTimeout(timer);controller.abort()}},[notebookId,body]);
  return <details className="task-estimate"><summary>本次范围与用量 · {sourceIds.length} 份资料{value?` · 约 ${value.calls_range.join('–')} 次调用`:''}</summary>{value?<><p>选材输入估算约 {value.estimated_input_tokens.toLocaleString()} tokens；单次输出最多 {value.output_limit.toLocaleString()}。{value.token_limit?`任务预算上限 ${value.token_limit.toLocaleString()} tokens。`:''}</p>{value.weekly_budget?<p>本周剩余可用估算：{value.weekly_budget.global_usage.remaining_tokens===null?'未设 token 额度':`${value.weekly_budget.global_usage.remaining_tokens.toLocaleString()} tokens`} · {value.weekly_budget.global_usage.remaining_calls===null?'未设调用额度':`${value.weekly_budget.global_usage.remaining_calls} 次请求`}。{value.weekly_budget.global_usage.limit?.mode==='block'?'不足时停止新请求。':'仅提醒，可继续使用。'}</p>:null}<small>{value.notice} 当前{value.strategy==='balanced'?'均衡':'保守'}策略；有效窗口 {value.context_tokens.toLocaleString()}{value.context_source==='fallback'?'（服务未报告，采用兼容值）':''}。</small></>:<p>{error||(!sourceIds.length?'先选择已就绪资料。':'正在本地估算…')}</p>}<label>本次模型用量上限（可选）<input aria-label="本次模型用量上限" type="number" min="1024" max="4194304" step="1" value={limit??''} placeholder="留空按任务规划" onChange={event=>onLimit(event.target.value?Number(event.target.value):undefined)}/></label>{!validTokenLimit(limit)?<p role="alert">请输入 1024–4194304 之间的整数。</p>:null}<small>包括生成、审校和恢复；不足时保留可用内容并停止后续请求。语音字符和时长另计，上限不是货币账单承诺。</small></details>
}
export function validTokenLimit(limit?:number){return limit===undefined||Number.isInteger(limit)&&limit>=1024&&limit<=4194304}

export function ReviewControl({targetType,targetId}:{targetType:'artifact'|'message';targetId?:string}){
  const[reports,setReports]=useState<ReviewReport[]>([]);const[busy,setBusy]=useState(false);const[error,setError]=useState('');const[limit,setLimit]=useState<number>();
  if(!targetId||targetId.startsWith('local-'))return null;
  async function load(){try{setReports(await reviewHistory(targetType,targetId!));setError('')}catch(error){setError(error instanceof Error?error.message:'读取失败')}}
  async function run(){if(busy||!validTokenLimit(limit))return;setBusy(true);setError('');try{const report=await recheck(targetType,targetId!,limit);setReports(previous=>[report,...previous])}catch(error){setError(error instanceof Error?error.message:'重新审校失败')}finally{setBusy(false)}}
  return <details className="review-control"><summary>核对原文 / 重新审校</summary><p>点击内容中的引用可自行核对原文。重新审校会额外调用当前文字模型，最多抽查 12 项，不改写原内容，结果保存为独立报告。</p><label>重新审校用量上限（可选）<input type="number" min="1024" max="4194304" step="1" value={limit??''} onChange={event=>setLimit(event.target.value?Number(event.target.value):undefined)} placeholder="留空使用模型窗口约束"/></label><div className="review-actions"><button disabled={busy||!validTokenLimit(limit)} onClick={()=>void run()}>{busy?'正在核对…':'开始重新审校（额外调用模型）'}</button><button disabled={busy} onClick={()=>void load()}>查看审校记录</button></div>{error?<p role="alert">{error}</p>:null}{reports.map(report=><section key={report.id}><p>已核对 {report.assessment.reviewed_units}/{report.assessment.total_units} 项 · 原文支持 {report.assessment.supported_units} 项。{report.assessment.reason}</p>{report.assessment.issues.map((issue,index)=><p key={index}>{issue.unit}：{issue.message}</p>)}<UsageDetails value={report.usage}/></section>)}</details>
}

export function NotebookUsage({id}:{id?:string}){
  const[value,setValue]=useState<Usage>();const[error,setError]=useState('');const[busy,setBusy]=useState(false);const generation=useRef(0);
  useEffect(()=>{generation.current++;setValue(undefined);setError('');setBusy(false);return()=>{generation.current++}},[id]);
  return <details className="notebook-usage"><summary>资料库累计用量</summary><button disabled={!id||busy} onClick={()=>{if(!id)return;setBusy(true);setError('');const request=generation.current;void notebookUsage(id).then(value=>{if(request===generation.current)setValue(value)}).catch(error=>{if(request===generation.current)setError(String(error))}).finally(()=>{if(request===generation.current)setBusy(false)})}}>{busy?'正在读取…':'读取 / 刷新统计'}</button>{error?<p role="alert">{error}</p>:null}{value?<UsageDetails value={value}/>:<p>包括已记录的生成、导入和重新审校；旧版本未记录的消耗不包含在内。</p>}</details>
}

export function imageProcessorState(processor:string,providers:import('./api').Provider[]){
  if(processor==='ocr')return '本地文字识别（不调用模型，运行时检查引擎）';
  const provider=providers.find(item=>item.role===processor&&item.active);
  if(!provider)return '将跳过：未配置或已暂停';
  return provider.capabilities?.vision?`使用 ${provider.name}`:'将跳过：未确认视觉能力';
}

export function GettingStarted({hasModel,hasNotebook,sourceCount,hasResult,onSettings,onCreate,onImport,onTry}:{hasModel:boolean;hasNotebook:boolean;sourceCount:number;hasResult:boolean;onSettings:()=>void;onCreate:()=>void;onImport:()=>void;onTry:()=>void}){
  const[demo,setDemo]=useState(false);const[answer,setAnswer]=useState<number>();const[quote,setQuote]=useState(false);
  const steps=[hasModel,hasNotebook,sourceCount>0,hasResult];const current=steps.findIndex(done=>!done);
  const actions=[onSettings,onCreate,onImport,onTry];const names=['连接文字模型','创建资料库','导入并选择资料','生成内容并点击引用核对'];
  return <details className="getting-started" open={current>=0?true:undefined}><summary>{current>=0?`开始使用 · ${names[current]}`:'使用指南 · 阅读 → 核对 → 自测 → 复习'}</summary><ol>{names.map((name,index)=><li key={name}>{steps[index]?'✓ ':`${index+1}. `}{name}</li>)}</ol><div className="review-actions">{current>=0?<button onClick={actions[current]}>下一步：{names[current]}</button>:null}<button onClick={()=>{setDemo(true);setAnswer(undefined);setQuote(false)}}>体验示例（不消耗 token）</button></div>{demo?<Overlay label="使用示例" className="example-tour" onClose={()=>setDemo(false)}><button className="drawer-close" data-autofocus onClick={()=>setDemo(false)}>关闭 ×</button><h2>从资料到可复习的知识</h2><p>以下是内置的虚构示例，不会上传资料或调用模型。</p><h3>1. 阅读与核对</h3><p>试验 A 的发芽率为 80%，试验 B 为 60%。两次试验光照不同，不能仅据此认定差异来自肥料。<button onClick={()=>setQuote(value=>!value)} aria-expanded={quote}>[S1] 查看原文</button></p>{quote?<blockquote>示例试验记录，第 1 页：A 组 10 粒种子发芽 8 粒，B 组 10 粒发芽 6 粒。两组的光照时间不同。</blockquote>:null}<h3>2. 自测</h3><p>可以从资料中确认哪项结论？</p><button disabled={answer!==undefined} onClick={()=>setAnswer(0)}>A. A 组肥料一定更好</button><button disabled={answer!==undefined} onClick={()=>setAnswer(1)}>B. A 组的发芽比例较高</button>{answer!==undefined?<p role="status">{answer===1?'回答正确。':'这次答错了。'}记录支持发芽比例的差异，不能排除光照的影响。<button onClick={()=>setAnswer(undefined)}>重新练习</button></p>:null}<h3>3. 持续复习</h3><p>真实测验支持错题重练，闪卡会依据反馈安排到期复习。复习已有内容不会重新调用生成模型。</p><p>每项任务都可查看资料范围、原文核查情况和用量；自动核查不能保证事实正确。</p></Overlay>:null}</details>
}
