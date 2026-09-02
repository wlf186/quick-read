import {useEffect,useMemo,useRef,useState} from 'react';
import {BookOpen,BrainCircuit,Check,ChevronDown,ChevronLeft,ChevronRight,ClipboardList,Download,FileText,Headphones,Layers3,Library,Lightbulb,Maximize2,PanelRightOpen,Plus,RotateCcw,Search,Settings2,Shuffle,Sparkles,Trash2,Upload,Wifi,Zap} from 'lucide-react';
import {answerQuizItem,createStudySession,flashcardsCsvUrl,reviewStudyCard,suspendFlashcard,type Artifact,type Citation,type Job,type Notebook,type PodcastOptions,type Provider,type Source,type StudyOptions,type StudySession} from './api';
import {CitationIndex,ConfirmDialog,Overlay,RichText} from './ui';

const SUPPORTED_EXTENSIONS=new Set(['pdf','docx','pptx','epub','txt','md','markdown','html','htm']);
const PODCAST_LANGUAGE_KEY='sread_podcast_language_v1';

function artifactStatusText(item:Artifact){
  const report=item.payload?.quality_report;
  if(item.status==='partial'||report?.partial){
    const generated=Number(report?.generated_count??item.payload?.turns?.length??0);
    const requested=Number(report?.requested_count??report?.target_turn_count??0);
    return requested>0?`PARTIAL · ${generated}/${requested}`:'PARTIAL';
  }
  return String(item.status||'ready').toUpperCase();
}

export function Logo(){return <div className="brand"><div className="mark"><Zap size={20}/></div><div><strong>SANDEVISTAN</strong><span>// READ</span></div></div>}

export function LoginScreen({onLogin,error}:{onLogin:(key:string)=>Promise<void>;error:string}){
  const[value,setValue]=useState('');const[busy,setBusy]=useState(false);
  async function submit(){if(!value||busy)return;setBusy(true);try{await onLogin(value)}catch{/* Parent supplies the inline error. */}finally{setBusy(false)}}
  return <main className="login-screen"><form className="login-box" onSubmit={event=>{event.preventDefault();void submit()}}>
    <Logo/><span>SECURE PERIMETER // LAN MODE</span><h1>访问授权</h1>
    <p>服务正在局域网接口监听。请输入项目配置中的访问密钥以建立会话。</p>
    <label>访问密钥<input type="password" autoFocus autoComplete="current-password" placeholder="ACCESS KEY" value={value} onChange={event=>setValue(event.target.value)}/></label>
    <button disabled={!value||busy} type="submit">{busy?'正在验证…':'授权访问'} <Zap size={15}/></button>
    {error?<em role="alert">{error}</em>:null}<small>密钥位于本机项目的 runtime/config.toml，不会发送给其它服务。</small>
  </form></main>;
}

export function Header({notebook,notebooks,status,onSelect,onCreate,onSettings,route}:{notebook?:Notebook;notebooks:Notebook[];status:any;onSelect:(id:string)=>void;onCreate:(title:string)=>Promise<void>;onSettings:()=>void;route:string}){
  const[open,setOpen]=useState(false);const[title,setTitle]=useState('');const[creating,setCreating]=useState(false);const menuRef=useRef<HTMLDivElement>(null);
  useEffect(()=>setOpen(false),[route]);
  useEffect(()=>{
    if(!open)return;
    const outside=(event:PointerEvent)=>{if(!menuRef.current?.contains(event.target as Node))setOpen(false)};
    const escape=(event:KeyboardEvent)=>{if(event.key==='Escape')setOpen(false)};
    document.addEventListener('pointerdown',outside);document.addEventListener('keydown',escape);
    return()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape)};
  },[open]);
  async function create(){const next=title.trim();if(!next||creating)return;setCreating(true);try{await onCreate(next);setTitle('');setOpen(false)}catch{/* Keep the draft open. */}finally{setCreating(false)}}
  return <header>
    <button className="brand-link" onClick={()=>{location.hash='workspace'}} aria-label="返回工作台"><Logo/></button>
    <nav className="top-nav" aria-label="主导航">
      <button aria-current={route==='workspace'?'page':undefined} className={route==='workspace'?'active':''} onClick={()=>{location.hash='workspace'}}><Zap size={16}/><span>工作台</span></button>
      <button aria-current={route==='jobs'?'page':undefined} className={route==='jobs'?'active':''} onClick={()=>{location.hash='jobs'}}><ClipboardList size={16}/><span>任务</span></button>
      <button aria-current={route==='notebooks'?'page':undefined} className={route==='notebooks'?'active':''} onClick={()=>{location.hash='notebooks'}}><Library size={16}/><span>Notebook</span></button>
    </nav>
    <div className="notebook-menu" ref={menuRef}>
      <button className="notebook-switch" aria-expanded={open} aria-controls="notebook-popover" onClick={()=>setOpen(value=>!value)}><span>当前 NOTEBOOK</span><b>{notebook?.title||'选择或新建 Notebook'}</b><ChevronDown size={15}/></button>
      {open?<div className="notebook-popover" id="notebook-popover" role="dialog" aria-label="选择 Notebook">
        <div className="notebook-options">{notebooks.map(item=><button key={item.id} className={item.id===notebook?.id?'active':''} onClick={()=>{onSelect(item.id);location.hash='workspace';setOpen(false)}}>{item.title}</button>)}</div>
        <form className="notebook-create" onSubmit={event=>{event.preventDefault();void create()}}><label className="sr-only" htmlFor="quick-notebook-title">新 Notebook 名称</label><input id="quick-notebook-title" value={title} onChange={event=>setTitle(event.target.value)} placeholder="新 Notebook 名称"/><button aria-label="创建 Notebook" disabled={!title.trim()||creating} type="submit"><Plus size={15}/></button></form>
      </div>:null}
    </div>
    <div className="header-actions"><div className={`connection ${status?.providers?.main?.ok?'online':status?'':'pending'}`}><Wifi size={15}/><span>{status?.providers?.main?.ok?'本地核心在线':status?'核心能力降级':'正在连接'}</span></div><button className="icon-button" onClick={onSettings} aria-label="设置"><Settings2 size={19}/></button></div>
  </header>;
}

export function SourceRail({sources,hasNotebook,onUpload,onToggle,onDelete,onNotify}:{sources:Source[];hasNotebook:boolean;onUpload:(files:File[])=>Promise<void>;onToggle:(source:Source)=>Promise<void>;onDelete:(source:Source)=>Promise<void>;onNotify:(message:string,tone?:'info'|'success'|'error')=>void}){
  const[query,setQuery]=useState('');const[dragging,setDragging]=useState(false);const[uploading,setUploading]=useState(false);const[working,setWorking]=useState('');const[deleting,setDeleting]=useState<Source>();const input=useRef<HTMLInputElement>(null);
  const visible=useMemo(()=>sources.filter(source=>source.filename.toLowerCase().includes(query.toLowerCase())),[sources,query]);
  const selected=sources.filter(source=>source.selected&&source.state==='ready').length;
  async function accept(files:FileList|File[]){
    if(!hasNotebook){onNotify('请先新建或选择一个 Notebook','error');return}
    const all=Array.from(files);const supported=all.filter(file=>SUPPORTED_EXTENSIONS.has(file.name.split('.').pop()?.toLowerCase()||''));
    if(supported.length!==all.length)onNotify('已忽略不支持的文件格式','error');if(!supported.length)return;
    setUploading(true);try{await onUpload(supported);if(input.current)input.current.value=''}catch{/* Parent shows the error. */}finally{setUploading(false)}
  }
  async function toggle(source:Source){setWorking(source.id);try{await onToggle(source)}catch{/* Parent shows the error. */}finally{setWorking('')}}
  return <aside className="sources panel" aria-label="资料">
    <div className="panel-title"><span>资料 SOURCES</span><em>{selected}/{sources.length}</em></div>
    <label className={`upload-zone ${dragging?'drag-active':''} ${!hasNotebook?'disabled':''}`} onDragEnter={event=>{event.preventDefault();if(hasNotebook)setDragging(true)}} onDragOver={event=>event.preventDefault()} onDragLeave={event=>{if(!event.currentTarget.contains(event.relatedTarget as Node))setDragging(false)}} onDrop={event=>{event.preventDefault();setDragging(false);void accept(event.dataTransfer.files)}}>
      <input ref={input} type="file" multiple disabled={!hasNotebook||uploading} accept=".pdf,.docx,.pptx,.epub,.txt,.md,.markdown,.html,.htm" onChange={event=>event.target.files&&void accept(event.target.files)}/><Upload/><b>{uploading?'正在接入资料…':dragging?'松开以上传':'拖放或点击上传'}</b><small>PDF · EPUB · DOCX · PPTX · TXT · MD · HTML</small>
    </label>
    <label className="source-tools"><Search size={15}/><span className="sr-only">搜索资料</span><input value={query} onChange={event=>setQuery(event.target.value)} placeholder="搜索资料"/></label>
    <div className="source-list">{visible.length===0?<div className="empty"><FileText/><p>{sources.length?'没有匹配资料':hasNotebook?'等待资料接入':'请先选择 Notebook'}</p></div>:null}{visible.map(source=><div key={source.id} className={`source-row ${source.selected?'selected':''}`}>
      <button className="source" aria-pressed={Boolean(source.selected)} disabled={working===source.id||source.state!=='ready'} onClick={()=>void toggle(source)}><span className="check">{source.selected?<Check size={13}/>:null}</span><span><b title={source.filename}>{source.filename}</b><small className={source.state==='failed'?'failed':''}>{source.state==='ready'?`${source.page_count||1} ${source.metadata?.locator_unit==='slide'?'SLIDES':source.metadata?.locator_unit==='chapter'?'CHAPTERS':'PAGES'} · 已索引`:source.state==='failed'?source.error||'解析失败':source.state.toUpperCase()}</small></span></button>
      <button className="source-delete" aria-label={`删除 ${source.filename}`} onClick={()=>setDeleting(source)}><Trash2 size={15}/></button>
    </div>)}</div>
    <button className="add-source" disabled={!hasNotebook||uploading} onClick={()=>input.current?.click()}><Plus size={15}/> 添加资料</button>
    {deleting?<ConfirmDialog key={deleting.id} title={`删除资料“${deleting.filename}”`} description="将删除该资料的本地文件和索引，Notebook 中其它内容不受影响。" confirmLabel="删除资料" onCancel={()=>setDeleting(undefined)} onConfirm={async()=>{await onDelete(deleting);setDeleting(undefined)}}/>:null}
  </aside>;
}

export function ChatPanel({messages,question,setQuestion,onAsk,busy,onCitation,onNewConversation,onOpenStudio,hasNotebook,selectedCount}:{messages:any[];question:string;setQuestion:(value:string)=>void;onAsk:()=>Promise<void>;busy:boolean;onCitation:(citation:Citation)=>void;onNewConversation:()=>void;onOpenStudio:()=>void;hasNotebook:boolean;selectedCount:number}){
  const disabledReason=!hasNotebook?'请先选择或新建 Notebook':selectedCount===0?'请先选择至少一份资料':'';
  return <main className="chat panel"><div className="chat-heading"><div><span>GROUNDING CONSOLE</span><h1>向资料提问</h1></div><div className="chat-actions"><button className="tablet-studio-trigger" onClick={onOpenStudio}><PanelRightOpen/> Studio</button><button onClick={onNewConversation} disabled={!messages.length}>新对话</button><div className="strict"><span/> 严格溯源</div></div></div>
    <div className="messages" role="log" aria-live="polite">{messages.length===0?<div className="welcome"><div className="scan-icon"><BrainCircuit/></div><span>{hasNotebook?'NEURAL LINK READY':'SELECT A NOTEBOOK'}</span><h2>{hasNotebook?'从资料中，得到可验证的答案。':'先选择一个 Notebook 开始研究。'}</h2><p>{hasNotebook?'回答仅依据当前勾选的文档。每个事实都附带可追溯引用，点击即可核对原文位置。':'你可以在顶部切换 Notebook，或前往 Notebook 管理页新建资料库。'}</p>{hasNotebook?<div className="prompts"><button disabled={!selectedCount} onClick={()=>setQuestion('这些资料最重要的三个结论是什么？')}>提炼三个核心结论</button><button disabled={!selectedCount} onClick={()=>setQuestion('资料之间有哪些观点冲突？')}>查找观点冲突</button></div>:<button className="primary welcome-action" onClick={()=>{location.hash='notebooks'}}>管理 Notebook</button>}</div>:messages.map((message,index)=><article key={message.id||index} className={`message ${message.role}`}><label>{message.role==='user'?'你的问题':'S-READ · GROUNDED'}</label>{message.degraded?<em className="degraded">安全原文摘录模式</em>:null}{message.metadata?.context_usage?.adjusted?<em className="context-adjusted">已按模型窗口调整证据量</em>:null}<div className="message-body"><RichText content={String(message.content||'')} citations={message.citations||[]} onCitation={onCitation}/></div><CitationIndex citations={message.citations||[]} onCitation={onCitation}/></article>)}{busy?<div className="thinking" role="status"><i/><i/><i/> 正在比对资料并核验引用</div>:null}</div>
    <div className="composer"><label className="sr-only" htmlFor="grounded-question">向已选资料提问</label><textarea id="grounded-question" value={question} disabled={!hasNotebook||busy} onChange={event=>setQuestion(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();void onAsk()}}} placeholder={disabledReason||'向当前选中的资料提问…'}/><div className="composer-meta"><span><BookOpen size={14}/> {selectedCount?`已选择 ${selectedCount} 份资料`:'仅检索已选择资料'} · Enter 发送</span><button onClick={()=>void onAsk()} disabled={busy||!question.trim()||Boolean(disabledReason)}>{busy?'核验中…':'发送'} <Zap size={15}/></button></div></div>
  </main>;
}

const cards=[['summary','资料摘要',Sparkles,'凝练核心结论与限制'],['podcasts','双人音频',Headphones,'两位主持人深入解读'],['quiz','Quiz 题库',Check,'生成可验证的单选题'],['flashcards','Flashcards',Layers3,'用闪卡巩固关键知识']] as const;
function etaText(job:Job){if(job.eta?.status==='learning')return `ETA 学习中 · ${job.eta.sample_count}/5 样本`;const range=job.eta?.remaining_range||[];return range.length===2?`ETA ${Math.ceil(range[0]/60)}–${Math.ceil(range[1]/60)} 分钟`:'ETA 计算中'}
export function StudioRail({onCreate,onOpen,artifacts,jobs,hasNotebook,selectedCount}:{onCreate:(type:string)=>Promise<void>;onOpen:(artifact:Artifact)=>void;artifacts:Artifact[];jobs:Job[];hasNotebook:boolean;selectedCount:number}){
  const active=jobs.filter(job=>['running','queued','cancelling'].includes(job.state)).slice(0,3);const failed=jobs.filter(job=>job.state==='failed').slice(0,2);const disabled=!hasNotebook||selectedCount===0;
  return <aside className="studio panel" aria-label="Studio"><div className="panel-title"><span>STUDIO</span><em>{selectedCount} SELECTED</em></div><p className="studio-intro">将已选资料转化为可学习内容。</p>{disabled?<p className="studio-hint">{hasNotebook?'选择至少一份已索引资料后即可生成。':'先选择或新建 Notebook。'}</p>:null}<div className="studio-cards">{cards.map(([type,title,Icon,description])=><button key={type} disabled={disabled} onClick={()=>void onCreate(type)}><Icon/><span><b>{title}</b><small>{description}</small></span><Plus size={15}/></button>)}</div>
    <div className="activity"><div className="panel-title"><span>输出记录</span><button className="inline-link" onClick={()=>{location.hash='jobs'}}>全部任务</button></div>{active.map(job=><button className="job job-button" key={job.id} onClick={()=>{location.hash='jobs'}}><span>{job.display_name||job.kind.toUpperCase()} · {Math.round(job.progress*100)}%</span><small>{job.stage}{job.stage_total?` · ${job.stage_current}/${job.stage_total}${job.stage_unit||''}`:''}</small><i><b style={{width:`${job.progress*100}%`}}/></i><small>{job.state==='queued'&&job.eta?.queue_position?`队列第 ${job.eta.queue_position} 位 · `:''}{etaText(job)}</small></button>)}{failed.map(job=><button className="job job-button failed-job" key={job.id} onClick={()=>{location.hash='jobs'}}><span>{job.kind.toUpperCase()} · FAILED</span><small>{job.error}</small></button>)}{artifacts.slice(0,6).map(item=><button className="artifact" key={item.id} onClick={()=>onOpen(item)}><FileText size={16}/><span><b>{item.title}</b><small>{item.type.toUpperCase()} · {artifactStatusText(item)}</small></span></button>)}{!artifacts.length&&!active.length?<p className="no-output">尚无生成内容</p>:null}</div>
  </aside>;
}

export function CitationDrawer({citation,onClose}:{citation:Citation|null;onClose:()=>void}){
  if(!citation)return null;
  return <Overlay className="citation-drawer" layer="nested" label={`引用 ${citation.id}`} onClose={onClose}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>SOURCE TRACE // {citation.id}</span><h2>{citation.filename}</h2><div className="locator">{Object.entries(citation.locator||{}).map(([key,value])=><i key={key}>{key}: {String(value)}</i>)}</div><blockquote>{citation.quote}</blockquote><small>该片段来自上传资料的固定修订版本。</small></Overlay>;
}

function initialPodcastLanguage(){try{return (localStorage.getItem(PODCAST_LANGUAGE_KEY) as 'zh-CN'|'auto'|'en')||'zh-CN'}catch{return 'zh-CN' as const}}
export function PodcastCreateModal({provider,sourceCount,onClose,onCreate}:{provider?:Provider;sourceCount:number;onClose:()=>void;onCreate:(options:PodcastOptions)=>Promise<void>}){
  const[duration,setDuration]=useState<'auto'|'5'|'10'|'20'|'30'>('auto');const[language,setLanguage]=useState<'zh-CN'|'auto'|'en'>(initialPodcastLanguage);const[focus,setFocus]=useState('');const[busy,setBusy]=useState(false);
  const device=String(provider?.config?.compute_device||'未配置').toUpperCase(),model=provider?.model||'未配置',audioRange=duration==='auto'?'12–25':duration;
  async function submit(){if(busy||!provider)return;setBusy(true);try{try{localStorage.setItem(PODCAST_LANGUAGE_KEY,language)}catch{/* Optional preference. */}await onCreate({duration_mode:duration==='auto'?'auto':'fixed',minutes:duration==='auto'?undefined:Number(duration) as 5|10|20|30,language,focus})}catch{/* Keep the form open. */}finally{setBusy(false)}}
  return <Overlay className="podcast-create" label="生成双人深度播客" onClose={onClose} closeOnBackdrop={false}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>PODCAST V4 // EDITORIAL ACTS</span><h2>生成双人深度播客</h2><div className="podcast-provider"><b>{model}</b><small>{device} · {sourceCount} SOURCES · 严格资料内</small></div><label>目标时长<select value={duration} onChange={event=>setDuration(event.target.value as typeof duration)}><option value="auto">AUTO · 12–25 分钟</option><option value="5">5 分钟 · QUICK</option><option value="10">10 分钟 · STANDARD</option><option value="20">20 分钟 · DEEP</option><option value="30">30 分钟 · EXTENDED</option></select></label><label>输出语言<select value={language} onChange={event=>setLanguage(event.target.value as typeof language)}><option value="zh-CN">简体中文</option><option value="auto">跟随资料</option><option value="en">English</option></select></label><label>关注重点<textarea value={focus} onChange={event=>setFocus(event.target.value)} maxLength={1000} placeholder="可选：希望两位主持人重点讨论什么？"/></label><div className="podcast-estimate"><span>预计成片</span><b>{audioRange} MIN</b><span>预计生成</span><b>提交后按本地历史预测</b></div><p>少量长 Act 完成整集编排和审校，成品再由本地 ASR 验收；脚本或音频不合格时不会发布。</p><button className="primary" disabled={busy||!provider} onClick={()=>void submit()}>{busy?'正在提交…':'生成深度播客'}</button></Overlay>;
}

export function StudyCreateModal({kind,provider,sourceCount,onClose,onCreate}:{kind:'quiz'|'flashcard';provider?:Provider;sourceCount:number;onClose:()=>void;onCreate:(options:StudyOptions)=>Promise<void>}){
  const defaults=kind==='quiz'?{few:5,standard:10,more:20}:{few:10,standard:20,more:40};
  const[amount,setAmount]=useState<keyof typeof defaults>('standard');const[difficulty,setDifficulty]=useState<StudyOptions['difficulty']>('mixed');const[language,setLanguage]=useState<StudyOptions['language']>('auto');const[customPrompt,setCustomPrompt]=useState('');const[busy,setBusy]=useState(false);
  const profile=provider?.capabilities?.study_generation,tier=profile?.tier||'lite',hardDisabled=tier==='lite';
  async function submit(){if(busy||!provider)return;setBusy(true);try{await onCreate({count:defaults[amount],difficulty,language,custom_prompt:customPrompt.trim()})}catch{/* Keep options available after an API error. */}finally{setBusy(false)}}
  const title=kind==='quiz'?'生成理解型 Quiz':'生成可复习 Flashcards';
  return <Overlay className="study-create" label={title} onClose={onClose} closeOnBackdrop={false}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>STUDY PIPELINE // {tier.toUpperCase()}</span><h2>{title}</h2><div className={`study-tier study-tier-${tier}`}><b>{provider?.model||'未配置 MAIN Provider'}</b><small>{tier==='full'?'完整知识蓝图 · 独立证据审校':'兼容档 · 小批次严格淘汰'} · {sourceCount} SOURCES</small>{profile?.reason?<em>{profile.reason}</em>:null}</div>
    <label>数量<select value={amount} onChange={event=>setAmount(event.target.value as keyof typeof defaults)}><option value="few">较少 · {defaults.few}</option><option value="standard">标准 · {defaults.standard}</option><option value="more">较多 · {defaults.more}</option></select></label>
    <label>难度<select value={difficulty} onChange={event=>setDifficulty(event.target.value as StudyOptions['difficulty'])}><option value="mixed">混合</option><option value="easy">简单</option><option value="medium">中等</option><option value="hard" disabled={hardDisabled}>困难{hardDisabled?' · 需要完整档':''}</option></select></label>
    <label>输出语言<select value={language} onChange={event=>setLanguage(event.target.value as StudyOptions['language'])}><option value="auto">跟随资料</option><option value="zh-CN">简体中文</option><option value="en">English</option></select></label>
    <label>定制要求<textarea value={customPrompt} onChange={event=>setCustomPrompt(event.target.value)} maxLength={1000} placeholder="可选：指定考试范围、受众、侧重点或题目风格。"/></label>
    {hardDisabled?<p>当前模型不会生成困难题；系统宁可少生成，也不会用无法核验的模板内容补数。</p>:<p>先规划知识覆盖，再生成、审校、修复并去除重复题卡。</p>}
    <button className="primary" disabled={busy||!provider} onClick={()=>void submit()}>{busy?'正在提交…':`生成 ${kind==='quiz'?'Quiz':'Flashcards'}`}</button>
  </Overlay>;
}

function pendingIndex(session:StudySession){const index=session.items.findIndex(item=>session.kind==='quiz'?!item.result:!item.review);return index<0?Math.max(0,session.items.length-1):index}

function StudyArtifactDrawer({artifact,onClose,onCitation}:{artifact:Artifact;onClose:()=>void;onCitation:(citation:Citation)=>void}){
  const[session,setSession]=useState<StudySession>();const[index,setIndex]=useState(0);const[selection,setSelection]=useState<Record<string,number>>({});const[flipped,setFlipped]=useState(false);const[hint,setHint]=useState(false);const[explain,setExplain]=useState(false);const[busy,setBusy]=useState(false);const[error,setError]=useState('');const[fullscreen,setFullscreen]=useState(false);
  useEffect(()=>{let active=true;setSession(undefined);setError('');void createStudySession(artifact.id,artifact.type==='quiz'?'all':'due').then(value=>{if(active){setSession(value);setIndex(pendingIndex(value))}}).catch(reason=>{if(active)setError(reason instanceof Error?reason.message:'无法开始学习会话')});return()=>{active=false}},[artifact.id,artifact.type]);
  const item=session?.items[index];
  async function restart(mode:'all'|'missed'|'due'|'same',shuffle=false){setBusy(true);setError('');try{const value=await createStudySession(artifact.id,mode,shuffle);setSession(value);setIndex(pendingIndex(value));setFlipped(false);setHint(false);setExplain(false)}catch(reason){setError(reason instanceof Error?reason.message:'无法创建学习会话')}finally{setBusy(false)}}
  async function checkAnswer(){if(!session||!item||selection[item.id]===undefined||item.result||busy)return;setBusy(true);try{const response=await answerQuizItem(session.id,item.id,selection[item.id]);setSession(response.session)}catch(reason){setError(reason instanceof Error?reason.message:'答案提交失败')}finally{setBusy(false)}}
  async function rate(rating:'again'|'hard'|'good'|'easy'){if(!session||!item||busy)return;setBusy(true);try{const response=await reviewStudyCard(session.id,item.id,rating);const value=response.session as StudySession;setSession(value);setIndex(pendingIndex(value));setFlipped(false);setExplain(false)}catch(reason){setError(reason instanceof Error?reason.message:'复习记录保存失败')}finally{setBusy(false)}}
  async function removeCard(){if(!session||!item||busy)return;setBusy(true);try{await suspendFlashcard(artifact.id,item.id);const items=session.items.filter(value=>value.id!==item.id);setSession({...session,items,progress:{...session.progress,total:items.length}});setIndex(Math.min(index,Math.max(0,items.length-1)));setFlipped(false)}catch(reason){setError(reason instanceof Error?reason.message:'闪卡移除失败')}finally{setBusy(false)}}
  const completed=session?.status==='complete',quizCorrect=session?.items.filter(value=>value.result?.correct).length||0;
  return <Overlay className={`artifact-drawer study-drawer ${fullscreen?'study-fullscreen':''}`} label={artifact.title} onClose={onClose}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>ACTIVE STUDY // {artifact.type.toUpperCase()}</span><h2>{artifact.title}</h2><div className="study-head"><div><b>{session?.progress.current||0} / {session?.progress.total||0}</b><small>{session?.mode?.toUpperCase()||'LOADING'} · {artifact.payload?.quality_report?.pipeline_tier?.toUpperCase()||'LEGACY'}</small></div><button aria-label="切换全屏" onClick={()=>setFullscreen(value=>!value)}><Maximize2/></button></div>{session?<i className="study-progress"><b style={{width:`${session.progress.total?session.progress.current/session.progress.total*100:0}%`}}/></i>:null}{artifact.status==='partial'||artifact.payload?.quality_report?.partial?<em className="degraded">仅保留 {artifact.payload?.quality_report?.generated_count} 个通过校验的内容</em>:null}{error?<p className="study-error" role="alert">{error}</p>:null}
    {!session&&!error?<p className="study-empty">正在恢复学习进度…</p>:null}
    {session&&!item?<div className="study-empty"><BrainCircuit/><h3>当前队列没有待学习内容</h3><p>可以重新学习全部内容，或稍后回来到期复习。</p><button className="primary" onClick={()=>void restart('all')}>学习全部</button></div>:null}
    {session&&item&&session.kind==='quiz'?<div className="quiz-study"><div className="question-nav"><button disabled={index===0} onClick={()=>{setIndex(value=>value-1);setHint(false)}}><ChevronLeft/>上一题</button><span>{index+1} / {session.items.length}</span><button disabled={index===session.items.length-1} onClick={()=>{setIndex(value=>value+1);setHint(false)}}>下一题<ChevronRight/></button></div><small className="learning-objective">{item.difficulty?.toUpperCase()} · {item.cognitive_level?.toUpperCase()} · {item.learning_objective}</small><h3>{item.question}</h3><div className="quiz-options">{item.options?.map((option:string,optionIndex:number)=>{const result=item.result;const state=result?optionIndex===result.answer_index?'correct':optionIndex===result.selected_index?'wrong':'':selection[item.id]===optionIndex?'selected':'';return <button className={state} disabled={Boolean(result)} key={optionIndex} onClick={()=>setSelection(value=>({...value,[item.id]:optionIndex}))}><b>{String.fromCharCode(65+optionIndex)}</b><span>{option}</span></button>})}</div><div className="quiz-actions"><button onClick={()=>setHint(value=>!value)}><Lightbulb/>提示</button><button className="primary" disabled={selection[item.id]===undefined||Boolean(item.result)||busy} onClick={()=>void checkAnswer()}>{busy?'核验中…':'检查答案'}</button></div>{hint&&!item.result?<p className="study-feedback hint"><Lightbulb/> {item.hint||'回到资料中的核心概念与关系进行判断。'}</p>:null}{item.result?<div className={`study-feedback ${item.result.correct?'feedback-correct':'feedback-wrong'}`}><b>{item.result.correct?'回答正确':'需要再看一遍'}</b><p>{item.result.explanation}</p><div className="item-citations">{item.result.citation_details?.map((citation:Citation)=><button key={citation.id} onClick={()=>onCitation(citation)}>[{citation.id}] {citation.filename}</button>)}</div></div>:null}</div>:null}
    {session&&item&&session.kind==='flashcard'?<div className="flash-study"><div className="flash-tools"><button onClick={()=>void restart('same',true)} disabled={busy}><Shuffle/>洗牌</button><a href={flashcardsCsvUrl(artifact.id)} download><Download/>CSV</a><button className="danger-action" onClick={()=>void removeCard()} disabled={busy}><Trash2/>移除</button></div><button className={`flash-card ${flipped?'flipped':''}`} onClick={()=>setFlipped(value=>!value)}><small>{item.difficulty?.toUpperCase()} · {item.card_type?.toUpperCase()}</small><span>{flipped?item.back:item.front}</span><em>{flipped?'背面 · 点击翻回':'正面 · 回忆后翻面'}</em></button>{flipped?<><button className="explain-toggle" onClick={()=>setExplain(value=>!value)}><BrainCircuit/>解释</button>{explain?<div className="study-feedback"><p>{item.explanation||'该答案由下列资料片段支持。'}</p><div className="item-citations">{item.citation_details?.map((citation:Citation)=><button key={citation.id} onClick={()=>onCitation(citation)}>[{citation.id}] {citation.filename}</button>)}</div></div>:null}<div className="ratings ratings-four">{([['again','再来'],['hard','困难'],['good','良好'],['easy','简单']] as const).map(([rating,label])=><button disabled={busy} key={rating} onClick={()=>void rate(rating)}>{label}</button>)}</div></>:null}</div>:null}
    {completed&&session.items.length?<div className="study-complete"><Check/><h3>{session.kind==='quiz'?`完成 · ${quizCorrect}/${session.items.length}`:'本轮复习完成'}</h3><p>{session.kind==='quiz'?'可以只重做错题，或重新挑战全部题目。':'FSRS 已根据反馈安排下次复习。'}</p><div>{session.kind==='quiz'?<button onClick={()=>void restart('missed')}>只练错题</button>:<button onClick={()=>void restart('missed')}>只练再来卡</button>}<button onClick={()=>void restart('same')}><RotateCcw/>相同内容</button><button className="primary" onClick={()=>void restart('all')}>全部重练</button></div></div>:null}
  </Overlay>;
}

export function ArtifactDrawer({artifact,onClose,onCitation}:{artifact:Artifact|null;onClose:()=>void;onCitation:(citation:Citation)=>void;onSubmitQuiz:(id:string,answers:Record<string,number>)=>Promise<any>;onReview:(id:string,cardId:string,rating:string)=>Promise<void>}){
  const[currentTime,setCurrentTime]=useState(0);const audioRef=useRef<HTMLAudioElement>(null);
  const citationById=useMemo(()=>new Map((artifact?.citations||[]).map(citation=>[citation.id,citation])),[artifact]);
  if(!artifact)return null;
  if(artifact.type==='quiz'||artifact.type==='flashcard')return <StudyArtifactDrawer artifact={artifact} onClose={onClose} onCitation={onCitation}/>;
  const turns=Array.isArray(artifact.payload?.turns)?artifact.payload.turns:[],chapters=Array.isArray(artifact.payload?.chapters)?artifact.payload.chapters:[],activeTurn=turns.findIndex((turn:any)=>currentTime>=Number(turn.start_seconds||0)&&currentTime<(Number(turn.end_seconds||0)));
  function seek(seconds:number){if(audioRef.current){audioRef.current.currentTime=seconds;void audioRef.current.play()}}
  return <Overlay className="artifact-drawer" label={artifact.title} onClose={onClose}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>STUDIO OUTPUT // {artifact.type.toUpperCase()}</span><h2>{artifact.title}</h2>{artifact.payload?.degraded?<em className="degraded">SAFE EVIDENCE MODE</em>:null}{artifact.payload?.context_usage?.adjusted?<em className="context-adjusted">已按模型窗口调整证据量</em>:null}
    {artifact.type==='summary'?<div className="artifact-copy"><RichText content={String(artifact.payload?.content||'')} citations={artifact.citations} onCitation={onCitation}/></div>:null}
    {artifact.type==='podcast'?<div className="podcast-view"><audio ref={audioRef} controls preload="metadata" src={artifact.media_url} onTimeUpdate={event=>setCurrentTime(event.currentTarget.currentTime)}/>{Number(artifact.payload?.version)>=2?<><div className="podcast-meta"><span>TARGET <b>{artifact.payload?.duration?.target_minutes} MIN</b></span><span>ACTUAL <b>{Math.round(Number(artifact.payload?.duration?.actual_seconds||0)/60)} MIN</b></span><span>ENGINE <b>{artifact.payload?.provider?.model}</b></span><span>SCRIPT <b>V{artifact.payload?.version}{artifact.payload?.quality?.passed===true?' · PASSED':''}</b></span><span>AUDIO <b>{artifact.payload?.audio_quality?.passed===true?'VERIFIED':'UNVERIFIED'}</b></span><span>DEVICE <b>{String(artifact.payload?.provider?.compute_device||'').toUpperCase()}{artifact.payload?.provider?.fallback_used?' · CPU FALLBACK':''}</b></span></div><nav className="podcast-chapters" aria-label="播客章节">{chapters.map((chapter:any)=><button key={chapter.id} onClick={()=>seek(Number(chapter.start_seconds||0))}>{chapter.title}</button>)}</nav><div className="podcast-transcript">{turns.map((turn:any,turnIndex:number)=><div role="button" tabIndex={0} key={turn.id} className={`podcast-turn ${turnIndex===activeTurn?'active':''}`} onClick={()=>seek(Number(turn.start_seconds||0))} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();seek(Number(turn.start_seconds||0))}}}><b>{turn.speaker==='HOST_A'?'A':'B'}</b><span>{turn.text}<small>{turn.citation_ids?.map((citationId:string)=>{const citation=citationById.get(citationId);return citation?<button key={citationId} onClick={event=>{event.stopPropagation();onCitation(citation)}}>[{citationId}]</button>:null})}</small></span></div>)}</div></>:<div className="artifact-copy"><RichText content={String(artifact.payload?.script||'')} citations={artifact.citations} onCitation={onCitation}/></div>}</div>:null}
    <CitationIndex citations={artifact.citations||[]} onCitation={onCitation}/>
  </Overlay>;
}
