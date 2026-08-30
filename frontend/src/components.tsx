import {useEffect,useMemo,useRef,useState} from 'react';
import {BookOpen,BrainCircuit,Check,ChevronDown,ClipboardList,FileText,Headphones,Layers3,Library,PanelRightOpen,Plus,Search,Settings2,Sparkles,Trash2,Upload,Wifi,Zap} from 'lucide-react';
import type {Artifact,Citation,Job,Notebook,PodcastOptions,Provider,Source} from './api';
import {CitationIndex,ConfirmDialog,Overlay,RichText} from './ui';

const SUPPORTED_EXTENSIONS=new Set(['pdf','docx','pptx','epub','txt','md','markdown','html','htm']);
const PODCAST_LANGUAGE_KEY='sread_podcast_language_v1';

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
    <div className="messages" role="log" aria-live="polite">{messages.length===0?<div className="welcome"><div className="scan-icon"><BrainCircuit/></div><span>{hasNotebook?'NEURAL LINK READY':'SELECT A NOTEBOOK'}</span><h2>{hasNotebook?'从资料中，得到可验证的答案。':'先选择一个 Notebook 开始研究。'}</h2><p>{hasNotebook?'回答仅依据当前勾选的文档。每个事实都附带可追溯引用，点击即可核对原文位置。':'你可以在顶部切换 Notebook，或前往 Notebook 管理页新建资料库。'}</p>{hasNotebook?<div className="prompts"><button disabled={!selectedCount} onClick={()=>setQuestion('这些资料最重要的三个结论是什么？')}>提炼三个核心结论</button><button disabled={!selectedCount} onClick={()=>setQuestion('资料之间有哪些观点冲突？')}>查找观点冲突</button></div>:<button className="primary welcome-action" onClick={()=>{location.hash='notebooks'}}>管理 Notebook</button>}</div>:messages.map((message,index)=><article key={message.id||index} className={`message ${message.role}`}><label>{message.role==='user'?'你的问题':'S-READ · GROUNDED'}</label>{message.degraded?<em className="degraded">安全原文摘录模式</em>:null}<div className="message-body"><RichText content={String(message.content||'')} citations={message.citations||[]} onCitation={onCitation}/></div><CitationIndex citations={message.citations||[]} onCitation={onCitation}/></article>)}{busy?<div className="thinking" role="status"><i/><i/><i/> 正在比对资料并核验引用</div>:null}</div>
    <div className="composer"><label className="sr-only" htmlFor="grounded-question">向已选资料提问</label><textarea id="grounded-question" value={question} disabled={!hasNotebook||busy} onChange={event=>setQuestion(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();void onAsk()}}} placeholder={disabledReason||'向当前选中的资料提问…'}/><div className="composer-meta"><span><BookOpen size={14}/> {selectedCount?`已选择 ${selectedCount} 份资料`:'仅检索已选择资料'} · Enter 发送</span><button onClick={()=>void onAsk()} disabled={busy||!question.trim()||Boolean(disabledReason)}>{busy?'核验中…':'发送'} <Zap size={15}/></button></div></div>
  </main>;
}

const cards=[['summary','资料摘要',Sparkles,'凝练核心结论与限制'],['podcasts','双人音频',Headphones,'两位主持人深入解读'],['quiz','Quiz 题库',Check,'生成可验证的单选题'],['flashcards','Flashcards',Layers3,'用闪卡巩固关键知识']] as const;
function etaText(job:Job){if(job.eta?.status==='learning')return `ETA 学习中 · ${job.eta.sample_count}/5 样本`;const range=job.eta?.remaining_range||[];return range.length===2?`ETA ${Math.ceil(range[0]/60)}–${Math.ceil(range[1]/60)} 分钟`:'ETA 计算中'}
export function StudioRail({onCreate,onOpen,artifacts,jobs,hasNotebook,selectedCount}:{onCreate:(type:string)=>Promise<void>;onOpen:(artifact:Artifact)=>void;artifacts:Artifact[];jobs:Job[];hasNotebook:boolean;selectedCount:number}){
  const active=jobs.filter(job=>['running','queued','cancelling'].includes(job.state)).slice(0,3);const failed=jobs.filter(job=>job.state==='failed').slice(0,2);const disabled=!hasNotebook||selectedCount===0;
  return <aside className="studio panel" aria-label="Studio"><div className="panel-title"><span>STUDIO</span><em>{selectedCount} SELECTED</em></div><p className="studio-intro">将已选资料转化为可学习内容。</p>{disabled?<p className="studio-hint">{hasNotebook?'选择至少一份已索引资料后即可生成。':'先选择或新建 Notebook。'}</p>:null}<div className="studio-cards">{cards.map(([type,title,Icon,description])=><button key={type} disabled={disabled} onClick={()=>void onCreate(type)}><Icon/><span><b>{title}</b><small>{description}</small></span><Plus size={15}/></button>)}</div>
    <div className="activity"><div className="panel-title"><span>输出记录</span><button className="inline-link" onClick={()=>{location.hash='jobs'}}>全部任务</button></div>{active.map(job=><button className="job job-button" key={job.id} onClick={()=>{location.hash='jobs'}}><span>{job.display_name||job.kind.toUpperCase()} · {Math.round(job.progress*100)}%</span><small>{job.stage}{job.stage_total?` · ${job.stage_current}/${job.stage_total}${job.stage_unit||''}`:''}</small><i><b style={{width:`${job.progress*100}%`}}/></i><small>{job.state==='queued'&&job.eta?.queue_position?`队列第 ${job.eta.queue_position} 位 · `:''}{etaText(job)}</small></button>)}{failed.map(job=><button className="job job-button failed-job" key={job.id} onClick={()=>{location.hash='jobs'}}><span>{job.kind.toUpperCase()} · FAILED</span><small>{job.error}</small></button>)}{artifacts.slice(0,6).map(item=><button className="artifact" key={item.id} onClick={()=>onOpen(item)}><FileText size={16}/><span><b>{item.title}</b><small>{item.type.toUpperCase()} · READY</small></span></button>)}{!artifacts.length&&!active.length?<p className="no-output">尚无生成内容</p>:null}</div>
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
  return <Overlay className="podcast-create" label="生成双人深度播客" onClose={onClose} closeOnBackdrop={false}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>PODCAST V2 // DEEP DIVE</span><h2>生成双人深度播客</h2><div className="podcast-provider"><b>{model}</b><small>{device} · {sourceCount} SOURCES · 严格资料内</small></div><label>目标时长<select value={duration} onChange={event=>setDuration(event.target.value as typeof duration)}><option value="auto">AUTO · 12–25 分钟</option><option value="5">5 分钟 · QUICK</option><option value="10">10 分钟 · STANDARD</option><option value="20">20 分钟 · DEEP</option><option value="30">30 分钟 · EXTENDED</option></select></label><label>输出语言<select value={language} onChange={event=>setLanguage(event.target.value as typeof language)}><option value="zh-CN">简体中文</option><option value="auto">跟随资料</option><option value="en">English</option></select></label><label>关注重点<textarea value={focus} onChange={event=>setFocus(event.target.value)} maxLength={1000} placeholder="可选：希望两位主持人重点讨论什么？"/></label><div className="podcast-estimate"><span>预计成片</span><b>{audioRange} MIN</b><span>预计生成</span><b>提交后按本地历史预测</b></div><p>先构建全文证据地图，再按章节编写和审校。生成时间只采用本机同类任务历史估算。</p><button className="primary" disabled={busy||!provider} onClick={()=>void submit()}>{busy?'正在提交…':'生成深度播客'}</button></Overlay>;
}

export function ArtifactDrawer({artifact,onClose,onCitation,onSubmitQuiz,onReview}:{artifact:Artifact|null;onClose:()=>void;onCitation:(citation:Citation)=>void;onSubmitQuiz:(id:string,answers:Record<string,number>)=>Promise<any>;onReview:(id:string,cardId:string,rating:string)=>Promise<void>}){
  const[answers,setAnswers]=useState<Record<string,number>>({});const[score,setScore]=useState<any>();const[index,setIndex]=useState(0);const[flipped,setFlipped]=useState(false);const[currentTime,setCurrentTime]=useState(0);const[reviewing,setReviewing]=useState('');const audioRef=useRef<HTMLAudioElement>(null);
  const citationById=useMemo(()=>new Map((artifact?.citations||[]).map(citation=>[citation.id,citation])),[artifact]);
  if(!artifact)return null;
  const artifactId=artifact.id,items=Array.isArray(artifact.payload?.items)?artifact.payload.items:[],card=items[index],turns=Array.isArray(artifact.payload?.turns)?artifact.payload.turns:[],chapters=Array.isArray(artifact.payload?.chapters)?artifact.payload.chapters:[],activeTurn=turns.findIndex((turn:any)=>currentTime>=Number(turn.start_seconds||0)&&currentTime<(Number(turn.end_seconds||0)));
  function seek(seconds:number){if(audioRef.current){audioRef.current.currentTime=seconds;void audioRef.current.play()}}
  async function rate(rating:string){if(!card||reviewing)return;setReviewing(rating);try{await onReview(artifactId,card.id,rating);if(index<items.length-1){setIndex(value=>value+1);setFlipped(false)}}catch{/* Keep the current card. */}finally{setReviewing('')}}
  return <Overlay className="artifact-drawer" label={artifact.title} onClose={onClose}><button className="drawer-close" data-autofocus onClick={onClose}>关闭 ×</button><span>STUDIO OUTPUT // {artifact.type.toUpperCase()}</span><h2>{artifact.title}</h2>{artifact.payload?.degraded?<em className="degraded">SAFE EVIDENCE MODE</em>:null}
    {artifact.type==='summary'?<div className="artifact-copy"><RichText content={String(artifact.payload?.content||'')} citations={artifact.citations} onCitation={onCitation}/></div>:null}
    {artifact.type==='quiz'?<div className="quiz-view">{items.map((item:any)=><fieldset key={item.id}><legend>{item.id.toUpperCase()} · {item.question}</legend>{item.options.map((option:string,optionIndex:number)=><label key={optionIndex}><input type="radio" name={item.id} checked={answers[item.id]===optionIndex} onChange={()=>setAnswers(value=>({...value,[item.id]:optionIndex}))}/>{option}</label>)}</fieldset>)}<button className="primary" disabled={Object.keys(answers).length!==items.length} onClick={async()=>setScore(await onSubmitQuiz(artifact.id,answers))}>提交答案</button>{score?<strong className="score">得分 {(score.score*100).toFixed(0)}% · {score.correct}/{score.total}</strong>:null}</div>:null}
    {artifact.type==='flashcard'&&card?<div className="flash-view"><button className={`flash-card ${flipped?'flipped':''}`} onClick={()=>setFlipped(value=>!value)}><span>{flipped?card.back:card.front}</span><small>{flipped?'背面 · 点击翻回':'正面 · 点击翻面'}</small></button><div className="flash-nav"><button disabled={index===0} onClick={()=>{setIndex(value=>value-1);setFlipped(false)}}>上一张</button><span>{index+1} / {items.length}</span><button disabled={index===items.length-1} onClick={()=>{setIndex(value=>value+1);setFlipped(false)}}>下一张</button></div><div className="ratings">{[['again','再学'],['hard','困难'],['mastered','掌握']].map(([rating,label])=><button disabled={Boolean(reviewing)} key={rating} onClick={()=>void rate(rating)}>{reviewing===rating?'保存中…':label}</button>)}</div></div>:null}
    {artifact.type==='podcast'?<div className="podcast-view"><audio ref={audioRef} controls preload="metadata" src={artifact.media_url} onTimeUpdate={event=>setCurrentTime(event.currentTarget.currentTime)}/>{artifact.payload?.version===2?<><div className="podcast-meta"><span>TARGET <b>{artifact.payload?.duration?.target_minutes} MIN</b></span><span>ACTUAL <b>{Math.round(Number(artifact.payload?.duration?.actual_seconds||0)/60)} MIN</b></span><span>ENGINE <b>{artifact.payload?.provider?.model}</b></span><span>DEVICE <b>{String(artifact.payload?.provider?.compute_device||'').toUpperCase()}</b></span></div><nav className="podcast-chapters" aria-label="播客章节">{chapters.map((chapter:any)=><button key={chapter.id} onClick={()=>seek(Number(chapter.start_seconds||0))}>{chapter.title}</button>)}</nav><div className="podcast-transcript">{turns.map((turn:any,turnIndex:number)=><div role="button" tabIndex={0} key={turn.id} className={`podcast-turn ${turnIndex===activeTurn?'active':''}`} onClick={()=>seek(Number(turn.start_seconds||0))} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();seek(Number(turn.start_seconds||0))}}}><b>{turn.speaker==='HOST_A'?'A':'B'}</b><span>{turn.text}<small>{turn.citation_ids?.map((citationId:string)=>{const citation=citationById.get(citationId);return citation?<button key={citationId} onClick={event=>{event.stopPropagation();onCitation(citation)}}>[{citationId}]</button>:null})}</small></span></div>)}</div></>:<div className="artifact-copy"><RichText content={String(artifact.payload?.script||'')} citations={artifact.citations} onCitation={onCitation}/></div>}</div>:null}
    <CitationIndex citations={artifact.citations||[]} onCitation={onCitation}/>
  </Overlay>;
}
