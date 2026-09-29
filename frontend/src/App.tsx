import {StudyReminders} from './study_overview';
import {ConversationHistory,HistoricalScope} from './conversation_history';
import {getConversation,selectSources,submitChat} from './api';
import {lazy,Suspense,useCallback,useEffect,useRef,useState} from 'react';
import {BookOpen,Headphones,MessageSquare} from 'lucide-react';
import {ChatResultDrawer,SummaryCreateModal,ArtifactDrawer,ChatPanel,CitationDrawer,Header,LoginScreen,PodcastCreateModal,SourceRail,StudioRail,StudyCreateModal} from './components';
import {authStatus,createArtifact,createNotebook,createProvider,deleteSource,getArtifact,getArtifacts,getConversations,getImageProcessingPolicy,getJobs,getMessages,getNotebook,getNotebooks,getProviderRoles,getProviders,getStatus,getWorkspaceState,inspectProvider,login,reviewFlashcard,selectSource,submitQuiz,updateImageProcessingPolicy,updateProvider,updateProviderRole,upload,type Artifact,type Citation,type ConfigurableProviderRole,type ImageProcessingPolicy,type Job,type Notebook,type PodcastOptions,type Provider,type ProviderDraft,type ProviderInspection,type ProviderRoleState,type Source,type StudyOptions} from './api';
import {JobsPage,NotebooksPage} from './management';
const SettingsDrawer=lazy(()=>import('./provider_settings').then(module=>({default:module.SettingsDrawer})));
import {WeeklyBudget} from './weekly_usage';
import {SourceRangePicker,Overlay,GettingStarted,validTokenLimit} from './ui';

const ACTIVE_NOTEBOOK_KEY='sread_active_notebook_v1';
type Route='workspace'|'jobs'|'notebooks';
type AppPhase='loading'|'locked'|'ready'|'error';
type WorkspacePanel='chat'|'sources'|'studio';
type ToastState={id:number;message:string;tone:'info'|'success'|'error'};

function storedNotebook(){try{return localStorage.getItem(ACTIVE_NOTEBOOK_KEY)||''}catch{return ''}}
function persistNotebook(id:string){try{if(id)localStorage.setItem(ACTIVE_NOTEBOOK_KEY,id);else localStorage.removeItem(ACTIVE_NOTEBOOK_KEY)}catch{/* Storage may be unavailable. */}}
function routeFromHash():Route{const value=location.hash.replace('#','');return value==='jobs'||value==='notebooks'?value:'workspace'}

function BootScreen({error,onRetry}:{error?:string;onRetry?:()=>void}){
  return <div className="boot-screen" role={error?'alert':'status'}><div className="boot-mark"><span/></div><b>{error?'启动失败':'正在连接本地研究核心'}</b><p>{error||'正在载入 Notebook、Provider 与本地索引状态…'}</p>{onRetry?<button className="primary" onClick={onRetry}>重试连接</button>:null}</div>;
}

export default function App(){
  const[phase,setPhase]=useState<AppPhase>('loading');
  const[route,setRoute]=useState<Route>(routeFromHash);
  const[notebooks,setNotebooks]=useState<Notebook[]>([]);
  const[activeId,setActiveId]=useState(storedNotebook);
  const[notebook,setNotebook]=useState<Notebook>();
  const[status,setStatus]=useState<any>();
  const[providers,setProviders]=useState<Provider[]>([]);
  const[providerRoles,setProviderRoles]=useState<ProviderRoleState[]>([]);
  const[imagePolicy,setImagePolicy]=useState<ImageProcessingPolicy>({mode:'process',processors:['vlm','main','ocr']});
  const[messages,setMessages]=useState<any[]>([]);
  const[conversationId,setConversationId]=useState<string>();
  const[question,setQuestion]=useState('');
  const[busy,setBusy]=useState(false);
  const[citation,setCitation]=useState<Citation|null>(null);
  const[openedChat,setOpenedChat]=useState<{conversationId:string;messageId:string}>();
  const[openedArtifact,setOpenedArtifact]=useState<Artifact|null>(null);
  const[settings,setSettings]=useState(false);
  const[settingsPanel,setSettingsPanel]=useState<'models'|'audio'>('models');
  const[podcastOpen,setPodcastOpen]=useState(false);
  const[studyCreate,setStudyCreate]=useState<'quiz'|'flashcard'|null>(null);
  const[tabletStudio,setTabletStudio]=useState(false);
  const[workspacePanel,setWorkspacePanel]=useState<WorkspacePanel>('chat');
  const[summaryOpen,setSummaryOpen]=useState(false);
  const[chatTokenLimit,setChatTokenLimit]=useState<number>();
  const[artifacts,setArtifacts]=useState<Artifact[]>([]);
  const[jobs,setJobs]=useState<Job[]>([]);
  const[toast,setToast]=useState<ToastState>();
  const[loginError,setLoginError]=useState('');
  const[bootError,setBootError]=useState('');
  const loadSequence=useRef(0);
  const providerSequence=useRef(0);
  const[providerRefreshError,setProviderRefreshError]=useState(false);
  async function refreshProviders(){const sequence=++providerSequence.current;try{const [nextProviders,nextRoles,nextStatus]=await Promise.all([getProviders(),getProviderRoles(),getStatus()]);if(sequence!==providerSequence.current)return;setProviders(nextProviders);setProviderRoles(nextRoles);setStatus(nextStatus);setProviderRefreshError(false)}catch(error){if(sequence===providerSequence.current){setProviderRefreshError(true);setStatus({providers:{},refresh_error:true})}if(error instanceof Error&&error.message==='AUTH_REQUIRED')reportError(error)}}
  const drafts=useRef<Record<string,string>>({});
  const draftKey=useRef('');
  useEffect(()=>{draftKey.current=`${activeId}:${conversationId||'new'}`},[activeId,conversationId]);
  function editQuestion(value:string){drafts.current[draftKey.current]=value;setQuestion(value)}
  function rememberConversation(notebookId:string,id?:string){try{if(id)localStorage.setItem(`sread-conversation-v1-${notebookId}`,id);else localStorage.removeItem(`sread-conversation-v1-${notebookId}`)}catch{}}
  async function openConversation(id:string){const context=chatContext.current;const generation=++context.generation;context.request++;setChatLoading(true);setBusy(false);try{const next=await getMessages(id);if(chatContext.current!==context||context.generation!==generation)return;setConversationId(id);setMessages(next);setQuestion(drafts.current[`${context.notebookId}:${id}`]||'');rememberConversation(context.notebookId,id)}finally{if(chatContext.current===context&&context.generation===generation)setChatLoading(false)}}
  const chatContext=useRef({notebookId:activeId,generation:0,request:0});
  const[chatLoading,setChatLoading]=useState(true);
  const workspaceVersions=useRef<Record<string,string>>({});

  const notify=useCallback((message:string,tone:ToastState['tone']='info')=>setToast({id:Date.now(),message,tone}),[]);
  const reportError=useCallback((error:unknown)=>{
    const message=error instanceof Error?error.message:'请求失败';
    if(message==='AUTH_REQUIRED'){setPhase('locked');setLoginError('会话已失效，请重新输入访问密钥。')}
    else notify(message,'error');
  },[notify]);

  const activateNotebook=useCallback((id:string)=>{
    const context=chatContext.current;
    if(context.notebookId===id)return;
    chatContext.current={notebookId:id,generation:context.generation+1,request:context.request+1};
    loadSequence.current+=1;
    setActiveId(id);persistNotebook(id);setNotebook(undefined);setArtifacts([]);setJobs([]);
    setConversationId(undefined);setMessages([]);setQuestion('');setBusy(false);setChatLoading(Boolean(id));
    workspaceVersions.current={};
  },[]);
  const newConversation=useCallback(()=>{
    chatContext.current.generation+=1;chatContext.current.request+=1;
    setConversationId(undefined);setMessages([]);setQuestion(drafts.current[`${chatContext.current.notebookId}:new`]||'');setBusy(false);setChatLoading(false);rememberConversation(chatContext.current.notebookId);
  },[]);
  const reconcileNotebooks=useCallback((list:Notebook[])=>{
    setNotebooks(list);
    if(!list.some(item=>item.id===chatContext.current.notebookId))activateNotebook(list[0]?.id||'');
  },[activateNotebook]);

  const initialize=useCallback(async()=>{
    setBootError('');setPhase('loading');
    try{
      const access=await authStatus();
      if(access.required&&!access.authenticated){setPhase('locked');return}
      const[list,nextProviders,nextRoles,nextImagePolicy]=await Promise.all([getNotebooks(),getProviders(),getProviderRoles(),getImageProcessingPolicy()]);
      setNotebooks(list);setProviders(nextProviders);setProviderRoles(nextRoles);setImagePolicy(nextImagePolicy);
      const preferred=storedNotebook();const next=list.some(item=>item.id===preferred)?preferred:list[0]?.id||'';
      activateNotebook(next);setPhase('ready');
      const sequence=providerSequence.current;void getStatus().then(value=>{if(sequence===providerSequence.current)setStatus(value)}).catch(()=>{if(sequence===providerSequence.current)setStatus({providers:{}})});
    }catch(error){
      if(error instanceof Error&&error.message==='AUTH_REQUIRED'){setPhase('locked');return}
      const message=error instanceof Error?error.message:'无法启动应用';setBootError(message);setPhase('error');
    }
  },[activateNotebook]);

  const loadGlobal=useCallback(async()=>{
    const sequence=++providerSequence.current;
    const[list,nextProviders,nextRoles,nextImagePolicy,nextStatus]=await Promise.all([getNotebooks(),getProviders(),getProviderRoles(),getImageProcessingPolicy(),getStatus()]);
    reconcileNotebooks(list);if(sequence!==providerSequence.current)return;setProviders(nextProviders);
    setProviderRoles(nextRoles);setImagePolicy(nextImagePolicy);setStatus(nextStatus);
  },[reconcileNotebooks]);

  const refreshNotebooks=useCallback(async()=>reconcileNotebooks(await getNotebooks()),[reconcileNotebooks]);

  const loadCurrent=useCallback(async(id:string,loadHistory=false)=>{
    if(id!==chatContext.current.notebookId)return;
    const sequence=++loadSequence.current,generation=chatContext.current.generation,request=chatContext.current.request;
    const isCurrent=()=>id===chatContext.current.notebookId&&generation===chatContext.current.generation;
    if(!id){setNotebook(undefined);setArtifacts([]);setJobs([]);setConversationId(undefined);setMessages([]);setChatLoading(false);return}
    try{
      const historyPromise=loadHistory?getConversations(id):Promise.resolve<any[]|null>(null);
      const[current,nextArtifacts,nextJobs,conversations]=await Promise.all([getNotebook(id),getArtifacts(id),getJobs(id),historyPromise]);
      if(!isCurrent())return;
      if(sequence===loadSequence.current){setNotebook(current);setArtifacts(nextArtifacts);setJobs(nextJobs)}
      if(loadHistory&&conversations&&request===chatContext.current.request){
        let remembered='';try{remembered=localStorage.getItem(`sread-conversation-v1-${id}`)||''}catch{}const latest=conversations.find(item=>item.id===remembered)||conversations[0];
        const nextMessages=latest?await getMessages(latest.id):[];
        if(!isCurrent()||request!==chatContext.current.request)return;
        setConversationId(latest?.id);setQuestion(drafts.current[`${id}:${latest?.id||'new'}`]||'');setMessages(nextMessages);setChatLoading(false);
      }
    }catch(error){if(isCurrent()&&(!loadHistory||request===chatContext.current.request))throw error}
  },[]);

  useEffect(()=>{void initialize()},[initialize]);
  useEffect(()=>{
    const update=()=>{setRoute(routeFromHash());setCitation(null);setOpenedArtifact(null);setSettings(false);setPodcastOpen(false);setStudyCreate(null);setTabletStudio(false)};
    window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);
  },[]);
  useEffect(()=>{if(phase==='ready'){persistNotebook(activeId);void loadCurrent(activeId,true).catch(reportError)}},[activeId,loadCurrent,phase,reportError]);
  useEffect(()=>{
    if(phase!=='ready')return;
    const refresh=()=>{if(document.hidden)return;void loadGlobal().catch(reportError);if(activeId)void loadCurrent(activeId,chatLoading).catch(reportError)};
    window.addEventListener('focus',refresh);return()=>window.removeEventListener('focus',refresh);
  },[activeId,chatLoading,loadCurrent,loadGlobal,phase,reportError]);
  useEffect(()=>{
    if(phase!=='ready'||!activeId||!jobs.some(job=>['queued','running','cancelling'].includes(job.state)))return;
    let timer:number|undefined,stopped=false,inFlight=false,failures=0;
    const controller=new AbortController();
    const poll=async()=>{
      if(stopped||inFlight||document.hidden||!navigator.onLine)return;
      inFlight=true;
      try{
        const state=await getWorkspaceState(activeId,controller.signal);if(stopped)return;failures=0;
        const prior=workspaceVersions.current;
        const refreshNotebook=prior.notebook!==state.versions.notebook||prior.sources!==state.versions.sources;
        const refreshArtifacts=prior.artifacts!==state.versions.artifacts;
        const[current,nextArtifacts]=await Promise.all([refreshNotebook?getNotebook(activeId):undefined,refreshArtifacts?getArtifacts(activeId):undefined]);
        if(stopped)return;
        workspaceVersions.current=state.versions;
        if(current)setNotebook(current);
        if(nextArtifacts)setArtifacts(nextArtifacts);
        setJobs([...state.active_jobs,...state.failed_jobs]);
        if(state.has_active_jobs&&!stopped)timer=window.setTimeout(poll,3000);
      }catch(error){if(!stopped&&!(error instanceof DOMException&&error.name==='AbortError')){failures+=1;timer=window.setTimeout(poll,Math.min(30000,3000*2**failures))}}
      finally{inFlight=false}
    };
    timer=window.setTimeout(poll,3000);
    const resume=()=>{if(document.hidden){if(timer)window.clearTimeout(timer);timer=undefined;return}if(!stopped&&!inFlight){if(timer)window.clearTimeout(timer);timer=undefined;void poll()}};
    document.addEventListener('visibilitychange',resume);window.addEventListener('online',resume);
    return()=>{stopped=true;controller.abort();if(timer)window.clearTimeout(timer);document.removeEventListener('visibilitychange',resume);window.removeEventListener('online',resume)};
  },[activeId,jobs,phase]);
  useEffect(()=>{
    if(!toast||toast.tone==='error')return;
    const timer=window.setTimeout(()=>setToast(current=>current?.id===toast.id?undefined:current),5000);return()=>window.clearTimeout(timer);
  },[toast]);

  useEffect(()=>{let inFlight=false;const refresh=()=>{if(!activeId||inFlight)return;inFlight=true;void loadCurrent(activeId).finally(()=>{inFlight=false})};window.addEventListener('sread-quality-changed',refresh);return()=>window.removeEventListener('sread-quality-changed',refresh)},[activeId,loadCurrent]);
  const improvingMessages=messages.some(message=>message.role==='assistant'&&['queued','generating','scoring','improving','rendering'].includes(message.metadata?.quality_control?.phase));
  useEffect(()=>{
    if(!conversationId)return;
    let stopped=false,inFlight=false;const context=chatContext.current,generation=context.generation;
    const refresh=async()=>{if(inFlight||document.hidden)return;inFlight=true;try{const next=await getMessages(conversationId);if(!stopped&&chatContext.current===context&&context.generation===generation)setMessages(next)}catch{/* Retry after reconnect. */}finally{inFlight=false}};
    const timer=improvingMessages?window.setInterval(()=>void refresh(),2000):undefined;
    window.addEventListener('sread-quality-changed',refresh);window.addEventListener('online',refresh);
    return()=>{stopped=true;if(timer)clearInterval(timer);window.removeEventListener('sread-quality-changed',refresh);window.removeEventListener('online',refresh)};
  },[conversationId,improvingMessages]);
  useEffect(()=>{const open=(event:Event)=>{const id=(event as CustomEvent<string>).detail;if(id)void getArtifact(id).then(setOpenedArtifact).catch(reportError)};window.addEventListener('sread-quality-open',open);return()=>window.removeEventListener('sread-quality-open',open)},[]);

  const rangePicker=notebook?<SourceRangePicker sources={notebook.sources||[]} onSave={async ids=>{await selectSources(notebook.id,ids);await loadCurrent(notebook.id)}}/>:null;
  const selected=(notebook?.sources||[]).filter(source=>source.selected&&source.state==='ready').map(source=>source.id);
  const audioProvider=providers.find(provider=>provider.role==='audio'&&provider.active);
  const audioHealth=status?.providers?.audio;
  const podcastUnavailableReason=!audioProvider?'请先配置并启用 AUDIO Provider':status===undefined?'正在检查 AUDIO Provider…':audioHealth?.ok?'':audioHealth?.message||'无法确认 AUDIO Provider 状态，请检查 Provider 配置';
  const mainProvider=providers.find(provider=>provider.role==='main'&&provider.active);
  function openAudioSettings(){setTabletStudio(false);setSettingsPanel('audio');setSettings(true)}
  async function onAsk(){
    if(!validTokenLimit(chatTokenLimit))return;
    if(!notebook||notebook.id!==chatContext.current.notebookId||chatLoading||!question.trim()||busy)return;
    if(!selected.length){notify('请先选择至少一份已完成索引的资料','error');return}
    const context=chatContext.current,generation=context.generation,request=++context.request;
    const isCurrent=()=>chatContext.current===context&&context.generation===generation&&context.request===request;
    const submittedDraftKey=draftKey.current;const content=question.trim(),optimisticId=`local-${Date.now()}`;setQuestion('');setMessages(value=>[...value,{id:optimisticId,role:'user',content}]);setBusy(true);
    try{const result=await submitChat(notebook.id,content,selected,conversationId,chatTokenLimit);if(drafts.current[submittedDraftKey]?.trim()===content)delete drafts.current[submittedDraftKey];if(!isCurrent())return;setConversationId(result.conversation_id);rememberConversation(notebook.id,result.conversation_id);const next=await getMessages(result.conversation_id);if(!isCurrent())return;setMessages(next);await loadCurrent(notebook.id)}
    catch(error){if(!isCurrent())return;setMessages(value=>value.filter(message=>message.id!==optimisticId));setQuestion(content);reportError(error)}finally{if(isCurrent())setBusy(false)}
  }

  async function onUpload(files:FileList|File[],policy:ImageProcessingPolicy=imagePolicy){
    if(!notebook){notify('请先新建或选择一个 Notebook','error');throw new Error('NO_NOTEBOOK')}
    try{await upload(notebook.id,files,policy);notify('资料已接入，正在本地解析','success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}
  }
  async function onToggle(source:Source){try{await selectSource(source.id,!source.selected);if(notebook)await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}}
  async function onDeleteSource(source:Source){try{await deleteSource(source.id);if(notebook)await loadCurrent(notebook.id);notify('资料及本地文件已删除','success')}catch(error){reportError(error);throw error}}
  async function onCreate(type:string){
    if(!notebook){notify('请先新建或选择一个 Notebook','error');return}
    if(!selected.length){notify('请先选择至少一份已完成索引的资料','error');return}
    if(type==='podcasts'){setPodcastOpen(true);return}
    if(type==='quiz'||type==='flashcards'){setStudyCreate(type==='quiz'?'quiz':'flashcard');return}
    setSummaryOpen(true);
  }
  async function onCreateStudy(options:StudyOptions){
    if(!notebook||!studyCreate)throw new Error('NO_NOTEBOOK');
    try{await createArtifact(notebook.id,studyCreate==='flashcard'?'flashcards':'quiz',selected,options);setStudyCreate(null);notify(`${studyCreate==='quiz'?'QUIZ':'FLASHCARD'} · 正在构建知识蓝图并核验证据`,'success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}
  }
  async function onCreatePodcast(options:PodcastOptions){
    if(!notebook)throw new Error('NO_NOTEBOOK');
    try{await createArtifact(notebook.id,'podcasts',selected,options);setPodcastOpen(false);notify('正在生成有原文依据的双人对话','success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}
  }
  async function onCreateNotebook(title:string){
    try{const created=await createNotebook(title);const list=await getNotebooks();setNotebooks(list);activateNotebook(created.id);location.hash='workspace';notify('Notebook 已创建','success')}catch(error){reportError(error);throw error}
  }
  async function saveProvider(id:string,body:Record<string,any>){
    try{await updateProvider(id,body);notify('配置已保存','success');await refreshProviders()}catch(error){reportError(error);throw error}
  }
  async function addProvider(body:Record<string,any>){
    try{await createProvider(body);notify(body.active?'配置已创建并启用':'配置已保存为未启用','success');await refreshProviders()}catch(error){reportError(error);throw error}
  }
  async function inspectConfiguration(draft:ProviderDraft,mode:'catalog'|'deep'):Promise<ProviderInspection>{
    try{return await inspectProvider({provider_id:draft.provider_id,role:draft.role,kind:draft.kind,base_url:draft.base_url,model:draft.model,api_key:draft.api_key||undefined,config:draft.config,mode})}catch(error){reportError(error);throw error}
  }
  async function saveRole(role:ConfigurableProviderRole,body:Record<string,any>){try{await updateProviderRole(role,body);await refreshProviders();notify(`${role.toUpperCase()} 角色设置已更新`,'success')}catch(error){reportError(error);throw error}}
  async function saveImagePolicy(policy:ImageProcessingPolicy){try{const saved=await updateImageProcessingPolicy(policy);setImagePolicy(saved);notify('图片处理策略已保存','success')}catch(error){reportError(error);throw error}}
  async function openArtifact(summary:Artifact){try{setOpenedArtifact(await getArtifact(summary.id))}catch(error){reportError(error)}}
  async function handleReview(id:string,cardId:string,rating:string){try{await reviewFlashcard(id,cardId,rating);notify(`FLASHCARD · ${rating.toUpperCase()}`,'success')}catch(error){reportError(error);throw error}}

  if(phase==='loading')return <BootScreen/>;
  if(phase==='error')return <BootScreen error={bootError} onRetry={()=>void initialize()}/>;
  if(phase==='locked')return <LoginScreen error={loginError} onLogin={async key=>{try{await login(key);setLoginError('');await initialize()}catch(error){const message=error instanceof Error?error.message:'认证失败';setLoginError(message);throw error}}}/>;

  const studio=<StudioRail reminders={<StudyReminders notebookId={notebook?.id} onOpen={id=>void getArtifact(id).then(setOpenedArtifact).catch(reportError)}/>} notebookId={notebook?.id} hasNotebook={Boolean(notebook)} selectedCount={selected.length} podcastUnavailableReason={podcastUnavailableReason} onCreate={onCreate} onOpen={summary=>void openArtifact(summary)} onConfigureAudio={openAudioSettings} artifacts={artifacts} jobs={jobs}/>;
  const guide=<GettingStarted hasModel={Boolean(mainProvider&&status?.providers?.main?.ok)} hasNotebook={Boolean(notebook)} sourceCount={selected.length} hasResult={artifacts.length>0||messages.some(message=>message.role==='assistant')} onSettings={()=>{setSettingsPanel('models');setSettings(true)}} onCreate={()=>{location.hash='notebooks'}} onImport={()=>{setWorkspacePanel('sources');requestAnimationFrame(()=>document.querySelector<HTMLInputElement>('.upload-zone input')?.click())}} onTry={()=>{setWorkspacePanel('chat');setQuestion('请提炼核心结论与适用限制，并给出原文引用。')}}/>;
  const hasResult=artifacts.length>0||messages.some(message=>message.role==='assistant');
  return <div className={`shell route-${route}`}>
    <Header route={route} notebook={notebook} notebooks={notebooks} status={status} onSelect={activateNotebook} onCreate={onCreateNotebook} onSettings={()=>{setSettingsPanel('models');setSettings(true)}}/>
    <WeeklyBudget providers={providers} running={busy||jobs.some(job=>['running','queued','cancelling'].includes(job.state))} revision={`${messages.length}:${jobs.map(job=>`${job.id}:${job.state}`).join(',')}:${providers.length}`}/>
    {route==='jobs'?<JobsPage onOpenChat={(conversationId,messageId)=>setOpenedChat({conversationId,messageId})} onError={reportError} onOpenArtifact={id=>void getArtifact(id).then(setOpenedArtifact).catch(reportError)}/>:route==='notebooks'?<NotebooksPage onError={reportError} onNotify={notify} onChanged={refreshNotebooks} onOpen={id=>{activateNotebook(id);location.hash='workspace'}}/>:<section className="workspace-shell">

      <nav className="workspace-tabs" aria-label="工作区面板">
        <button className={workspacePanel==='chat'?'active':''} aria-pressed={workspacePanel==='chat'} onClick={()=>setWorkspacePanel('chat')}><MessageSquare/>对话</button>
        <button className={workspacePanel==='sources'?'active':''} aria-pressed={workspacePanel==='sources'} onClick={()=>setWorkspacePanel('sources')}><BookOpen/>资料 <b>{selected.length}</b></button>
        <button className={workspacePanel==='studio'?'active':''} aria-pressed={workspacePanel==='studio'} onClick={()=>setWorkspacePanel('studio')}><Headphones/>学习与输出</button>
      </nav>
      <div className={`workspace workspace-panel-${workspacePanel}`}>
        <SourceRail providers={providers} hasNotebook={Boolean(notebook)} sources={notebook?.sources||[]} imagePolicy={imagePolicy} onUpload={onUpload} onToggle={onToggle} onDelete={onDeleteSource} onNotify={notify}/>
        <ChatPanel tools={notebook?<div className="workspace-tools"><ConversationHistory notebookId={notebook.id} conversationId={conversationId} onOpen={openConversation}/><HistoricalScope scope={messages.filter(m=>m.role==='user').at(-1)?.metadata?.source_scope} sources={notebook.sources||[]} onRestore={async ids=>{await selectSources(notebook.id,ids);await loadCurrent(notebook.id)}}/><SourceRangePicker sources={notebook.sources||[]} onSave={async ids=>{await selectSources(notebook.id,ids);await loadCurrent(notebook.id)}}/></div>:null} guide={guide} showOnboarding={!hasResult&&!messages.length} setupRequired={!mainProvider||!notebook||!selected.length} notebookId={notebook?.id} sourceIds={selected} tokenLimit={chatTokenLimit} onTokenLimit={setChatTokenLimit} hasNotebook={Boolean(notebook)} selectedCount={selected.length} messages={messages} question={question} setQuestion={editQuestion} onAsk={onAsk} busy={busy} loading={chatLoading} onCitation={setCitation} onNewConversation={newConversation} onOpenStudio={()=>setTabletStudio(true)}/>
        {studio}
      </div>
    </section>}
    <footer><span>© 2077 SANDEVISTAN RESEARCH SYSTEMS</span><b>LOCAL-FIRST // SOURCE-GROUNDED // TRACEABLE</b><span>BUILD {status?.version||'—'}</span></footer>
    {summaryOpen&&notebook?<SummaryCreateModal rangePicker={rangePicker} notebookId={notebook.id} sourceIds={selected} onClose={()=>setSummaryOpen(false)} onCreate={async options=>{try{await createArtifact(notebook.id,'summary',selected,options);setSummaryOpen(false);notify('摘要任务已进入队列','success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}}}/>:null}
    {openedChat?<ChatResultDrawer {...openedChat} onContinue={()=>{void getConversation(openedChat.conversationId).then(async value=>{rememberConversation(value.notebook_id,value.id);if(value.notebook_id===activeId)await openConversation(value.id);else activateNotebook(value.notebook_id);setOpenedChat(undefined);location.hash='workspace'}).catch(reportError)}} onClose={()=>setOpenedChat(undefined)} onCitation={setCitation}/>:null}
    <CitationDrawer citation={citation} onClose={()=>setCitation(null)}/>
    <ArtifactDrawer key={openedArtifact?.id||'closed'} artifact={openedArtifact} onClose={()=>setOpenedArtifact(null)} onCitation={setCitation} onSubmitQuiz={submitQuiz} onReview={handleReview}/>
    {settings?<Suspense fallback={<div role="status">正在打开设置…</div>}><SettingsDrawer initialPanel={settingsPanel} status={status} providers={providers} roles={providerRoles} imagePolicy={imagePolicy} notebookId={notebook?.id} sourceIds={selected} onClose={()=>setSettings(false)} onSave={saveProvider} onCreate={addProvider} onInspect={inspectConfiguration} onSaveRole={saveRole} onSaveImagePolicy={saveImagePolicy}/></Suspense>:null}
    {podcastOpen?<PodcastCreateModal rangePicker={rangePicker} notebookId={notebook?.id} sourceIds={selected} provider={audioProvider} unavailableReason={podcastUnavailableReason||undefined} sourceCount={selected.length} onClose={()=>setPodcastOpen(false)} onCreate={onCreatePodcast}/>:null}
    {studyCreate?<StudyCreateModal rangePicker={rangePicker} notebookId={notebook?.id} sourceIds={selected} kind={studyCreate} provider={mainProvider} sourceCount={selected.length} onClose={()=>setStudyCreate(null)} onCreate={onCreateStudy}/>:null}
    {tabletStudio?<Overlay className="tablet-studio-drawer" label="Studio" onClose={()=>setTabletStudio(false)}>{studio}</Overlay>:null}
    {providerRefreshError?<div className="toast toast-error" role="alert">配置已保存，状态刷新失败。<button onClick={()=>void refreshProviders()}>重试状态刷新</button></div>:null}
    {toast?<div className={`toast toast-${toast.tone}`} role={toast.tone==='error'?'alert':'status'}><span>{toast.message}</span><button aria-label="关闭提示" onClick={()=>setToast(undefined)}>×</button></div>:null}
  </div>;
}
