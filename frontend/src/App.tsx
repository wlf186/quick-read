import {useCallback,useEffect,useRef,useState} from 'react';
import {BookOpen,Headphones,MessageSquare} from 'lucide-react';
import {ArtifactDrawer,ChatPanel,CitationDrawer,Header,LoginScreen,PodcastCreateModal,SourceRail,StudioRail,StudyCreateModal} from './components';
import {authStatus,ask,createArtifact,createNotebook,createProvider,deleteSource,getArtifacts,getConversations,getJobs,getMessages,getNotebook,getNotebooks,getProviders,getStatus,inspectProvider,login,reviewFlashcard,selectSource,submitQuiz,updateProvider,upload,type Artifact,type Citation,type Job,type Notebook,type PodcastOptions,type Provider,type ProviderDraft,type ProviderInspection,type Source,type StudyOptions} from './api';
import {JobsPage,NotebooksPage} from './management';
import {SettingsDrawer} from './provider_settings';
import {Overlay} from './ui';

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
  const[messages,setMessages]=useState<any[]>([]);
  const[conversationId,setConversationId]=useState<string>();
  const[question,setQuestion]=useState('');
  const[busy,setBusy]=useState(false);
  const[citation,setCitation]=useState<Citation|null>(null);
  const[openedArtifact,setOpenedArtifact]=useState<Artifact|null>(null);
  const[settings,setSettings]=useState(false);
  const[podcastOpen,setPodcastOpen]=useState(false);
  const[studyCreate,setStudyCreate]=useState<'quiz'|'flashcard'|null>(null);
  const[tabletStudio,setTabletStudio]=useState(false);
  const[workspacePanel,setWorkspacePanel]=useState<WorkspacePanel>('chat');
  const[artifacts,setArtifacts]=useState<Artifact[]>([]);
  const[jobs,setJobs]=useState<Job[]>([]);
  const[toast,setToast]=useState<ToastState>();
  const[loginError,setLoginError]=useState('');
  const[bootError,setBootError]=useState('');
  const loadSequence=useRef(0);

  const notify=useCallback((message:string,tone:ToastState['tone']='info')=>setToast({id:Date.now(),message,tone}),[]);
  const reportError=useCallback((error:unknown)=>{
    const message=error instanceof Error?error.message:'请求失败';
    if(message==='AUTH_REQUIRED'){setPhase('locked');setLoginError('会话已失效，请重新输入访问密钥。')}
    else notify(message,'error');
  },[notify]);

  const reconcileNotebooks=useCallback((list:Notebook[])=>{
    setNotebooks(list);
    setActiveId(current=>{
      if(current&&list.some(item=>item.id===current))return current;
      const next=list[0]?.id||'';persistNotebook(next);return next;
    });
  },[]);

  const initialize=useCallback(async()=>{
    setBootError('');setPhase('loading');
    try{
      const access=await authStatus();
      if(access.required&&!access.authenticated){setPhase('locked');return}
      const[list,nextProviders]=await Promise.all([getNotebooks(),getProviders()]);
      setNotebooks(list);setProviders(nextProviders);
      const preferred=storedNotebook();const next=list.some(item=>item.id===preferred)?preferred:list[0]?.id||'';
      setActiveId(next);persistNotebook(next);setPhase('ready');
      void getStatus().then(setStatus).catch(()=>setStatus({providers:{}}));
    }catch(error){
      if(error instanceof Error&&error.message==='AUTH_REQUIRED'){setPhase('locked');return}
      const message=error instanceof Error?error.message:'无法启动应用';setBootError(message);setPhase('error');
    }
  },[]);

  const loadGlobal=useCallback(async()=>{
    const[list,nextProviders]=await Promise.all([getNotebooks(),getProviders()]);
    reconcileNotebooks(list);setProviders(nextProviders);
    void getStatus().then(setStatus).catch(()=>setStatus({providers:{}}));
  },[reconcileNotebooks]);

  const refreshNotebooks=useCallback(async()=>reconcileNotebooks(await getNotebooks()),[reconcileNotebooks]);

  const loadCurrent=useCallback(async(id:string,loadHistory=false)=>{
    const sequence=++loadSequence.current;
    if(!id){setNotebook(undefined);setArtifacts([]);setJobs([]);setConversationId(undefined);setMessages([]);return}
    const historyPromise=loadHistory?getConversations(id):Promise.resolve<any[]|null>(null);
    const[current,nextArtifacts,nextJobs,conversations]=await Promise.all([getNotebook(id),getArtifacts(id),getJobs(id),historyPromise]);
    if(sequence!==loadSequence.current)return;
    setNotebook(current);setArtifacts(nextArtifacts);setJobs(nextJobs);
    if(loadHistory&&conversations){
      const latest=conversations[0];
      if(latest){const nextMessages=await getMessages(latest.id);if(sequence!==loadSequence.current)return;setConversationId(latest.id);setMessages(nextMessages)}
      else{setConversationId(undefined);setMessages([])}
    }
  },[]);

  useEffect(()=>{void initialize()},[initialize]);
  useEffect(()=>{
    const update=()=>{setRoute(routeFromHash());setCitation(null);setOpenedArtifact(null);setSettings(false);setPodcastOpen(false);setStudyCreate(null);setTabletStudio(false)};
    window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);
  },[]);
  useEffect(()=>{if(phase==='ready'){persistNotebook(activeId);void loadCurrent(activeId,true).catch(reportError)}},[activeId,loadCurrent,phase,reportError]);
  useEffect(()=>{
    if(phase!=='ready')return;
    const timer=window.setInterval(()=>{void loadGlobal().catch(reportError);if(activeId)void loadCurrent(activeId,false).catch(reportError)},3000);
    return()=>window.clearInterval(timer);
  },[activeId,loadCurrent,loadGlobal,phase,reportError]);
  useEffect(()=>{
    if(!toast||toast.tone==='error')return;
    const timer=window.setTimeout(()=>setToast(current=>current?.id===toast.id?undefined:current),5000);return()=>window.clearTimeout(timer);
  },[toast]);

  const selected=(notebook?.sources||[]).filter(source=>source.selected&&source.state==='ready').map(source=>source.id);
  async function onAsk(){
    if(!notebook||!question.trim()||busy)return;
    if(!selected.length){notify('请先选择至少一份已完成索引的资料','error');return}
    const content=question.trim(),optimisticId=`local-${Date.now()}`;setQuestion('');setMessages(value=>[...value,{id:optimisticId,role:'user',content}]);setBusy(true);
    try{const result=await ask(notebook.id,content,selected,conversationId);setConversationId(result.conversation_id);setMessages(value=>[...value,{role:'assistant',...result}])}
    catch(error){setMessages(value=>value.filter(message=>message.id!==optimisticId));setQuestion(content);reportError(error)}finally{setBusy(false)}
  }
  async function onUpload(files:FileList|File[]){
    if(!notebook){notify('请先新建或选择一个 Notebook','error');throw new Error('NO_NOTEBOOK')}
    try{await upload(notebook.id,files);notify('资料已接入，正在本地解析','success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}
  }
  async function onToggle(source:Source){try{await selectSource(source.id,!source.selected);if(notebook)await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}}
  async function onDeleteSource(source:Source){try{await deleteSource(source.id);if(notebook)await loadCurrent(notebook.id);notify('资料及本地文件已删除','success')}catch(error){reportError(error);throw error}}
  async function onCreate(type:string){
    if(!notebook){notify('请先新建或选择一个 Notebook','error');return}
    if(!selected.length){notify('请先选择至少一份已完成索引的资料','error');return}
    if(type==='podcasts'){setPodcastOpen(true);return}
    if(type==='quiz'||type==='flashcards'){setStudyCreate(type==='quiz'?'quiz':'flashcard');return}
    try{await createArtifact(notebook.id,type,selected);notify('生成任务已进入本地队列','success');await loadCurrent(notebook.id)}catch(error){reportError(error)}
  }
  async function onCreateStudy(options:StudyOptions){
    if(!notebook||!studyCreate)throw new Error('NO_NOTEBOOK');
    try{await createArtifact(notebook.id,studyCreate==='flashcard'?'flashcards':'quiz',selected,options);setStudyCreate(null);notify(`${studyCreate==='quiz'?'QUIZ':'FLASHCARD'} · 正在构建知识蓝图并核验证据`,'success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}
  }
  async function onCreatePodcast(options:PodcastOptions){
    if(!notebook)throw new Error('NO_NOTEBOOK');
    try{await createArtifact(notebook.id,'podcasts',selected,options);setPodcastOpen(false);notify('PODCAST V2 · 正在构建全篇证据地图','success');await loadCurrent(notebook.id)}catch(error){reportError(error);throw error}
  }
  async function onCreateNotebook(title:string){
    try{const created=await createNotebook(title);const list=await getNotebooks();setNotebooks(list);setActiveId(created.id);persistNotebook(created.id);location.hash='workspace';notify('Notebook 已创建','success')}catch(error){reportError(error);throw error}
  }
  async function saveProvider(id:string,body:Record<string,any>){
    try{await updateProvider(id,body);setProviders(await getProviders());notify('Provider 配置已保存','success');void getStatus().then(setStatus).catch(()=>setStatus({providers:{}}))}catch(error){reportError(error);throw error}
  }
  async function addProvider(body:Record<string,any>){
    try{await createProvider(body);setProviders(await getProviders());notify(body.active?'Provider 已创建并启用':'Provider 已保存为未启用','success');void getStatus().then(setStatus).catch(()=>setStatus({providers:{}}))}catch(error){reportError(error);throw error}
  }
  async function inspectConfiguration(draft:ProviderDraft,mode:'catalog'|'deep'):Promise<ProviderInspection>{
    try{return await inspectProvider({provider_id:draft.provider_id,role:draft.role,kind:draft.kind,base_url:draft.base_url,model:draft.model,api_key:draft.api_key||undefined,config:draft.config,mode})}catch(error){reportError(error);throw error}
  }
  async function handleReview(id:string,cardId:string,rating:string){try{await reviewFlashcard(id,cardId,rating);notify(`FLASHCARD · ${rating.toUpperCase()}`,'success')}catch(error){reportError(error);throw error}}

  if(phase==='loading')return <BootScreen/>;
  if(phase==='error')return <BootScreen error={bootError} onRetry={()=>void initialize()}/>;
  if(phase==='locked')return <LoginScreen error={loginError} onLogin={async key=>{try{await login(key);setLoginError('');await initialize()}catch(error){const message=error instanceof Error?error.message:'认证失败';setLoginError(message);throw error}}}/>;

  const ttsProvider=providers.find(provider=>provider.role==='tts'&&provider.active);
  const mainProvider=providers.find(provider=>provider.role==='main'&&provider.active);
  const studio=<StudioRail hasNotebook={Boolean(notebook)} selectedCount={selected.length} onCreate={onCreate} onOpen={setOpenedArtifact} artifacts={artifacts} jobs={jobs}/>;
  return <div className={`shell route-${route}`}>
    <Header route={route} notebook={notebook} notebooks={notebooks} status={status} onSelect={setActiveId} onCreate={onCreateNotebook} onSettings={()=>setSettings(true)}/>
    {route==='jobs'?<JobsPage onError={reportError}/>:route==='notebooks'?<NotebooksPage onError={reportError} onNotify={notify} onChanged={refreshNotebooks} onOpen={id=>{setActiveId(id);location.hash='workspace'}}/>:<section className="workspace-shell">
      <nav className="workspace-tabs" aria-label="工作区面板">
        <button className={workspacePanel==='chat'?'active':''} aria-pressed={workspacePanel==='chat'} onClick={()=>setWorkspacePanel('chat')}><MessageSquare/>对话</button>
        <button className={workspacePanel==='sources'?'active':''} aria-pressed={workspacePanel==='sources'} onClick={()=>setWorkspacePanel('sources')}><BookOpen/>资料 <b>{selected.length}</b></button>
        <button className={workspacePanel==='studio'?'active':''} aria-pressed={workspacePanel==='studio'} onClick={()=>setWorkspacePanel('studio')}><Headphones/>Studio</button>
      </nav>
      <div className={`workspace workspace-panel-${workspacePanel}`}>
        <SourceRail hasNotebook={Boolean(notebook)} sources={notebook?.sources||[]} onUpload={onUpload} onToggle={onToggle} onDelete={onDeleteSource} onNotify={notify}/>
        <ChatPanel hasNotebook={Boolean(notebook)} selectedCount={selected.length} messages={messages} question={question} setQuestion={setQuestion} onAsk={onAsk} busy={busy} onCitation={setCitation} onNewConversation={()=>{setConversationId(undefined);setMessages([])}} onOpenStudio={()=>setTabletStudio(true)}/>
        {studio}
      </div>
    </section>}
    <footer><span>© 2077 SANDEVISTAN RESEARCH SYSTEMS</span><b>LOCAL-FIRST // SOURCE-GROUNDED // TRACEABLE</b><span>BUILD 0.4.0</span></footer>
    <CitationDrawer citation={citation} onClose={()=>setCitation(null)}/>
    <ArtifactDrawer key={openedArtifact?.id||'closed'} artifact={openedArtifact} onClose={()=>setOpenedArtifact(null)} onCitation={setCitation} onSubmitQuiz={submitQuiz} onReview={handleReview}/>
    {settings?<SettingsDrawer status={status} providers={providers} onClose={()=>setSettings(false)} onSave={saveProvider} onCreate={addProvider} onInspect={inspectConfiguration}/>:null}
    {podcastOpen?<PodcastCreateModal provider={ttsProvider} sourceCount={selected.length} onClose={()=>setPodcastOpen(false)} onCreate={onCreatePodcast}/>:null}
    {studyCreate?<StudyCreateModal kind={studyCreate} provider={mainProvider} sourceCount={selected.length} onClose={()=>setStudyCreate(null)} onCreate={onCreateStudy}/>:null}
    {tabletStudio?<Overlay className="tablet-studio-drawer" label="Studio" onClose={()=>setTabletStudio(false)}><button className="drawer-close" data-autofocus onClick={()=>setTabletStudio(false)}>关闭 ×</button>{studio}</Overlay>:null}
    {toast?<div className={`toast toast-${toast.tone}`} role={toast.tone==='error'?'alert':'status'}><span>{toast.message}</span><button aria-label="关闭提示" onClick={()=>setToast(undefined)}>×</button></div>:null}
  </div>;
}
