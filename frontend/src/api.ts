export type Source={id:string;filename:string;state:string;selected:number;page_count:number;parser?:string;error?:string;metadata?:Record<string,any>};
export type Notebook={id:string;title:string;description:string;state?:string;sources?:Source[];source_count?:number;source_bytes?:number;artifact_count?:number;active_jobs?:number;cleanup_error?:string};
export type NotebookDeleteResult={id:string;accepted:boolean;operation_id?:string;error?:string};
export type NotebookBatchDeleteResponse={items:NotebookDeleteResult[]};
export type Citation={id:string;filename:string;locator:Record<string,unknown>;quote:string;source_id:string};
export type Job={id:string;notebook_id?:string;notebook_title?:string;display_name:string;kind:string;state:string;stage:string;stage_code:string;progress:number;stage_current?:number;stage_total?:number;stage_unit?:string;progress_basis?:string;error?:string;result?:Record<string,any>;created_at:string;updated_at:string;started_at?:string;finished_at?:string;eta:{status:'learning'|'ready';sample_count:number;confidence?:string;queue_position:number;remaining_seconds?:number;remaining_range?:number[]}};
export type Page<T>={items:T[];page:number;page_size:number;total:number;pages:number};
export type Artifact={id:string;type:'summary'|'quiz'|'flashcard'|'podcast';title:string;status:string;payload:Record<string,any>;citations:Citation[];media_url?:string;language:string};
export type ProviderRole='main'|'vlm'|'tts';
export type ProviderKind='ollama'|'openai'|'sandevistan_tts'|'openai_tts';
export type StudyDifficulty='easy'|'medium'|'hard'|'mixed';
export type StudyOptions={count:number;difficulty:StudyDifficulty;language:'auto'|'zh-CN'|'en';custom_prompt:string};
export type StudySession={id:string;artifact_id:string;kind:'quiz'|'flashcard';mode:'all'|'missed'|'due'|'same';status:'active'|'complete';items:Array<Record<string,any>>;progress:{current:number;total:number};created_at:string;updated_at:string};
export type TokenLimits={model_context_tokens?:number|null;effective_context_tokens:number;max_input_tokens?:number|null;max_output_tokens:number;context_source:'manual'|'ollama_runtime'|'ollama_modelfile'|'provider_metadata'|'fallback'|string;output_source:'manual'|'provider_metadata'|'derived'|string;image_tokens_per_image?:number;probed_at?:string};
export type Provider={id:string;name:string;role:ProviderRole;kind:ProviderKind;base_url:string;model:string;active:number;has_api_key:boolean;capabilities:Record<string,any>&{study_generation?:{tier:'lite'|'full';source:string;reason:string;parameter_count?:number|null;supports_difficulties:string[]}};config:Record<string,any>};
export type ProviderModel={id:string;name:string;installed?:boolean;devices?:Array<{id:string;available:boolean;precision?:string;reason?:string}>;controls?:Record<string,any>;details?:Record<string,any>;token_limits?:Partial<TokenLimits>};
export type ProviderInspection={status:'passed'|'warning'|'failed';connection_ok:boolean;activation_eligible:boolean;latency_ms:number;catalog_supported:boolean;models:ProviderModel[];capabilities:Record<string,any>;recommended?:{model?:string;compute_device?:string}|null;warning?:string|null;error?:{code:string;stage:string;message:string;hint:string;upstream_status?:number|null}|null};
export type ProviderDraft={provider_id?:string;name:string;role:ProviderRole;kind:ProviderKind;base_url:string;model:string;api_key?:string;config:Record<string,any>};
export type PodcastOptions={duration_mode:'auto'|'fixed';minutes?:5|10|20|30;language:'zh-CN'|'auto'|'en';focus:string};
export type AuthStatus={required:boolean;authenticated:boolean};

const jsonHeaders={'Content-Type':'application/json'};
function storedToken(){try{return localStorage.getItem('sread_token')}catch{return null}}
function saveToken(token:string){try{localStorage.setItem('sread_token',token)}catch{/* The secure cookie remains the source of truth. */}}
export function clearSession(){try{localStorage.removeItem('sread_token')}catch{/* Storage may be unavailable. */}}
export async function authStatus(){const headers=new Headers();const token=storedToken();if(token)headers.set('Authorization',`Bearer ${token}`);const response=await fetch('/auth/status',{headers});if(!response.ok)throw new Error('无法确认访问状态');return response.json() as Promise<AuthStatus>}
export async function api<T>(path:string,init?:RequestInit):Promise<T>{const token=storedToken();const headers=new Headers(init?.headers);if(token)headers.set('Authorization',`Bearer ${token}`);const response=await fetch(`/api${path}`,{...init,headers});if(response.status===401){clearSession();throw new Error('AUTH_REQUIRED')}if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));const detail=body.detail;const message=typeof detail==='string'?detail:detail?.message||detail?.inspection?.error?.message||body.error||'请求失败';throw new Error(message)}if(response.status===204)return undefined as T;return response.json()}
export async function login(access_key:string){const response=await fetch('/auth/login',{method:'POST',headers:jsonHeaders,body:JSON.stringify({access_key})});if(!response.ok)throw new Error('访问密钥错误');const result=await response.json();saveToken(result.token);return result}
export const getNotebooks=()=>api<Notebook[]>('/notebooks');
export const createNotebook=(title:string,description='')=>api<Notebook>('/notebooks',{method:'POST',headers:jsonHeaders,body:JSON.stringify({title,description})});
export const deleteNotebook=(id:string)=>api<void>(`/notebooks/${id}`,{method:'DELETE'});
export const batchDeleteNotebooks=(notebook_ids:string[])=>api<NotebookBatchDeleteResponse>('/notebooks/batch-delete',{method:'POST',headers:jsonHeaders,body:JSON.stringify({notebook_ids})});
export const getNotebook=(id:string)=>api<Notebook>(`/notebooks/${id}`);
export const getStatus=()=>api<any>('/status');
export const selectSource=(id:string,selected:boolean)=>api(`/sources/${id}/selection`,{method:'PATCH',headers:jsonHeaders,body:JSON.stringify({selected})});
export const deleteSource=(id:string)=>api<void>(`/sources/${id}`,{method:'DELETE'});
export async function upload(id:string,files:FileList|File[]){const body=new FormData();Array.from(files).forEach(file=>body.append('files',file));return api(`/notebooks/${id}/sources`,{method:'POST',body})}
export const ask=(id:string,question:string,source_ids:string[],conversation_id?:string)=>api<any>(`/notebooks/${id}/chat`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({question,source_ids,conversation_id,language:'auto'})});
export const getConversations=(id:string)=>api<any[]>(`/notebooks/${id}/conversations`);
export const getMessages=(id:string)=>api<any[]>(`/conversations/${id}/messages`);
export const createArtifact=(id:string,type:string,source_ids:string[],options?:PodcastOptions|StudyOptions)=>api<any>(`/notebooks/${id}/${type}`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({source_ids,...options})});
export const getArtifacts=(id:string)=>api<Artifact[]>(`/notebooks/${id}/artifacts`);
export const getArtifact=(id:string)=>api<Artifact>(`/artifacts/${id}`);
export const getJobsPage=(params:Record<string,string|number|undefined>={})=>{const query=new URLSearchParams();Object.entries(params).forEach(([key,value])=>value!==undefined&&query.set(key,String(value)));return api<Page<Job>>(`/jobs?${query}`)};
export const getJobs=async(id?:string)=>(await getJobsPage({notebook_id:id,page_size:100})).items;
export const getJob=(id:string)=>api<Job>(`/jobs/${id}`);
export const getJobEvents=(id:string)=>api<any[]>(`/jobs/${id}/events`);
export const cancelJob=(id:string)=>api(`/jobs/${id}/cancel`,{method:'POST'});
export const deleteJob=(id:string)=>api<{deleted:boolean;bytes_freed:number}>(`/jobs/${id}`,{method:'DELETE'});
export const batchPurgeJobs=(job_ids:string[])=>api<any>('/jobs/batch-purge',{method:'POST',headers:jsonHeaders,body:JSON.stringify({job_ids})});
export const getNotebookManagement=(params:Record<string,string|number|undefined>={})=>{const query=new URLSearchParams();Object.entries(params).forEach(([key,value])=>value!==undefined&&query.set(key,String(value)));return api<Page<Notebook>>(`/notebook-management?${query}`)};
export const retryNotebookCleanup=(id:string)=>api(`/notebooks/${id}/cleanup/retry`,{method:'POST'});
export const getSummary=(id:string)=>api<any>(`/notebooks/${id}/summary`);
export const submitQuiz=(id:string,answers:Record<string,number>)=>api<any>(`/artifacts/${id}/quiz/submit`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({answers})});
export const reviewFlashcard=(id:string,card_id:string,rating:string)=>api<any>(`/artifacts/${id}/flashcards/review`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({card_id,rating})});
export const createStudySession=(id:string,mode:'all'|'missed'|'due'|'same'='all',shuffle=false)=>api<StudySession>(`/artifacts/${id}/study-sessions`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({mode,shuffle})});
export const getStudySession=(id:string)=>api<StudySession>(`/study-sessions/${id}`);
export const answerQuizItem=(id:string,item_id:string,option_index:number)=>api<{result:Record<string,any>;session:StudySession}>(`/study-sessions/${id}/quiz-answer`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({item_id,option_index})});
export const reviewStudyCard=(id:string,item_id:string,rating:'again'|'hard'|'good'|'easy')=>api<any>(`/study-sessions/${id}/flashcard-review`,{method:'POST',headers:jsonHeaders,body:JSON.stringify({item_id,rating})});
export const suspendFlashcard=(artifactId:string,cardId:string)=>api<void>(`/artifacts/${artifactId}/flashcards/${cardId}`,{method:'DELETE'});
export const flashcardsCsvUrl=(artifactId:string)=>`/api/artifacts/${artifactId}/flashcards.csv`;
export const getProviders=()=>api<Provider[]>('/providers');
export const createProvider=(body:Record<string,any>)=>api<{id:string;active:boolean;inspection?:ProviderInspection}>('/providers',{method:'POST',headers:jsonHeaders,body:JSON.stringify(body)});
export const updateProvider=(id:string,body:Record<string,any>)=>api(`/providers/${id}`,{method:'PATCH',headers:jsonHeaders,body:JSON.stringify(body)});
export const inspectProvider=(body:Omit<ProviderDraft,'name'>&{mode:'catalog'|'deep'})=>api<ProviderInspection>('/providers/inspect',{method:'POST',headers:jsonHeaders,body:JSON.stringify(body)});
export const testProvider=(id:string)=>api<any>(`/providers/${id}/test`,{method:'POST'});
export const probeProvider=(id:string)=>api<any>(`/providers/${id}/probe`,{method:'POST'});
