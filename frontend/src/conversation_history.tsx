import {useEffect,useState} from 'react';
import {getConversations,type Source} from './api';
import {Overlay} from './ui';
export function ConversationHistory({notebookId,conversationId,onOpen}:{notebookId:string;conversationId?:string;onOpen:(id:string)=>Promise<void>}){
  const[open,setOpen]=useState(false),[rows,setRows]=useState<Array<{id:string;title:string;updated_at:string}>>([]),[query,setQuery]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  useEffect(()=>{if(!open)return;let stale=false;setError('');void getConversations(notebookId).then(value=>{if(!stale)setRows(value)}).catch(e=>{if(!stale)setError(e.message)});return()=>{stale=true}},[notebookId,open]);
  return <><button onClick={()=>setOpen(true)}>历史对话</button>{open?<Overlay label="历史对话" title={<>历史对话</>} onClose={()=>setOpen(false)}><label>搜索标题或日期<input value={query} onChange={e=>setQuery(e.target.value)}/></label><p>打开对话保留当前资料范围；历史范围可另行恢复。</p>{error?<p role="alert">{error}</p>:null}{rows.filter(row=>`${row.title} ${row.updated_at}`.toLowerCase().includes(query.toLowerCase())).map(row=><button className="artifact" aria-current={row.id===conversationId?'true':undefined} disabled={busy} key={row.id} onClick={()=>{setBusy(true);void onOpen(row.id).then(()=>setOpen(false)).catch(e=>setError(e.message)).finally(()=>setBusy(false))}}><span>{row.title}<small>{new Date(row.updated_at).toLocaleString()}</small></span></button>)}{!rows.length?<p>暂无历史对话</p>:null}</Overlay>:null}</>
}
export function HistoricalScope({scope,sources,onRestore}:{scope?:Array<{id:string;revision_id:string;filename:string}>|null;sources:Source[];onRestore:(ids:string[])=>Promise<void>}){
  const[open,setOpen]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState('');
  if(!scope)return <small>历史资料范围未知；后续问题使用当前勾选资料。</small>;
  const usable=scope.filter(old=>sources.some(s=>s.id===old.id&&s.revision_id===old.revision_id&&s.state==='ready'));
  const same=usable.length===scope.length&&sources.filter(s=>s.selected&&s.state==='ready').length===scope.length&&usable.every(old=>sources.some(s=>s.id===old.id&&s.selected));
  if(same)return <small>当前资料范围与本轮历史一致。</small>;
  return <><p>当前资料范围与历史不同。<button onClick={()=>{setError('');setOpen(true)}}>查看并恢复历史范围</button></p>{open?<Overlay label="恢复历史资料范围" title={<>恢复历史资料范围</>} onClose={()=>setOpen(false)}>{scope.map(old=><p key={old.id}>{old.filename} · {usable.includes(old)?'可恢复':sources.some(s=>s.id===old.id)?'修订变化或未就绪':'已删除'}</p>)}<p>仅恢复可用的相同修订，不根据引用片段推断范围。</p>{error?<p role="alert">{error}</p>:null}<button disabled={busy||!usable.length} onClick={()=>{setBusy(true);void onRestore(usable.map(s=>s.id)).then(()=>setOpen(false)).catch(e=>setError(e.message)).finally(()=>setBusy(false))}}>应用可用范围（{usable.length}）</button></Overlay>:null}</>
}
