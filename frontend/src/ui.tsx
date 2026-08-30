import {type ReactNode,useEffect,useId,useMemo,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import type {Citation} from './api';

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
  danger?:boolean;
};

export function ConfirmDialog({title,description,confirmLabel,onCancel,onConfirm,requireText,danger=true}:ConfirmDialogProps){
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
    {requireText!==undefined?<label>输入完整名称确认
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
