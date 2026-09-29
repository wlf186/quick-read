export type QualityLevel='low'|'medium'|'high'|'extreme';
export const qualityLevels:QualityLevel[]=['low','medium','high','extreme'];
export const qualityLabels:Record<QualityLevel,string>={low:'快速',medium:'均衡',high:'精细',extreme:'更多核查'};
export const qualityTargets:Record<QualityLevel,number>={low:60,medium:75,high:85,extreme:92};
const memory:Record<string,QualityLevel>={};
export function qualityPreference(kind:string):QualityLevel{const key=kind==='podcasts'?'podcast':kind==='flashcards'?'flashcard':kind;try{const value=localStorage.getItem(`sread-quality-v1-${key}`);if(qualityLevels.includes(value as QualityLevel))return value as QualityLevel}catch{/* Optional browser preference. */}return memory[key]||'low'}
export function rememberQuality(kind:string,value:QualityLevel){memory[kind]=value;try{localStorage.setItem(`sread-quality-v1-${kind}`,value)}catch{/* Continue with in-memory preference. */}window.dispatchEvent(new Event('sread-quality-preference'))}
