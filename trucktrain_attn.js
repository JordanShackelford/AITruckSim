'use strict';
const { PPO } = require('./ppo.js');
const { AttnNet } = require('./attn.js');
const { TruckEnv, OBS_DIM, ACT_DIM } = require('./env.js');
function mkrng(s){ return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }

const NET=process.env.NET||'attn';
const ITERS=+(process.env.ITERS||50);
const rng=mkrng(+(process.env.SEED||11));
// env with per-episode randomized obstacle count (train on a MIX -> tests count-invariance)
const env=new TruckEnv(rng,{nObs:3,goalX:1100,maxT:700});
const TRAIN_COUNTS=(process.env.COUNTS||'2,3,4').split(',').map(Number);

let cfg={rng,lr:3e-4,initStd:0.7,entCoef:0.005,gamma:0.99,lam:0.95,epochs:8,mb:128,targetKL:0.05};
if(NET==='attn'){
  const head=[64,64];
  cfg.makePi=(o,a,r)=>new AttnNet({rng:r,egoDim:6,K:5,agentDim:5,dh:16,d:16,dv:16,head,outDim:a,outScale:0.01});
  cfg.makeVf=(o,a,r)=>new AttnNet({rng:r,egoDim:6,K:5,agentDim:5,dh:16,d:16,dv:16,head,outDim:1,outScale:1.0});
}else{ cfg.hidden=[64,64]; }
const agent=new PPO(OBS_DIM,ACT_DIM,cfg);

const T=2048, LR0=+(process.env.LR||2e-4), ENT0=0.006;
function setEpCount(){ env.nObs=TRAIN_COUNTS[(Math.random()*TRAIN_COUNTS.length)|0]; }
function rollout(){
  const obs=[],act=[],logp=[],rew=[],val=[],done=[];
  setEpCount(); let o=env.reset(), epRet=0; const rets=[],ends={};
  for(let t=0;t<T;t++){
    const {action,logProb,value}=agent.act(o);
    const r=env.step(action);
    obs.push(o);act.push(action);logp.push(logProb);val.push(value);rew.push(r.reward);done.push(r.done?1:0);
    epRet+=r.reward; o=r.obs;
    if(r.done){ rets.push(epRet);epRet=0; ends[r.info.end]=(ends[r.info.end]||0)+1; setEpCount(); o=env.reset(); }
  }
  const lastVal=agent.act(o).value;
  const {adv,ret}=agent.computeGAE(rew,val,done,lastVal);
  const nEp=rets.length||1;
  return {buf:{obs,act,logp,adv,ret}, meanRet:rets.reduce((a,b)=>a+b,0)/nEp, nEp, ends};
}
// deterministic eval at a fixed obstacle count
function evalAt(nObs, episodes=40){
  const e=new TruckEnv(mkrng(9999+nObs),{nObs,goalX:1100,maxT:700}); let goals=0;
  for(let ep=0;ep<episodes;ep++){ let o=e.reset(); for(let t=0;t<700;t++){ const {action}=agent.act(o,true); const r=e.step(action); o=r.obs; if(r.done){ if(r.info.end==='goal')goals++; break; } } }
  return goals/episodes*100;
}
console.log('NET='+NET+'  (train counts '+TRAIN_COUNTS.join(',')+')');
const t0=Date.now();
for(let it=0;it<=ITERS;it++){
  const frac=it/ITERS; agent.lr=LR0*(1-0.6*frac); agent.entCoef=Math.max(0.0015, ENT0*(1-1.1*frac));
  const {buf,meanRet,nEp,ends}=rollout(); const info=agent.update(buf);
  if(it%10===0||it===ITERS){ const g=((ends.goal||0)/nEp*100).toFixed(0);
    console.log('  it'+String(it).padStart(3)+'  ret '+meanRet.toFixed(2).padStart(6)+'  goal '+g.padStart(3)+'%  std '+info.std.toFixed(2)+'  ['+((Date.now()-t0)/1000).toFixed(0)+'s]'); }
}
const COUNTS_EVAL=[2,3,4,5,6]; const ev={}; for(const c of COUNTS_EVAL) ev[c]=evalAt(c,50);
console.log('  EVAL goal% by obstacle count (trained on '+TRAIN_COUNTS.join(',')+'):');
console.log('    '+COUNTS_EVAL.map(c=>'n'+c+'='+ev[c].toFixed(0)+'%').join('  '));
const base=ev[TRAIN_COUNTS[TRAIN_COUNTS.length-1]]||ev[2]||1;
console.log('    retention vs n'+ (TRAIN_COUNTS[TRAIN_COUNTS.length-1])+': '+COUNTS_EVAL.map(c=>'n'+c+'='+(base>0?(ev[c]/base*100).toFixed(0):'--')+'%').join('  '));
console.log('RESULT '+NET+' '+COUNTS_EVAL.map(c=>ev[c].toFixed(0)).join(' '));
