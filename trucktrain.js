'use strict';
const { PPO } = require('./ppo.js');
const { TruckEnv, OBS_DIM, ACT_DIM } = require('./env.js');
function mkrng(s){ return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }
const rng=mkrng(11);
const NOBS = +(process.env.NOBS||0);
const ITERS = +(process.env.ITERS||60);
const env=new TruckEnv(rng,{nObs:NOBS, goalX:1100, maxT:700});
const LR0=3e-4, ENT0=0.005;
const agent=new PPO(OBS_DIM,ACT_DIM,{rng,hidden:[48,48],lr:LR0,initStd:0.7,entCoef:ENT0,gamma:0.99,lam:0.95,epochs:8,mb:128,targetKL:0.05});

const T=2048;
function rollout(){
  const obs=[],act=[],logp=[],rew=[],val=[],done=[];
  let o=env.reset(), epRet=0; const rets=[], ends={};
  for(let t=0;t<T;t++){
    const {action,logProb,value}=agent.act(o);
    const r=env.step(action);
    obs.push(o);act.push(action);logp.push(logProb);val.push(value);rew.push(r.reward);done.push(r.done?1:0);
    epRet+=r.reward; o=r.obs;
    if(r.done){ rets.push(epRet); epRet=0; ends[r.info.end]=(ends[r.info.end]||0)+1; o=env.reset(); }
  }
  const lastVal=agent.act(o).value;
  const {adv,ret}=agent.computeGAE(rew,val,done,lastVal);
  const nEp=rets.length||1;
  return { buf:{obs,act,logp,adv,ret}, meanRet:rets.reduce((a,b)=>a+b,0)/nEp, nEp, ends };
}
console.log('NOBS='+NOBS+'  iter  meanRet  goal%  crash%  eps  std');
const t0=Date.now();
for(let it=0;it<=ITERS;it++){
  const frac=it/ITERS;
  agent.lr = LR0 * (1 - 0.7*frac);
  agent.entCoef = ENT0 * Math.max(0, 1 - 1.3*frac);   // anneal exploration toward exploitation
  const {buf,meanRet,nEp,ends}=rollout();
  const info=agent.update(buf);
  if(it%6===0||it===ITERS){
    const goalP=((ends.goal||0)/nEp*100).toFixed(0);
    const crashP=(((ends.crash||0)+(ends.jackknife||0)+(ends.offlane||0))/nEp*100).toFixed(0);
    console.log('     '+String(it).padStart(3)+'  '+meanRet.toFixed(2).padStart(7)+'   '+goalP.padStart(3)+'   '+crashP.padStart(3)+'   '+String(nEp).padStart(3)+'  '+info.std.toFixed(2)+'  ['+((Date.now()-t0)/1000).toFixed(0)+'s]');
  }
}
