'use strict';
const { PPO } = require('./ppo.js');
function mkrng(s){ return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }
const rng=mkrng(7);

// 2D point-mass: reach a random goal. obs=[px,py,vx,vy,gx-px,gy-py]
function makeEnv(){
  let px,py,vx,vy,gx,gy,step;
  function reset(){ px=0;py=0;vx=0;vy=0; gx=(rng()*2-1)*0.8; gy=(rng()*2-1)*0.8; step=0; return obs(); }
  function obs(){ return new Float64Array([px,py,vx,vy,gx-px,gy-py]); }
  function stepEnv(a){
    const ax=Math.max(-1,Math.min(1,a[0])), ay=Math.max(-1,Math.min(1,a[1]));
    vx=(vx+ax*0.06)*0.92; vy=(vy+ay*0.06)*0.92; px+=vx; py+=vy;
    step++;
    const dist=Math.hypot(gx-px,gy-py);
    let r=-dist*0.05; const reached=dist<0.08; if(reached)r+=1.0;
    const done = step>=60;
    return { obs:obs(), reward:r, done };
  }
  return { reset, step:stepEnv };
}

const env=makeEnv();
const agent=new PPO(6,2,{rng, hidden:[64,64], lr:3e-4, initStd:0.7, entCoef:0.003, epochs:10, mb:64});
const T=2048;
function rollout(){
  const obs=[],act=[],logp=[],rew=[],val=[],done=[];
  let o=env.reset(), epRet=0; const rets=[];
  for(let t=0;t<T;t++){
    const {action,logProb,value}=agent.act(o);
    const r=env.step(action);
    obs.push(o); act.push(action); logp.push(logProb); val.push(value); rew.push(r.reward); done.push(r.done?1:0);
    epRet+=r.reward; o=r.obs;
    if(r.done){ rets.push(epRet); epRet=0; o=env.reset(); }
  }
  const lastVal=agent.act(o).value;
  const {adv,ret}=agent.computeGAE(rew,val,done,lastVal);
  return { buf:{obs,act,logp,adv,ret}, meanRet: rets.length?rets.reduce((a,b)=>a+b,0)/rets.length:epRet };
}
console.log('iter  meanEpRet   std    vLoss');
for(let it=0;it<60;it++){
  const {buf,meanRet}=rollout();
  const info=agent.update(buf);
  if(it%6===0||it===59) console.log(String(it).padStart(3)+'   '+meanRet.toFixed(3).padStart(8)+'   '+info.std.toFixed(2)+'   '+info.vLoss.toFixed(3));
}
