'use strict';
const { AttnNet } = require('./attn.js');
function mkrng(s){ return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }
const rng=mkrng(3);

const egoDim=6, K=5, agentDim=5, outDim=3;
const net=new AttnNet({rng,egoDim,K,agentDim,dh:8,d:8,dv:8,head:[16,16],outDim,outScale:1.0});

// structured obs: ego(6) + K agents(5); mark some present, some absent (5th elem=present flag)
const obs=new Float64Array(egoDim+K*agentDim);
for(let i=0;i<egoDim;i++)obs[i]=rng()*2-1;
const present=[1,1,0,1,0];
for(let i=0;i<K;i++){ const b=egoDim+i*agentDim; for(let l=0;l<agentDim-1;l++)obs[b+l]=rng()*2-1; obs[b+agentDim-1]=present[i]; }
const target=new Float64Array(outDim); for(let i=0;i<outDim;i++)target[i]=rng()*2-1;

function loss(){ const out=net.forward(obs); let L=0; for(let i=0;i<outDim;i++)L+=0.5*(out[i]-target[i])**2; return L; }
// analytic
net.zeroGrad();
const out=net.forward(obs); const gOut=new Float64Array(outDim); for(let i=0;i<outDim;i++)gOut[i]=out[i]-target[i];
net.backward(gOut);

const eps=1e-5; let maxRel=0, worst='';
function check(name, W, gW){
  for(let i=0;i<W.length;i++){
    const o=W[i];
    W[i]=o+eps; const Lp=loss(); W[i]=o-eps; const Lm=loss(); W[i]=o;
    const num=(Lp-Lm)/(2*eps), ana=gW[i];
    const rel=Math.abs(num-ana)/(Math.max(1e-7,Math.abs(num)+Math.abs(ana)));
    if(rel>maxRel){ maxRel=rel; worst=name+'['+i+'] num='+num.toFixed(7)+' ana='+ana.toFixed(7); }
  }
}
check('Wa',net.Wa,net.gWa); check('ba',net.ba,net.gba);
check('Wq',net.Wq,net.gWq); check('Wk',net.Wk,net.gWk); check('Wv',net.Wv,net.gWv);
for(let l=0;l<net.head.L;l++){ check('head.W'+l,net.head.W[l],net.head.gW[l]); check('head.B'+l,net.head.b[l],net.head.gB[l]); }

console.log('AttnNet max relative grad error: '+maxRel.toExponential(3));
console.log('worst: '+worst);
console.log(maxRel<1e-5 ? 'PASS (attention backprop correct)' : 'FAIL');
