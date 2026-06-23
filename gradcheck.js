'use strict';
const { MLP } = require('./nn.js');
function rng(){ let s=12345; return ()=> (s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff; }
const r=rng();
const net=new MLP([4,6,5,3], r);
const x=new Float64Array([0.3,-0.7,0.5,0.1]);
const target=new Float64Array([0.2,-0.4,0.9]);
function lossOf(){ const o=net.forward(x); let L=0; for(let i=0;i<o.length;i++)L+=0.5*(o[i]-target[i])**2; return L; }
// analytic
net.zeroGrad();
const o=net.forward(x); const g=new Float64Array(o.length); for(let i=0;i<o.length;i++)g[i]=o[i]-target[i];
net.backward(g);
// numerical, compare for every W and b
let maxRel=0, eps=1e-6;
for(let l=0;l<net.L;l++){
  for(let i=0;i<net.W[l].length;i++){
    const old=net.W[l][i];
    net.W[l][i]=old+eps; const Lp=lossOf();
    net.W[l][i]=old-eps; const Lm=lossOf();
    net.W[l][i]=old;
    const num=(Lp-Lm)/(2*eps), ana=net.gW[l][i];
    const rel=Math.abs(num-ana)/(Math.abs(num)+Math.abs(ana)+1e-12);
    if(rel>maxRel)maxRel=rel;
  }
  for(let j=0;j<net.b[l].length;j++){
    const old=net.b[l][j];
    net.b[l][j]=old+eps; const Lp=lossOf();
    net.b[l][j]=old-eps; const Lm=lossOf();
    net.b[l][j]=old;
    const num=(Lp-Lm)/(2*eps), ana=net.gB[l][j];
    const rel=Math.abs(num-ana)/(Math.abs(num)+Math.abs(ana)+1e-12);
    if(rel>maxRel)maxRel=rel;
  }
}
console.log('max relative grad error: '+maxRel.toExponential(3)+'  -> '+(maxRel<1e-5?'PASS (backprop correct)':'FAIL'));
