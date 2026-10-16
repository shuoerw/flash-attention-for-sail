"""Dense K3 FA3 prefill kernel ablations on identical BF16 inputs.

B is requests per DP replica; H is the local head count on one TP rank.
The default V view reproduces the stride of K3's expanded KV projection.
"""
import argparse, ctypes, gc, itertools, json, math, random, statistics, time
from pathlib import Path
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--extra-libraries', type=Path, nargs='*', default=[])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--suite', choices=['smoke', 'signal', 'safety', 'boundaries', 'contexts'], default='smoke')
p.add_argument('--extra-mode', type=int, default=8)
p.add_argument('--exact-extra', action='store_true', help='Require extra libraries to match the normal dispatcher bit-for-bit.')
p.add_argument('--samples', type=int, default=31)
p.add_argument('--modes', type=int, nargs='+', choices=[0, 1, 8], default=[0, 1, 8])
p.add_argument('--case', help='B,Q,K,H,causal; overrides suite')
p.add_argument('--case-index', type=int, action='append', help='Repeat selected zero-based cases from a suite.')
p.add_argument('--installed', action='store_true')
a=p.parse_args()
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(1731)
def load_library(path):
    dll=ctypes.CDLL(str(path.resolve()),mode=ctypes.RTLD_LOCAL)
    dll.fa_k3_run.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.c_void_p]
    dll.fa_k3_run.restype=ctypes.c_int
    return dll
lib=load_library(a.library)
extra={p.parent.name:load_library(p) for p in a.extra_libraries}

def case(b,q,k,h=12,causal=True,ragged=False,strided=True):
    qs=[q]*b; ks=[k]*b
    if ragged:
        qs=[max(1,q*(i+1)//b-(i%3)) for i in range(b)]
        ks=[max(qs[i],k*(i+1)//b-(i%5)) for i in range(b)]
        qs[-1]=q; ks[-1]=k
    return dict(qs=qs,ks=ks,h=h,causal=causal,strided=strided)

if a.case:
    b,q,k,h,c=map(int,a.case.split(',')); cases=[case(b,q,k,h,bool(c))]
elif a.suite=='smoke':
    cases=[case(1,q,k,12,c) for q,k,c in [(17,17,True),(65,129,True),(129,257,False),(512,512,True),(4096,4096,True)]]
    cases += [case(4,257,1025,12,c,True) for c in [True,False]]
    cases += [dict(qs=[0,65,0,17],ks=[0,129,0,65],h=12,causal=True,strided=True)]
elif a.suite=='signal':
    cases=[case(1,q,k,12,c) for q,k,c in [(512,512,True),(2048,2048,True),(4096,4096,True),(16384,16384,True),(512,8192,True),(2048,32768,True),(512,32768,False),(2048,131072,False)]]
    cases += [case(4,512,512),case(4,512,8192,12,True,True)]
elif a.suite=='boundaries':
    cases=[case(1,512,k,12,c) for k in [4095,4096,4097] for c in [True,False]]
    cases += [case(4,512,k,12,c) for k in [4095,4096,4097] for c in [True,False]]
    cases += [case(4,513,k,12,c,True) for k in [6553,6554,8193] for c in [True,False]]
    # Many short requests with one long outlier.
    cases += [dict(qs=[64]*31+[512],ks=[128]*31+[k],h=12,causal=c,strided=True)
              for k in [65536,131072] for c in [True,False]]
    cases += [dict(qs=[0,512,0,129],ks=[0,8192,0,8192],h=12,causal=c,strided=True)
              for c in [True,False]]
    cases += [case(b,q,8192,12,c,True) for b,q,c in [(16,129,True),(64,65,False),(256,65,True)]]
elif a.suite=='contexts':
    contexts=[4096 * 2**i for i in range(9)]
    cases=[case(1,4096,k) for k in contexts]
    cases += [case(1,min(k,65536),k) for k in contexts[1:]]
    cases += [case(1,4096,k,causal=False) for k in contexts]
    cases += [dict(qs=[4096,8192,32768,20480],ks=ks,h=12,causal=True,strided=True)
              for ks in [[4096,8192,98304,151552], [4096,8192,524288,1048576]]]
    cases += [case(1,65537,1048577), case(257,256,4096)]
else:
    cases=[case(1,q,k,12,c) for q,k,c in [(64,64,True),(127,127,True),(704,704,True),(705,705,True),(1024,1024,True),(4097,4097,True),(32768,32768,True),(65536,65536,True),(512,131072,True),(512,1048576,True),(512,131072,False),(512,1048576,False)]]
    cases += [case(b,q,k,12,c,True) for b,q,k,c in [(4,513,8193,True),(16,129,16385,True),(32,65,8193,False),(64,33,1025,True),(128,17,513,False)]]
    cases += [case(1,2048,2048,h) for h in [6,24,96]]
    cases += [case(1,4096,4096,12,True,False,False)]

if a.case_index is not None:
    cases=[cases[i] for i in a.case_index]

results=[]
report=dict(library=str(a.library),suite=a.suite,extra_libraries=[str(p) for p in a.extra_libraries],samples=a.samples,device=torch.cuda.get_device_name(0),results=results)
a.output.parent.mkdir(parents=True,exist_ok=True)

def guard(shape,dtype,guards):
    n=math.prod(shape); t=torch.empty(n+128,device='cuda',dtype=dtype)
    t[:64].fill_(17); t[-64:].fill_(17); guards.extend([t[:64],t[-64:]])
    return t[64:-64].view(shape)

def run_case(spec):
    qs,ks,h=spec['qs'],spec['ks'],spec['h']; tq,tk=sum(qs),sum(ks)
    q=torch.randn(tq,h,192,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(tk,h,192,device='cuda',dtype=torch.bfloat16)
    vv=torch.randn(tk,h,256 if spec['strided'] else 128,device='cuda',dtype=torch.bfloat16)
    v=vv[...,128:] if spec['strided'] else vv
    cuq=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
    cuk=torch.tensor([0]+list(itertools.accumulate(ks)),device='cuda',dtype=torch.int32)
    guards=[]; o=guard((tq,h,128),q.dtype,guards); lse=guard((h,tq),torch.float32,guards); sem=guard((32,),torch.int32,guards)
    ptr=(ctypes.c_void_p*8)(*[x.data_ptr() for x in [q,k,v,o,lse,sem,cuq,cuk]])
    calls={}; configs={};tiles={}
    for label,mode,dll in [('base',0,lib),('aa',0,lib)]+[(f'm{m}',m,lib) for m in a.modes if m!=0]+[(name,a.extra_mode,dll) for name,dll in extra.items()]:
        cfg=(ctypes.c_int*14)(mode,len(qs),tq,tk,max(qs),max(ks),h,int(spec['causal']),q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1))
        configs[label]=cfg
        def call(cfg=cfg,label=label,dll=dll):
            n=dll.fa_k3_run(ptr,cfg,torch.cuda.current_stream().cuda_stream)
            if n<0: raise RuntimeError(f'{label}: CUDA error {-n}')
            tiles[label]=n
            return o,lse
        calls[label]=call
    if a.installed:
        from flash_attn_interface import flash_attn_varlen_func
        def installed():
            r=flash_attn_varlen_func(q,k,v,cuq,cuk,max(qs),max(ks),causal=spec['causal'],return_attn_probs=True)
            return r[0],r[1]
        calls['installed']=installed
    calls['base'](); torch.cuda.synchronize(); baseline=o.clone(); baseline_lse=lse.clone()
    checks={}
    for label,call in calls.items():
        actual,actual_lse=call(); torch.cuda.synchronize()
        if label!='installed':
            assert torch.equal(actual,baseline),f'{label}: output differs from default: max_abs={(actual.float()-baseline.float()).abs().max().item()}, count={(actual!=baseline).sum().item()}'
            assert torch.equal(actual_lse,baseline_lse),f'{label}: LSE differs from default'
        else:
            torch.testing.assert_close(actual,baseline,atol=0.015,rtol=0.015)
            torch.testing.assert_close(actual_lse,baseline_lse,atol=0.002,rtol=0.002)
        if label == 'm8':
            dispatch_output=actual.clone(); dispatch_lse=actual_lse.clone()
        if a.exact_extra and label in extra:
            assert torch.equal(actual,dispatch_output) and torch.equal(actual_lse,dispatch_lse), 'Refactor differs from validated library'
        checks[label]='passed'
    candidate = candidate_lse = None
    for label in ['m1', 'm8']:
        if label in calls:
            x,xl=calls[label](); candidate=x.clone(); candidate_lse=xl.clone()
            torch.cuda.synchronize()
            if label == 'm8' and tiles[label] < 1000:
                assert torch.equal(candidate,baseline) and torch.equal(candidate_lse,baseline_lse), 'Fallback differs from original'
            break
    # Full FP32 reference on small inputs, sampled rows and all heads otherwise.
    errors=[]; offset_q=offset_k=0
    for qn,kn in zip(qs,ks):
        if not qn: offset_k+=kn; continue
        rows=list(range(qn)) if qn<=257 and kn<=2048 else sorted(set([0,min(1,qn-1),qn//4,qn//2,qn-1]))
        ids=torch.tensor(rows,device='cuda')
        head_step=1 if kn>65536 else h
        for hs in range(0,h,head_step):
            he=min(h,hs+head_step)
            qq=q[offset_q+ids,hs:he].float().transpose(0,1)
            kk=k[offset_k:offset_k+kn,hs:he].float().transpose(0,1)
            scores=torch.matmul(qq,kk.transpose(-1,-2))/math.sqrt(192)
            if spec['causal']:
                mask=torch.arange(kn,device='cuda')[None,:] > ids[:,None]+kn-qn
                scores.masked_fill_(mask[None],float('-inf'))
            ref_lse=torch.logsumexp(scores,dim=-1)
            prob=torch.softmax(scores,dim=-1).nan_to_num()
            expected=torch.matmul(prob,v[offset_k:offset_k+kn,hs:he].float().transpose(0,1)).transpose(0,1)
            actual=baseline[offset_q+ids,hs:he].float()
            torch.testing.assert_close(actual,expected,atol=0.02,rtol=0.02)
            if candidate is not None:
                torch.testing.assert_close(candidate[offset_q+ids,hs:he].float(),expected,atol=0.02,rtol=0.02)
                torch.testing.assert_close(candidate_lse[hs:he,offset_q+ids],ref_lse,atol=0.003,rtol=0.003)
            torch.testing.assert_close(baseline_lse[hs:he,offset_q+ids],ref_lse,atol=0.003,rtol=0.003)
            errors.append(float((actual-expected).abs().max()))
            del qq,kk,scores,prob,expected,actual,ref_lse
        offset_q+=qn; offset_k+=kn
    for g in guards: assert bool((g==17).all()),'guard modified'
    graphs={}
    for label,call in calls.items():
        for _ in range(3): call()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): call()
        graphs[label]=g
    evict=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    samples={label:[] for label in graphs}
    rng=random.Random(51)
    for _ in range(a.samples):
        labels=list(graphs);rng.shuffle(labels)
        for label in labels:
            evict.zero_()
            st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
            st.record();graphs[label].replay();en.record();en.synchronize()
            samples[label].append(st.elapsed_time(en))
    med={label:statistics.median(vals) for label,vals in samples.items()}
    row=dict(**spec,tiles=tiles,correctness=checks,fp32_max_abs=max(errors),candidate_fp32_checked=candidate is not None,median_ms=med,change_pct={label:(ms/med['base']-1)*100 for label,ms in med.items()},samples_ms=samples)
    print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
    return row

try:
    for spec in cases:
        results.append(run_case(spec)); a.output.write_text(json.dumps(report,indent=2)+'\n')
        gc.collect();torch.cuda.empty_cache()
except Exception as ex:
    report['failure']=repr(ex);report['failed_case']=spec
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    raise
