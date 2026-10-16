"""Profile one native K3 attention call after warmup."""
import argparse,ctypes,math
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--library',type=Path,required=True)
p.add_argument('--mode',type=int,default=0)
p.add_argument('--q',type=int,default=4096)
p.add_argument('--k',type=int,default=4096)
p.add_argument('--heads',type=int,default=12)
p.add_argument('--noncausal',action='store_true')
a=p.parse_args();torch.set_grad_enabled(False)
h=a.heads;q=torch.randn(a.q,h,192,device='cuda',dtype=torch.bfloat16)
k=torch.randn(a.k,h,192,device='cuda',dtype=torch.bfloat16)
vv=torch.randn(a.k,h,256,device='cuda',dtype=torch.bfloat16);v=vv[...,128:]
o=torch.empty(a.q,h,128,device='cuda',dtype=torch.bfloat16)
lse=torch.empty(h,a.q,device='cuda',dtype=torch.float32)
sem=torch.empty(32,device='cuda',dtype=torch.int32)
cuq=torch.tensor([0,a.q],device='cuda',dtype=torch.int32);cuk=torch.tensor([0,a.k],device='cuda',dtype=torch.int32)
ptr=(ctypes.c_void_p*8)(*[x.data_ptr() for x in [q,k,v,o,lse,sem,cuq,cuk]])
cfg=(ctypes.c_int*14)(a.mode,1,a.q,a.k,a.q,a.k,h,int(not a.noncausal),q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1))
lib=ctypes.CDLL(str(a.library.resolve()),mode=ctypes.RTLD_LOCAL)
lib.fa_k3_run.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.c_void_p]
lib.fa_k3_run.restype=ctypes.c_int
for _ in range(5): lib.fa_k3_run(ptr,cfg,torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize();torch.cuda.profiler.start()
assert lib.fa_k3_run(ptr,cfg,torch.cuda.current_stream().cuda_stream)>0
torch.cuda.synchronize();torch.cuda.profiler.stop()
