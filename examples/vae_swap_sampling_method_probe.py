"""CPU oracle-flow comparisons with analytic endpoint quantiles and counted NFE."""
import json, math, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
folder=Path('local/verification/vae_swap_schedule_method');folder.mkdir(parents=True,exist_ok=True)
torch.manual_seed(721)
# Every arm starts with identical, BF16-representable noise.
initial=torch.randn(256,dtype=torch.float64).bfloat16().double()
means=torch.tensor([-1.,1.],dtype=torch.float64)
def cdf(x,std):
    return torch.special.ndtr((x[:,None]-means)/std).mean(1)
def quantile_endpoint(x,std):
    p=torch.special.ndtr(x);lo=torch.full_like(x,-1-9*std);hi=torch.full_like(x,1+9*std)
    for _ in range(90):
        mid=(lo+hi)/2;left=cdf(mid,std)<p
        lo=torch.where(left,mid,lo);hi=torch.where(left,hi,mid)
    return (lo+hi)/2
def field(x,t,std,bias=False):
    a=1-t;variance=a*a*std*std+t*t;offset=x[:,None]-a*means
    weights=torch.softmax(-offset.square()/(2*variance),dim=1)
    v=(((t-a*std*std)/variance)*offset-means)*weights
    result=v.sum(1)
    if bias:result=result+.25*math.exp(-((t-.12)/.08)**2)
    return result
def grid(label,n):
    u=torch.linspace(1,0,n+1,dtype=torch.float64)
    if label.startswith('shift'):
        factor={'shift3':3.,'shift3_tail02':3.,'shift3_tailcap02':3.,'shift25_legacy':25.27966,'shift3.776_current':3.77632}[label]
        s=factor*u/(1+(factor-1)*u)
        if label=='shift3_tail02' or (label=='shift3_tailcap02' and s[-2]>.02):
            s[:-1]=1-(1-s[:-1])/(1-s[-2])*(1-.02)
        return s
    if label=='uniform':return u
    if label=='power2':return u.square()
    if label=='tail35':
        tail=max(1,round(n*.35));head=n-tail
        return torch.cat((torch.linspace(1,.2,head+1,dtype=torch.float64),
                          torch.linspace(.2,0,tail+1,dtype=torch.float64)[1:]))
    raise ValueError(label)
def coefficient_mesh(std,n,order):
    # Independent pilot points, all oracle evaluations included in reported cost.
    nodes=torch.linspace(1,0,25,dtype=torch.float64);x=initial.clone();estimates=[];pilot_calls=0
    for s,e in zip(nodes[:-1],nodes[1:]):
        t=s.item();h=(e-s).item();v=field(x,t,std)
        full=x+h*v;half=x+(h/2)*v
        vh=field(half,t+h/2,std);two=half+(h/2)*vh
        if order==1:
            error=full-two;x=two;pilot_calls+=2
        else:
            one_mid=x+h*vh
            quarter=x+(h/4)*v;vq=field(quarter,t+h/4,std)
            first=x+(h/2)*vq;vm=field(first,t+h/2,std)
            third=first+(h/4)*vm;vt=field(third,t+3*h/4,std)
            two_mid=first+(h/2)*vt;error=one_mid-two_mid;x=two_mid;pilot_calls+=5
        c=error.square().mean().sqrt().item()/abs(h)**(order+1)
        estimates.append(max(c,1e-8))
    densities=np.array(estimates)**(1/(order+1));densities=np.clip(densities,np.median(densities)*.2,np.median(densities)*5.)
    mass=np.concatenate(([0.],np.cumsum(densities)));mass/=mass[-1]
    s=np.interp(np.linspace(0,1,n+1),mass,nodes.numpy())
    return torch.tensor(s,dtype=torch.float64),pilot_calls
def integrate(sigmas,std,method='euler',state_dtype=torch.float32,bias=False):
    x=initial.to(state_dtype).double();calls=0
    for si,sj in zip(sigmas[:-1],sigmas[1:]):
        t=si.item();h=(sj-si).item();v=field(x,t,std,bias);calls+=1
        if method=='midpoint':
            midpoint=x+h*.5*v;v=field(midpoint,t+h*.5,std,bias);calls+=1
        x=(x+h*v).to(state_dtype).double()
    assert torch.isfinite(x).all()
    return x,calls
rows=[]
for std in (.03,.15,.5):
    ref=quantile_endpoint(initial,std)
    for budget in (20,40,80,160):
        for scheme in ('shift25_legacy','shift3.776_current','shift3','shift3_tail02','shift3_tailcap02','uniform','power2','tail35','coefficient_mesh'):
            for method in ('euler','midpoint'):
                n=budget if method=='euler' else budget//2
                sigmas,pilot=(coefficient_mesh(std,n,1 if method=='euler' else 2) if scheme=='coefficient_mesh' else (grid(scheme,n),0))
                assert sigmas[0]==1 and sigmas[-1]==0 and torch.all(sigmas[:-1]>sigmas[1:])
                out,nfe=integrate(sigmas,std,method);assert nfe==budget
                rows.append({'fixture_std':std,'scheme':scheme,'solver':method,'nfe':nfe,
                    'pilot_nfe':pilot,'state_dtype':'fp32','endpoint_rmse':(out-ref).square().mean().sqrt().item(),
                    'last_sigma':sigmas[-2].item(),'below_0.1':int((sigmas[:-1]<.1).sum())})
    print('ORACLE',std,[(a['scheme'],a['solver'],round(a['endpoint_rmse'],6)) for a in rows if a['fixture_std']==std and a['nfe']==40],flush=True)
# Isolate state rounding and learned-field bias from grid discretization.
controls=[]
std=.03;ref=quantile_endpoint(initial,std)
for budget in (40,80,160,640):
    for dtype in (torch.bfloat16,torch.float32,torch.float64):
        for biased in (False,True):
            out,nfe=integrate(grid('uniform',budget),std,state_dtype=dtype,bias=biased)
            controls.append({'nfe':nfe,'state_dtype':str(dtype),'biased_field':biased,
                'endpoint_rmse':(out-ref).square().mean().sqrt().item()})
# Fine detail in an invertible decoder can magnify a small latent integration error.
gain=32.;delta=torch.tensor([.001,.001],dtype=torch.float64)
weighted=torch.tensor([1.,gain])*delta
assert weighted[-1]==gain*delta[-1]
reference_checks={str(std):(cdf(quantile_endpoint(initial,std),std)-torch.special.ndtr(initial)).abs().max().item()
                  for std in (.03,.15,.5)}
assert max(reference_checks.values())<1e-12
assert not torch.cuda.is_initialized()
result={'date':'2026-10-12','gpu_used':False,'model_loaded':False,'long_training_performed':False,
    'fixture':'Exact independent-noise rectified flow for equal N(-1,std^2)/N(1,std^2) mixture; endpoint = F0^-1(Phi(initial)).',
    'particles':len(initial),'seed':721,'input_noise_dtype':'bf16','rows':rows,'controls':controls,
    'quantile_reference_cdf_residual':reference_checks,
    'decoder_gain_control':{'latent_delta':delta.tolist(),'pixel_delta':weighted.tolist(),
        'note':'Synthetic invertible linear decoder, not a measured Qwen decoder Jacobian.'}}
(folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,std in zip(axes,(.03,.15,.5)):
    for scheme,solver in [('shift3.776_current','euler'),('shift3_tailcap02','euler'),('uniform','euler'),
                         ('tail35','euler'),('coefficient_mesh','euler'),('uniform','midpoint')]:
        selected=[a for a in rows if a['fixture_std']==std and a['scheme']==scheme and a['solver']==solver]
        ax.loglog([a['nfe'] for a in selected],[a['endpoint_rmse'] for a in selected],marker='o',label=scheme+'/'+solver)
    ax.set(title=f'Analytic RF mixture, component std={std}',xlabel='NFE (pilot excluded)',ylabel='Endpoint RMSE')
    ax.grid(alpha=.25)
axes[-1].legend(fontsize=7)
fig.savefig(folder/'oracle_comparison.png',dpi=150);plt.close(fig)
print('METHOD_PROBE_COMPLETED',len(rows),len(controls),flush=True)
