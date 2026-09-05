"""All input conditions are matched by group and canonical point ID.
Point metrics first; groups are the reporting unit. No point-as-independent tests.
"""
import itertools
import numpy as np
EPS=1e-8


def norm(x): return np.linalg.norm(x,axis=-1)

def cosine(x,y):
    den=norm(x)*norm(y)
    return np.divide(np.sum(x*y,axis=-1),den,out=np.full_like(den,np.nan),where=den>EPS)


def compute_metrics(features,delta):
    z={k:np.asarray(v,dtype=np.float32) for k,v in features.items()}
    dz={k:v-z[0,0] for k,v in z.items()}
    ve=(dz[1,0]-dz[-1,0])/(2*delta); vo=(dz[0,1]-dz[0,-1])/(2*delta)
    ve2=(dz[2,0]-dz[-2,0])/(4*delta); vo2=(dz[0,2]-dz[0,-2])/(4*delta)
    signed=cosine(ve,vo)
    m={'TC_e':cosine(ve,ve2),'TC_o':cosine(vo,vo2),'scale_e':norm(ve)/(norm(ve2)+EPS),'scale_o':norm(vo)/(norm(vo2)+EPS),'S_e':norm(ve),'S_o':norm(vo),'cos_signed':signed,'cos_abs':np.abs(signed)}
    pairs=[]
    for a,b in itertools.combinations(sorted(z),2):
        r=a[1]-a[0]
        if r!=0 and r==b[1]-b[0]:
            na,nb=norm(dz[a]),norm(dz[b]);distance=norm(dz[a]-dz[b])/(.5*(na+nb)+EPS)
            pairs.append({'a':a,'b':b,'r':r,'D_cause':distance,'cause_cos':cosine(dz[a],dz[b]),'delta_norm_a':na,'delta_norm_b':nb})
    assert len(pairs)==20
    m['D_cause']=np.mean([p['D_cause'] for p in pairs],axis=0)
    cc=np.array([p['cause_cos'] for p in pairs]);count=np.isfinite(cc).sum(0)
    m['cause_cos']=np.divide(np.nansum(cc,axis=0),count,out=np.full(cc.shape[1],np.nan,dtype=np.float32),where=count>0)
    cr=[];cn=[]
    for e in [-2,-1,1,2]:
        response=norm(dz[e,e]);ratio=response/(.5*(norm(dz[e,0])+norm(dz[0,e]))+EPS)
        m[f'C_resp_{e}']=response;m[f'C_norm_{e}']=ratio;cr.append(response);cn.append(ratio)
    m['C_resp']=np.mean(cr,axis=0);m['C_norm']=np.mean(cn,axis=0)
    return m,pairs


def summary(v):
    v=np.asarray(v,dtype=float);valid=v[np.isfinite(v)]
    if not len(valid):return {'median':float('nan'),'q25':float('nan'),'q75':float('nan'),'mean':float('nan'),'n':0,'undefined':len(v)}
    return {'median':float(np.median(valid)),'q25':float(np.quantile(valid,.25)),'q75':float(np.quantile(valid,.75)),'mean':float(np.mean(valid)),'n':len(valid),'undefined':len(v)-len(valid)}
