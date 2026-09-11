"""E2 extensions around the numerically identical E1 homogeneous-family metrics."""
import itertools
import numpy as np
from .metrics import EPS,compute_metrics,cosine,norm


def family_metrics(features,delta):
    legacy,pairs=compute_metrics(features,delta)
    z={k:np.asarray(v,dtype=np.float32) for k,v in features.items()}
    ve=(z[1,0]-z[-1,0])/(2*delta);vo=(z[0,1]-z[0,-1])/(2*delta)
    vr=(vo-ve)/2;vc=ve+vo;den=norm(ve)+norm(vo)+EPS
    extra={'S_r':norm(vr),'S_c':norm(vc),'M_c':norm(vc)/den,'M_r':norm(vo-ve)/den,
           'magnitude_log_ratio':np.abs(np.log((norm(ve)+EPS)/(norm(vo)+EPS)))}
    return {**legacy,'D_cause_legacy':legacy['D_cause'],**extra},decomposed_common_pairs(features)


def decomposed_common_pairs(features):
    z={k:np.asarray(v,dtype=np.float32) for k,v in features.items()};base=z[0,0];dz={k:v-base for k,v in z.items()};rows=[]
    for a,b in itertools.combinations(sorted(z),2):
        r=a[1]-a[0]
        if r==0 or r!=b[1]-b[0]:continue
        na,nb=norm(dz[a]),norm(dz[b]);distance=norm(dz[a]-dz[b])/(.5*(na+nb)+EPS)
        rows.append({'r':r,'c_a':(a[0]+a[1])/2,'c_b':(b[0]+b[1])/2,
            'delta_c':abs((b[0]+b[1]-a[0]-a[1])/2),'e_a':a[0],'o_a':a[1],'e_b':b[0],'o_b':b[1],
            'D_c_given_r':distance,'cause_cos':cosine(dz[a],dz[b]),'response_norm_a':na,'response_norm_b':nb})
    if len(rows)!=20:raise ValueError(f'Expected 20 nonzero matched-r pairs, found {len(rows)}')
    return rows


def response_vectors(features,delta):
    z={k:np.asarray(v,dtype=np.float32) for k,v in features.items()}
    return (z[1,0]-z[-1,0])/(2*delta),(z[0,1]-z[0,-1])/(2*delta)


def _basis(j,relative_tolerance):
    u,s,_=np.linalg.svd(j,full_matrices=False);threshold=relative_tolerance*(s[0] if len(s) else 0.)
    rank=int(np.sum(s>threshold));return u[:,:rank],s,rank


def xy_subspace(vex,vey,vox,voy,rank_relative_tolerance=1e-3,well_conditioned_rho=.05):
    je=np.stack([vex,vey],axis=-1);jo=np.stack([vox,voy],axis=-1)
    qe,se,re=_basis(je,rank_relative_tolerance);qo,so,ro=_basis(jo,rank_relative_tolerance)
    rhoe=float(se[1]/(se[0]+EPS));rhoo=float(so[1]/(so[0]+EPS));angles=[];overlap=float('nan')
    if re and ro:
        singular=np.linalg.svd(qe.T@qo,compute_uv=False);singular=np.clip(singular,0,1)
        angles=np.degrees(np.arccos(singular)).tolist();overlap=float(np.sum(singular**2)/min(re,ro))
    vrx=(vox-vex)/2;vry=(voy-vey)/2;vcx=vex+vox;vcy=vey+voy
    return {'sigma_e_1':float(se[0]),'sigma_e_2':float(se[1]),'sigma_o_1':float(so[0]),'sigma_o_2':float(so[1]),
        'rho_e':rhoe,'rho_o':rhoo,'rank_e':re,'rank_o':ro,'cos_ego_xy':float(cosine(vex,vey)),
        'cos_object_xy':float(cosine(vox,voy)),'principal_angles_deg':angles,'subspace_overlap':overlap,
        'well_conditioned_2d':bool(re==ro==2 and rhoe>=well_conditioned_rho and rhoo>=well_conditioned_rho),
        'relative_energy':float(np.linalg.norm(np.stack([vrx,vry],axis=-1))**2),
        'common_energy':float(np.linalg.norm(np.stack([vcx,vcy],axis=-1))**2)}


def scale_consistency(vectors_by_delta):
    rows=[]
    for a,b in itertools.combinations(sorted(vectors_by_delta),2):
        va,vb=vectors_by_delta[a],vectors_by_delta[b]
        rows.append({'delta_a':a,'delta_b':b,'cosine':cosine(va,vb),
                     'norm_ratio':norm(va)/(norm(vb)+EPS)})
    return rows
