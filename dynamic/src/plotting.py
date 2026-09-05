from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

COLORS={'Single':'#777777','Pair':'#2679b8','Full':'#db6b22','DINOv3':'#298c52'}

def read_csv(path):
    with open(path) as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['layer','unit_id']:
            if k in r:r[k]=int(r[k])
        for k in ['median','value']:
            if k in r:r[k]=float(r[k])
    return rows


def curve(ax,rows,metric,rep,model='vggt_omega',regimes=('Single','Pair','Full'),phases=('pre','post')):
    for regime in regimes:
        for phase in phases:
            subset=[r for r in rows if r['metric']==metric and r['representation']==rep and r['model']==model and r['regime']==regime and r['phase']==phase]
            if not subset:continue
            by=defaultdict(list)
            for r in subset:by[r['group']].append(r)
            color=COLORS['DINOv3' if model=='dinov3' else regime];style='--' if phase=='pre' else '-'
            all_y=[];x=None
            for rr in by.values():
                rr=sorted(rr,key=lambda r:r['layer']);x=[r['layer'] for r in rr];y=[r['median'] for r in rr];all_y.append(y)
                ax.plot(x,y,style,color=color,alpha=.16,lw=.9)
            label='DINOv3' if model=='dinov3' else f'{regime} {phase}'
            ax.plot(x,np.nanmedian(all_y,axis=0),style,color=color,lw=1.9,marker='o',ms=3,label=label,markerfacecolor='white' if phase=='pre' else color)
    ax.set_xlabel('Block index (zero-based)');ax.set_title(metric);ax.grid(alpha=.18)


def save(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def build_figures(cfg):
    out=Path(cfg['output_root']);folder=out/'figures';folder.mkdir(exist_ok=True)
    rows=read_csv(out/'metrics/group_metrics.csv');da=json.loads((out/'audit/dataset_audit.json').read_text());gate=json.loads((out/'audit/extraction_full_validation.json').read_text())
    labels={g:f'G{i+1}' for i,g in enumerate(sorted(da['core']))}
    fig,axs=plt.subplots(2,3,figsize=(14,8));ax=axs.flat
    ax[0].bar(list(labels.values()),[da['core'][g]['count'] for g in labels]);ax[0].set_title('Common physical core points / group');ax[0].axhline(16,ls='--',color='gray')
    ds=da['displacements']
    for gi,g in enumerate(labels):
        rr=[r for r in ds if r['group']==g]
        for level in [1,2]:
            vals=[r['median_px'] for r in rr if max(abs(r['e']),abs(r['o']))==level]
            ax[1].scatter([level]*len(vals),vals,label=labels[g] if level==1 else None,alpha=.6,color=f'C{gi}')
    ax[1].set_xticks([1,2],['|level|=1','|level|=2']);ax[1].set_ylabel('Endpoint displacement (px)');ax[1].set_title('Pure ego / object, both signs');ax[1].legend()
    for level in [1,2]:
        vals=[r['median_patch_ratio'] for r in ds if max(abs(r['e']),abs(r['o']))==level]
        ax[2].scatter([level]*len(vals),vals,alpha=.7)
    ax[2].set_xticks([1,2]);ax[2].set_title('Displacement / 16 px patch');ax[2].set_xlabel('|motion level|')
    for g in labels:
        dist=[r['distance_px'] for r in da['boundary_distances'] if r['group']==g]
        ax[3].hist(dist,bins=25,histtype='step',label=labels[g],density=True)
    ax[3].axvline(8,color='k',ls='--');ax[3].set_xlabel('Mask boundary distance (model px)');ax[3].set_title('All conditions, retained core points');ax[3].legend()
    ax[4].axis('off');ax[4].text(.05,.8,f'Register-only patch max |pre-post|\n{gate["register_only_patch_max_error"]:.1e}\n\nChecked: {gate["shards_checked"]} shards\nNonfinite: {gate["nonfinite"]}\n\nSame-r track error: {da["live_max_same_r_final_uv_error_px"]:.2e} px',va='top',fontsize=12)
    from .dataset import PilotDataset
    d=PilotDataset(cfg);d.core=da['core'];g=sorted(d.groups)[0];s=next(s for s in d.groups[g] if s['is_static']);uv=d.point_uv(s)
    ax[5].imshow(Image.open(d.root/d.frames_for(s)[-1]['rgb']),extent=(0,512,512,0));ax[5].scatter(uv[:,0],uv[:,1],s=8,c='yellow');ax[5].set_title('G1 static: retained physical core points');ax[5].set_xlim(0,512);ax[5].set_ylim(512,0)
    save(fig,folder,'figure1_sanity')
    for model in ['vggt_omega','dinov3']:
        fig,axs=plt.subplots(2,3,figsize=(14,8))
        for a,metric in zip(axs.flat,['TC_e','TC_o','S_e','S_o','cos_signed','cos_abs']):
            curve(a,rows,metric,'patch',model=model,regimes=('Single',) if model=='dinov3' else ('Single','Pair','Full'),phases=('norm',) if model=='dinov3' else ('pre','post'))
        for a,metric in zip(axs.flat,['TC_e','TC_o','S_e','S_o','cos_signed','cos_abs']):
            if metric in ['S_e','S_o']:a.set_yscale('log');a.set_ylabel('Feature norm / metre (log scale)')
        axs[0,0].legend(fontsize=8,handlelength=4);fig.suptitle(f'Figure 2: {model} tracked-point tangent geometry; faint = individual groups',y=1.01)
        save(fig,folder,'figure2_tangents_'+model)
    fig,ax=plt.subplots(figsize=(9,5))
    curve(ax,rows,'D_cause','patch');curve(ax,rows,'D_cause','patch',model='dinov3',regimes=('Single',),phases=('norm',))
    ax.set_ylabel('Normalized same-r distance (20 pairs / point)');ax.set_title('Figure 3: same relative motion, different physical causes');ax.legend(ncol=2,fontsize=9,handlelength=4);ax.set_ylim(bottom=0)
    save(fig,folder,'figure3_causal_distance')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for i,regime in enumerate(['Pair','Full']):
        for j,rep in enumerate(['patch','camera','register']):
            curve(axs[i,j],rows,'D_cause',rep,regimes=(regime,));axs[i,j].set_title(f'{regime}: {rep}');axs[i,j].legend(fontsize=8)
    fig.suptitle('Figure 4: observational routing; register main curve = median over 16 registers',y=1.01);save(fig,folder,'figure4_routing')
    rr=read_csv(out/'metrics/register_metrics.csv');fig,axs=plt.subplots(2,2,figsize=(12,9))
    for i,regime in enumerate(['Pair','Full']):
        for j,phase in enumerate(['pre','post']):
            arr=np.empty((16,len(cfg['vggt_omega']['layers'])))
            for u in range(16):
                for li,l in enumerate(cfg['vggt_omega']['layers']):
                    arr[u,li]=np.median([r['value'] for r in rr if r['regime']==regime and r['phase']==phase and r['layer']==l and r['unit_id']==u and r['metric']=='D_cause'])
            im=axs[i,j].imshow(arr,aspect='auto',vmin=0,vmax=2,cmap='viridis');axs[i,j].set_xticks(range(len(cfg['vggt_omega']['layers'])),cfg['vggt_omega']['layers']);axs[i,j].set_title(f'{regime} {phase}');axs[i,j].set_xlabel('Layer');axs[i,j].set_ylabel('Register ID')
    fig.subplots_adjust(right=.84,wspace=.3,hspace=.3)
    cax=fig.add_axes([.88,.2,.02,.6]);fig.colorbar(im,cax=cax,label='D_cause (median over groups)')
    # Avoid tight_layout moving colorbar into panels.
    fig.savefig(folder/'figure4_register_heatmap.png',dpi=180,bbox_inches='tight');fig.savefig(folder/'figure4_register_heatmap.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    curve(axs[0,0],rows,'D_cause','pool');curve(axs[0,0],rows,'D_cause','pool',model='dinov3',regimes=('Single',),phases=('norm',));axs[0,0].set_title('Clean interior entity pooling');axs[0,0].legend(fontsize=7)
    curve(axs[0,1],rows,'C_norm','patch',phases=('post',));axs[0,1].set_title('Patch compensated response');axs[0,1].legend(fontsize=8)
    curve(axs[0,2],rows,'C_norm','pool',phases=('post',));axs[0,2].set_title('Pool compensated response')
    for ax,metric in zip(axs[1],['D_cause','C_norm','S_e']):
        for ri,regime in enumerate(['Single','Pair','Full']):
            rr=[r for r in rows if r['representation']=='dense' and r['regime']==regime and r['metric']==metric]
            vals=[r['median'] for r in sorted(rr,key=lambda r:r['group'])]
            ax.scatter(np.full(len(vals),ri),vals,color=COLORS[regime],alpha=.4);ax.scatter([ri],[np.median(vals)],marker='_',s=250,color=COLORS[regime])
            if metric=='S_e':
                vo=[r['median'] for r in rows if r['representation']=='dense' and r['regime']==regime and r['metric']=='S_o'];ax.scatter(np.full(len(vo),ri+.15),vo,marker='x',color=COLORS[regime],alpha=.5)
        ax.set_xticks([0,1,2],['Single','Pair','Full']);ax.set_title('Dense fused: '+('S_e circles / S_o crosses' if metric=='S_e' else metric));ax.grid(alpha=.15)
    save(fig,folder,'figure5_robustness')
    return sorted(p.name for p in folder.glob('*.png'))
