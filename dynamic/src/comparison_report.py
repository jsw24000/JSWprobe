"""Model-independent E1 report: same metrics/statistical units for any registry subset."""
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .plotting import plt,curve,save,read_csv,LABELS
from .model_registry import configured_models,regimes_for,MODELS
from .feature_io import write_json,digest


def figures(cfg,rows,da):
    out=Path(cfg['output_root']);folder=out/'figures';folder.mkdir(exist_ok=True)
    names=configured_models(cfg);groups=sorted(da['core'])
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    axes[0,0].bar(range(len(groups)),[da['core'][g]['count'] for g in groups]);axes[0,0].set_xticks(range(len(groups)),[f'G{i+1}' for i in range(len(groups))]);axes[0,0].set_title('Common physical points (all models, all conditions)')
    for i,g in enumerate(groups):
        for level in [1,2]:
            values=[r['median_px'] for r in da['displacements'] if r['group']==g and max(abs(r['e']),abs(r['o']))==level]
            axes[0,1].scatter([level]*len(values),values,color=f'C{i}',label=f'G{i+1}' if level==1 else None,alpha=.7)
    axes[0,1].set_xticks([1,2]);axes[0,1].set_xlabel('|level|');axes[0,1].set_ylabel('Displacement (unchanged source pixels)');axes[0,1].legend()
    for i,g in enumerate(groups):axes[1,0].hist([r['distance_px'] for r in da['boundary_distances'] if r['group']==g],bins=25,histtype='step',label=f'G{i+1}',density=True)
    axes[1,0].axvline(8,ls='--',color='k');axes[1,0].set_xlabel('Mask boundary distance (px)');axes[1,0].legend()
    axes[1,1].axis('off');gate=json.loads((out/'audit/extraction_full_validation.json').read_text())
    lines=[f'Checked shards: {gate["shards_checked"]}',f'NaN/Inf: {gate["nonfinite"]}',f'Omega register-only patch error: {gate["register_only_patch_max_error"]}', 'VGGT: all-global attention; equality test N/A']
    for name in names:
        ds=np.array([r['median_px'] for r in da['displacements']])/MODELS[name]['patch']
        lines.append(f'{LABELS[name]}: patch={MODELS[name]["patch"]}; motion/patch={ds.min():.2f}-{ds.max():.2f}')
    axes[1,1].text(0,1,'\n'.join(lines),va='top',fontsize=10)
    save(fig,folder,'figure1_sanity')
    for name in names:
        fig,axes=plt.subplots(2,3,figsize=(14,8));image=MODELS[name]['family']=='image'
        for ax,metric in zip(axes.flat,['TC_e','TC_o','S_e','S_o','cos_signed','cos_abs']):
            curve(ax,rows,metric,'patch',model=name,regimes=regimes_for(name),phases=('norm',) if image else ('pre','post'))
            if metric in ['S_e','S_o']:ax.set_yscale('log')
        axes[0,0].legend(fontsize=7,handlelength=3);fig.suptitle(f'{LABELS[name]} physical-point tangent geometry; faint = groups')
        save(fig,folder,'figure2_tangents_'+name)
    fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
    for ax,phase in zip(axes,['pre','post']):
        for name in names:
            curve(ax,rows,'D_cause','patch',model=name,regimes=regimes_for(name),phases=('norm',) if MODELS[name]['family']=='image' else (phase,))
        ax.set_title('Same-r causal distance: '+phase+' (DINO norm in both)');ax.set_ylim(bottom=0);ax.legend(fontsize=8,ncol=2,handlelength=3)
    save(fig,folder,'figure3_four_model_causal_distance')
    register_rows=read_csv(out/'metrics/register_metrics.csv')
    geometry=[n for n in names if MODELS[n]['family']=='geometry']
    for name in geometry:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for i,reg in enumerate(['Pair','Full']):
            for j,rep in enumerate(['patch','camera','register']):
                curve(axes[i,j],rows,'D_cause',rep,model=name,regimes=(reg,));axes[i,j].set_title(f'{LABELS[name]} {reg} {rep}');axes[i,j].legend(fontsize=7)
        save(fig,folder,'figure4_routing_'+name)
        fig,axes=plt.subplots(2,2,figsize=(12,8));values=defaultdict(list)
        for r in register_rows:
            if r['model']==name and r['metric']=='D_cause':values[r['regime'],r['phase'],r['unit_id'],r['layer']].append(r['value'])
        for i,reg in enumerate(['Pair','Full']):
            for j,phase in enumerate(['pre','post']):
                arr=[[np.median(values[reg,phase,u,l]) for l in cfg[name]['layers']] for u in range(MODELS[name]['registers'])]
                im=axes[i,j].imshow(arr,aspect='auto',vmin=0,vmax=2,cmap='viridis');axes[i,j].set_title(f'{LABELS[name]} {reg} {phase}');axes[i,j].set_xticks(range(len(cfg[name]['layers'])),cfg[name]['layers']);axes[i,j].set_ylabel('Register ID');axes[i,j].set_xlabel('Layer')
        fig.subplots_adjust(right=.84,wspace=.3,hspace=.4);fig.colorbar(im,cax=fig.add_axes([.88,.2,.02,.6]),label='D_cause, group median')
        for ext in ['png','pdf']:fig.savefig(folder/f'figure4_register_heatmap_{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for name in names:
        for ax,metric in zip(axes,['D_cause','C_norm']):curve(ax,rows,metric,'pool',model=name,regimes=regimes_for(name),phases=('norm',) if MODELS[name]['family']=='image' else ('post',))
    axes[0].legend(fontsize=7,ncol=2);fig.suptitle('Entity pooling robustness (not physical-point correspondence)');save(fig,folder,'figure5_pooling')
    if geometry:
        fig,axes=plt.subplots(len(geometry),4,figsize=(16,4*len(geometry)),squeeze=False)
        for i,name in enumerate(geometry):
            for j,metric in enumerate(['D_cause','C_norm','S_e','S_o']):
                for ri,reg in enumerate(regimes_for(name)):
                    vals=[r['median'] for r in rows if r['model']==name and r['representation']=='dense' and r['regime']==reg and r['metric']==metric]
                    axes[i,j].scatter([ri]*len(vals),vals,alpha=.35,color=f'C{ri}');axes[i,j].scatter([ri],[np.median(vals)],marker='_',s=180,color=f'C{ri}')
                axes[i,j].set_xticks([0,1,2],regimes_for(name));axes[i,j].set_title(f'{LABELS[name]} dense {metric}');axes[i,j].grid(alpha=.2)
        save(fig,folder,'figure5_dense')
    return sorted(p.name for p in folder.glob('*.png'))


def build_model_report(cfg):
    out=Path(cfg['output_root']);names=configured_models(cfg)
    da=json.loads((out/'audit/dataset_audit.json').read_text());ma=json.loads((out/'audit/model_audit.json').read_text());gate=json.loads((out/'audit/extraction_full_validation.json').read_text());fidelity=json.loads((out/'audit/forward_fidelity.json').read_text())
    assert gate['passed'] and fidelity['passed']
    rows=read_csv(out/'metrics/group_metrics.csv');lr=read_csv(out/'metrics/layer_metrics.csv');pngs=figures(cfg,rows,da)
    metrics=['TC_e','TC_o','cos_signed','cos_abs','S_e','S_o','D_cause','C_norm']
    endpoints=[]
    for name in names:
        for reg in regimes_for(name):
            for rep in (['patch','pool'] if MODELS[name]['family']=='image' else ['patch','pool','camera','register','dense']):
                phase='fused' if rep=='dense' else ('norm' if MODELS[name]['family']=='image' else 'post')
                vals={r['metric']:r['median'] for r in lr if r['model']==name and r['regime']==reg and r['representation']==rep and r['layer']==max(cfg[name]['layers']) and r['phase']==phase}
                endpoints.append({'model':name,'regime':reg,'representation':rep,'phase':phase,**vals})
    contrasts=[]
    for name in names:
        if MODELS[name]['family']!='geometry':continue
        for rep in ['patch','pool','camera','register','dense']:
            values={(r['group'],r['regime']):r['median'] for r in rows if r['model']==name and r['representation']==rep and r['metric']=='D_cause' and r['layer']==max(cfg[name]['layers']) and r['phase'] in ['post','fused']}
            for a,b in [('Pair','Single'),('Full','Single'),('Full','Pair')]:
                diffs={g:values[g,a]-values[g,b] for g in da['core']}
                contrasts.append({'model':name,'representation':rep,'contrast':a+'-'+b,'by_group':diffs,'positive_groups':sum(v>0 for v in diffs.values()),'median_paired_difference':float(np.median(list(diffs.values())))})
    models={name:{'audit':ma[name],'runtime':json.loads((out/'audit'/f'{name}_runtime.json').read_text())} for name in names}
    summary={'status':'completed','models':names,'data':{k:da[k] for k in ['dataset_root','scenes','anchors','groups','sequences','frames','delta_m','core']},'model_provenance':models,'endpoint_group_medians':endpoints,'paired_contrasts':contrasts,'correctness':gate,'forward_fidelity':fidelity,'figures':pngs,'statistics':'group medians; no p-values; four groups nested in two scenes'}
    regression=out/'audit/legacy_feature_regression.json'
    if regression.exists():summary['legacy_feature_regression']=json.loads(regression.read_text())
    text=['# E1 model comparison report','',f'Completed models: {", ".join(names)}; {gate["shards_checked"]} validated feature shards.','',
          'The data, 25-condition factorial, I7 endpoint, within-regime static deltas, physical point IDs, metrics and statistical unit are shared. Existing two-model results remain in their original output root.','',
          '## Audited model differences','',
          '| Model | Patch | Input | Registers | Layers | Dense before prediction |','|---|---|---|---|---|---|']
    for name in names:
        s=MODELS[name];dense=f'{s["dense_dim"]} x {s["dense_hw"]} x {s["dense_hw"]}' if s['family']=='geometry' else 'N/A'
        text.append(f'| {LABELS[name]} | {s["patch"]} | {s["resolution"]} x {s["resolution"]} | {s["registers"]} | {cfg[name]["layers"]} | {dense} |')
    text += ['', 'DINOv2 and VGGT receive unchanged 512x512 pixels with symmetric 3px white padding, not rescaling or cropping. UV_model=UV+3; patch grid=(UV+3)/14-0.5. Original DINOv3/Omega remain 512x512. The shared core sets pass the same 8px interior rule in all four input spaces. Padding and different patch sizes remain model-preprocessing differences, not a perfectly matched architecture control.', '',
             'Both DINO models process I7 independently and use normalized block outputs. Both VGGT models use Single / Pair / Full and separate frame/pre and cross-frame/post halves. VGGT has 4 registers and global attention at every cross-frame block; Omega alone has register-only blocks and their patch-equality gate. VGGT dense hook is depth_head.scratch.output_conv2 input [128,518,518], whereas Omega is dense_head.proj input [256,128,128]. These are corresponding pre-prediction stages, not identical decoder features.', '',
             '## Endpoint results','',
             'Each entry is the median of group medians. D_cause averages the same 20 nonzero-r condition pairs within each physical point before group aggregation. Registers remain individual until metrics, followed by a group median over their actual count.','',
             '| Model | Regime | Representation | TC_e | TC_o | Signed cos | Abs cos | S_e | S_o | D_cause | C_norm |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in endpoints:text.append('| '+' | '.join([r['model'],r['regime'],r['representation']]+[f'{r[m]:.4f}' if m in r else 'N/A' for m in metrics])+' |')
    text += ['', '## Within-model paired context comparisons','', '| Model | Representation | Contrast | Positive groups | Median paired delta D |','|---|---|---|---|---|']
    for r in contrasts:text.append(f'| {r["model"]} | {r["representation"]} | {r["contrast"]} | {r["positive_groups"]}/{da["groups"]} | {r["median_paired_difference"]:+.4f} |')
    text += ['', '## Correctness and limitations','',f'- Core counts: {[da["core"][g]["count"] for g in sorted(da["core"])]}; no fallback.',
             f'- NaN/Inf: {gate["nonfinite"]}; Omega register-only patch max error: {gate["register_only_patch_max_error"]}. VGGT equality test is not applicable.',
             '- Saved features are compact float32 point/pool/camera/register vectors; no full patch tensors are persisted.',
             '- Model eval/inference mode, bf16 attention and float32 dense/metrics are shared; exact checkpoint hashes, source hashes, model runtime shapes and per-model spatial transforms are saved in audit/.',
             '- Read TC and norm scaling before interpreting tangent directions. Low TC does not establish a stable local linear regime. Raw sensitivities use different representation scales across architectures.',
             '- Same-r separation can reflect static global context. DINO baselines must be considered before attributing separation to reconstruction aggregation.',
             '- Special token first-frame roles differ in Single versus Pair/Full; prefer Pair vs Full. Register/patch routing comparisons are observational, not causal proof.',
             '- Dense decoders differ in channel count, spatial scale and operations. Agreement is auxiliary evidence, not elimination of all resolution effects.',
             '- Four groups share two scenes. Do not treat points/registers as independent samples or report population-level significance.', '', '## Source and reproduction','']
    for name in names:text += [f'- {name}: repo `{cfg[name]["repo"]}`; checkpoint `{cfg[name]["checkpoint"]}`; SHA256 `{ma[name]["checkpoint_sha256"]}`.']
    text += ['', '```bash', 'bash dynamic/scripts/run_e1_pipeline.sh --config dynamic/configs/e1_four_models.yaml', '```','', 'See metrics/*.csv for all layers, point distributions, matched-r pairs, individual registers and dense auxiliaries.','']
    for png in pngs:text.append(f'![{png}](../figures/{png})\n')
    dest=out/'report';dest.mkdir(exist_ok=True)
    write_json(dest/'E1_SUMMARY.json',summary);(dest/'E1_REPORT.md').write_text('\n'.join(text)+'\n')
    root=Path(__file__).resolve().parents[1]
    write_json(dest/'artifact_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'artifacts':{str(p.relative_to(out)):digest(p) for f in ['metrics','figures','report'] for p in sorted((out/f).glob('*')) if p.is_file() and p.name!='artifact_manifest.json'},'implementation_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.suffix in ['.py','.yaml','.sh','.md'] and 'outputs' not in p.parts}})
    print('Report:',dest/'E1_REPORT.md','figure groups=5, PNG/PDF pairs=',len(pngs))
