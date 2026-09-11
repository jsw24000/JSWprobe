#!/usr/bin/env python3
import csv,json,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.e2_dataset import E2Dataset
from src.feature_io import arguments,configuration,config_hash,digest,read_jsonl,records_hash,write_json

def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def numeric(rows,key):return np.asarray([float(r[key]) for r in rows if r.get(key) not in ('',None,'nan')],float)

def primary(rows,c,representation='patch'):
    return [r for r in rows if r.get('representation')==representation and r.get('regime')=='Single'
            and int(r['layer'])==max(c[r['model']]['layers'])
            and r['phase']==('norm' if r['model'].startswith('dino') else ('fused' if representation=='dense' else 'post'))]

def save(fig,folder,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    p=arguments('Build conservative E2 report from completed panel metrics')
    p.add_argument('--panels',nargs='+',default=['core_x_confirmation','xy_d004','representation_specialization','scale_locality'])
    a=p.parse_args();c=configuration(a);out=Path(c['output_root']);available=[]
    dataset=E2Dataset(c);manifest_rows=read_jsonl(out/'features/feature_manifest.jsonl') if (out/'features/feature_manifest.jsonl').exists() else []
    for panel in a.panels:
        folder=out/'metrics'/panel
        provenance=folder/'analysis_provenance.json'
        if provenance.exists():
            value=json.loads(provenance.read_text())
            allowed={s['sequence_id'] for s in dataset.panel_sequences(panel)}
            selection=[r for r in manifest_rows if r['sequence_id'] in allowed]
            if value.get('config_hash')!=config_hash(c) or value.get('feature_selection_sha256')!=records_hash(selection):
                raise ValueError(f'Stale analysis provenance for {panel}')
            available.append(panel)
    if not available:raise ValueError('No completed E2 panel analysis found')
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True);made=[]
    if 'core_x_confirmation' in available:
        rows=read_csv(out/'metrics/core_x_confirmation/local_geometry.csv');fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,metric in zip(axes,('cos_signed','M_c','TC_e')):
            rr=[r for r in rows if r['metric']==metric and r['representation']=='patch' and r['phase'] in ('norm','post')]
            for model in sorted({r['model'] for r in rr}):
                phase='norm' if model.startswith('dino') else 'post';z=sorted([r for r in rr if r['model']==model and r['regime']=='Single' and r['phase']==phase],key=lambda x:int(x['layer']))
                by_layer=defaultdict(list)
                for row in z:by_layer[int(row['layer'])].append(float(row['median']))
                ax.plot(sorted(by_layer),[np.median(by_layer[layer]) for layer in sorted(by_layer)],marker='o',label=model)
            ax.set_title(metric);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);save(fig,figures,'e2_figure1_panel_a');made.append('e2_figure1_panel_a')
    if 'xy_d004' in available:
        rows=read_csv(out/'metrics/xy_d004/local_geometry.csv');fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,metric in zip(axes,('cos_signed','M_c','M_r')):
            rr=[r for r in primary(rows,c) if r['metric']==metric]
            labels=sorted({(r['model'],r['axis']) for r in rr});data=[numeric([r for r in rr if (r['model'],r['axis'])==x],'median') for x in labels]
            ax.boxplot(data,labels=[f'{m}\n{axis}' for m,axis in labels]);ax.set_title(metric)
        save(fig,figures,'e2_figure2_xy_local');made.append('e2_figure2_xy_local')
        sub=read_csv(out/'metrics/xy_d004/xy_subspace_context_metrics.csv');fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,key in zip(axes,('rho_e','rho_o','subspace_overlap')):
            rr=primary(sub,c);models=sorted({r['model'] for r in rr})
            ax.boxplot([numeric([r for r in rr if r['model']==m],key+'_median') for m in models],labels=models);ax.set_title(key)
        save(fig,figures,'e2_figure3_xy_subspace');made.append('e2_figure3_xy_subspace')
        common=read_csv(out/'metrics/xy_d004/matched_common_mode_pairs.csv');fig,ax=plt.subplots(figsize=(9,5))
        rr=primary(common,c)
        for dc in sorted({float(r['delta_c']) for r in rr}):
            z=[r for r in rr if float(r['delta_c'])==dc];xs=sorted({int(r['r']) for r in z});ax.plot(xs,[np.median([float(q['D_c_given_r']) for q in z if int(q['r'])==x]) for x in xs],marker='o',label=f'delta_c={dc:g}')
        ax.set_xlabel('relative level r');ax.set_ylabel('D_c_given_r');ax.legend();save(fig,figures,'e2_figure4_common_mode');made.append('e2_figure4_common_mode')
    if 'representation_specialization' in available:
        rows=read_csv(out/'metrics/representation_specialization/local_geometry.csv');fig,ax=plt.subplots(figsize=(10,5))
        rr=[r for r in rows if r['metric']=='M_c' and r['model'] in ('vggt','vggt_omega') and r['phase'] in ('post','fused')]
        labels=sorted({(r['model'],r['representation']) for r in rr});ax.boxplot([numeric([r for r in rr if (r['model'],r['representation'])==x],'median') for x in labels],labels=[f'{a}\n{b}' for a,b in labels]);ax.set_ylabel('M_c')
        save(fig,figures,'e2_figure5_specialization');made.append('e2_figure5_specialization')
    if 'scale_locality' in available:
        rows=read_csv(out/'metrics/scale_locality/scale_consistency_context_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,key in zip(axes,('cross_scale_cosine','derivative_norm_ratio')):
            rr=primary(rows,c);pairs=sorted({(r['delta_a'],r['delta_b']) for r in rr})
            ax.boxplot([numeric([r for r in rr if (r['delta_a'],r['delta_b'])==p],key+'_median') for p in pairs],labels=[f'{a}/{b}' for a,b in pairs]);ax.set_title(key)
        save(fig,figures,'e2_figure6_scale_locality');made.append('e2_figure6_scale_locality')
        proj=read_csv(out/'metrics/scale_locality/projection_context_summary.csv');fig,ax=plt.subplots(figsize=(9,5));labels=sorted({(r['axis'],r['cause']) for r in proj})
        ax.boxplot([numeric([r for r in proj if (r['axis'],r['cause'])==x],'median_px_median') for x in labels],labels=[f'{a}/{b}' for a,b in labels]);ax.set_ylabel('context median target displacement px')
        save(fig,figures,'e2_figure7_input_controls');made.append('e2_figure7_input_controls')
    created=datetime.now(timezone.utc).isoformat();summary={'experiment':'E2','status':'partial' if set(available)!=set(c['panels']) else 'completed',
        'created_utc':created,'panels_available':available,'figures':made,'statistical_unit':'physical_context_id; point distributions are diagnostics, no p-values',
        'interpretation':{'D_c_given_r':'finite common/reference-frame variation at fixed nonzero relative translation',
            'C_norm':'compensated common-mode sensitivity at r=0; related but not the same formula',
            'subspace':'continuous singular-value ratios are primary; two-angle interpretation requires two well-conditioned rank-2 spaces'},
        'limitations':['The target/background design is partially crossed; 24 contexts are not claimed as fully independent scenes.',
            'Finite-difference directions are tangent-like until cross-scale locality is established.',
            'Matched r fixes target-camera relative translation but does not make full RGB observations identical.',
            'Blocks and raw feature norms are not calibrated or exactly homologous across architectures.']}
    text=['# E2 representation geometry report','',f'Status: {summary["status"]}. Available panels: {", ".join(available)}.','',
        'E2 measures finite-difference response geometry under controlled physical perturbation families. It does not claim causal identification, semantic understanding, or proven disentanglement.','',
        'The primary reporting unit is `physical_context_id`. Point-level values support visualization and diagnostics; they are not treated as independent replicates. No p-values are produced.','',
        '`D_c_given_r` measures representation variation along the common/world coordinate while nonzero target-camera relative translation is held fixed. `C_norm` measures compensated common-mode response at r=0. They probe related questions but are not special cases of one formula.','',
        'X/Y subspace results report exact singular values and continuous rank ratios. Two-dimensional principal-angle interpretation is enabled only when both response matrices pass the configured rank and conditioning gates.','',
        'Scale results determine whether finite-difference directions are stable enough to support a tangent-like interpretation. Matched-r does not imply pixel-identical observations because the background/reference frame can change.','']
    for name in made:text.append(f'![{name}](../figures/{name}.png)\n')
    report=out/'report';report.mkdir(parents=True,exist_ok=True);write_json(report/'E2_SUMMARY.json',summary);(report/'E2_REPORT.md').write_text('\n'.join(text)+'\n')
    root=Path(__file__).resolve().parents[1]
    write_json(report/'artifact_manifest.json',{'experiment':'E2','created_utc':created,'artifacts':{str(p.relative_to(out)):digest(p) for folder in ('metrics','figures','report') for p in sorted((out/folder).rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},
        'implementation_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.suffix in ('.py','.yaml','.sh','.md') and 'outputs' not in p.parts}})
    print('E2 report:',report/'E2_REPORT.md','panels=',available)
if __name__=='__main__':main()
