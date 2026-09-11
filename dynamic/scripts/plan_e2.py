#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.feature_io import arguments,configuration,write_json
from src.e2_dataset import E2Dataset
from src.model_registry import configured_models,regimes_for,shards_per_sequence

def main():
    p=arguments('Audit V2 structure and plan E2 without model forward')
    p.add_argument('--panel',default='all')
    a=p.parse_args();cfg=configuration(a);d=E2Dataset(cfg);audit=d.audit(hash_tracks=False)
    panels={}
    for name in cfg['panels']:
        seqs=d.panel_sequences(name);panels[name]={'sequences':len(seqs),'physical_contexts':len({s['physical_context_id'] for s in seqs}),
            'families':sorted({s['motion_family_id'] for s in seqs}),'model_regime_shards':len(seqs)*shards_per_sequence(cfg)}
    requested=d.sequences_for_request(a.panel)
    stages={};previous=set()
    for stage in ('stage1','stage2','stage3'):
        ids={s['sequence_id'] for s in d.sequences_for_request(stage)}
        stages[stage]={'unique_sequences':len(ids),'model_regime_shards':len(ids)*shards_per_sequence(cfg),
            'incremental_sequences':len(ids-previous),'incremental_shards':len(ids-previous)*shards_per_sequence(cfg)}
        previous=ids
    plan={'experiment':'E2','dataset_root':str(d.root),'output_root':cfg['output_root'],'physical_contexts':audit['physical_contexts'],
        'groups':audit['groups'],'unique_sequences_full_v2':audit['sequences'],'extension_context_ids':audit['extension_context_ids'],
        'family_coverage':audit['family_coverage'],'panels':panels,'stages':stages,'requested':a.panel,'requested_unique_sequences':len(requested),
        'requested_shards':len(requested)*shards_per_sequence(cfg),'models':{m:regimes_for(m) for m in configured_models(cfg)},
        'shards_per_sequence':shards_per_sequence(cfg),'analysis_families':sorted({f for p in d.panels_for_request(a.panel) for f in cfg['panels'][p]['families']}),
        'reuse':{'panel_a_intersection_panel_b_sequences':len(set(s['sequence_id'] for s in d.panel_sequences('core_x_confirmation'))&set(s['sequence_id'] for s in d.panel_sequences('xy_d004'))),
                 'panel_b_intersection_panel_d_sequences':len(set(s['sequence_id'] for s in d.panel_sequences('xy_d004'))&set(s['sequence_id'] for s in d.panel_sequences('scale_locality')))},
        'core_counts':{name:{pc:v['count'] for pc,v in rows.items()} for name,rows in audit['cores'].items()}}
    write_json(Path(cfg['output_root'])/'audit/e2_plan.json',plan);print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
