"""E2 V2 manifest planning with physical-context-level point correspondence."""
from collections import defaultdict
from pathlib import Path
import itertools
import json
import numpy as np
from .feature_io import read_jsonl,digest,json_safe
from .spatial_sampling import SpatialTransform,boundary_distances,farthest_points

EXPECTED_FAMILIES=('tx_d002','tx_d004','tx_d006','ty_d002','ty_d004','ty_d006')
CORE_FAMILIES={
    'core_tx004':('tx_d004',),
    'core_xy004':('tx_d004','ty_d004'),
    'core_scale6':EXPECTED_FAMILIES,
}
SMOKE_CONDITIONS={(0,0),(1,0),(-1,0),(2,0),(-2,0),(0,1),(0,-1),(0,2),(0,-2),(1,1)}


def intersect_canonical(records,min_points,max_points):
    """Intersect validity arrays only after exact canonical identity checks."""
    if not records:raise ValueError('No canonical records to intersect')
    ids0=np.asarray(records[0][0]);xyz0=np.asarray(records[0][1]);valid=[]
    for ids,xyz,good in records:
        if not np.array_equal(ids,ids0) or not np.array_equal(xyz,xyz0):raise ValueError('canonical point identity mismatch')
        valid.append(np.asarray(good,dtype=bool))
    common=np.logical_and.reduce(valid);selected=farthest_points(ids0[common],xyz0[common],max_points)
    if len(selected)<min_points:raise ValueError(f'only {len(selected)} common points; minimum is {min_points}')
    return selected,common


class E2Dataset:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['dataset_root'])
        self.sequences=sorted(read_jsonl(self.root/'manifests/sequences.jsonl'),key=lambda x:x['sequence_id'])
        self.frames={x['frame_id']:x for x in read_jsonl(self.root/'manifests/frames.jsonl')}
        self.contexts=read_jsonl(self.root/'manifests/contexts.jsonl')
        self.motion_families=read_jsonl(self.root/'manifests/motion_families.jsonl')
        self.matched=read_jsonl(self.root/'manifests/matched_relative_groups.jsonl')
        self.by_group=defaultdict(list);self.by_physical=defaultdict(lambda:defaultdict(list))
        for s in self.sequences:
            self.by_group[s['group_id']].append(s)
            self.by_physical[s['physical_context_id']][s['motion_family_id']].append(s)
        self.transform=SpatialTransform(input_hw=tuple(cfg['input_resolution']))
        self._tracks={};self.cores={};self.sequence_point_ids={}

    def track(self,s):
        key=s['sequence_id']
        if key not in self._tracks:
            with np.load(self.root/s['tracks_path']) as z:self._tracks[key]={k:z[k] for k in z.files}
        return self._tracks[key]

    def frames_for(self,s):return [self.frames[x] for x in s['frame_ids']]

    def family_coverage(self):return {pc:tuple(sorted(families)) for pc,families in self.by_physical.items()}

    def extended_contexts(self):
        wanted=set(EXPECTED_FAMILIES)
        return sorted(pc for pc,families in self.by_physical.items() if set(families)==wanted)

    def panel_sequences(self,panel,smoke=False):
        spec=self.cfg['panels'][panel];families=set(spec['families'])
        pcs=set(self.by_physical) if spec['context_rule']=='all' else set(self.extended_contexts())
        rows=[s for s in self.sequences if s['physical_context_id'] in pcs and s['motion_family_id'] in families]
        if smoke:
            pc=sorted({s['physical_context_id'] for s in rows})[0]
            rows=[s for s in rows if s['physical_context_id']==pc and (s['ego_level'],s['object_level']) in SMOKE_CONDITIONS]
        return rows

    def panels_for_request(self,panel):
        if panel in self.cfg['panels']:return [panel]
        if panel in self.cfg['stages']:return self.cfg['stages'][panel]
        if panel=='all':return list(self.cfg['panels'])
        raise ValueError(f'Unknown E2 panel/stage: {panel}')

    def sequences_for_request(self,panel,smoke=False):
        rows={s['sequence_id']:s for p in self.panels_for_request(panel) for s in self.panel_sequences(p,smoke)}
        return [rows[k] for k in sorted(rows)]

    def _canonical_and_valid(self,pc,families):
        records=[]
        for family in families:
            seqs=self.by_physical[pc].get(family,[])
            if len(seqs)!=25:raise ValueError(f'{pc}: {family} has {len(seqs)} conditions, expected 25')
            for s in seqs:
                tr=self.track(s);ids=tr['point_id'];xyz=tr['xyz_object_local']
                frame=self.frames_for(s)[-1]
                uv=self.transform.to_input(tr['projected_uv'][-1])
                mask=self.transform.mask(self.root/frame['target_mask'])
                dist=boundary_distances(mask,uv)
                good=tr['visible'][-1]&tr['in_front_of_camera'][-1]&tr['in_image'][-1]
                good &= np.isfinite(uv).all(1)&(dist>=self.cfg['min_mask_boundary_distance_px'])
                records.append((ids,xyz,good))
        try:
            selected,common=intersect_canonical(records,0,len(records[0][0]))
        except ValueError as exc:raise ValueError(f'{pc}: {exc}') from exc
        return np.asarray(records[0][0]),np.asarray(records[0][1]),common

    def build_cores(self):
        extended=set(self.extended_contexts());cores={name:{} for name in CORE_FAMILIES}
        for name,families in CORE_FAMILIES.items():
            pcs=sorted(self.by_physical) if name=='core_tx004' else sorted(extended)
            for pc in pcs:
                ids,xyz,valid=self._canonical_and_valid(pc,families)
                selected=farthest_points(ids[valid],xyz[valid],self.cfg['max_core_points'])
                if len(selected)<self.cfg['min_core_points']:
                    raise ValueError(f'{pc} {name}: only {len(selected)} common points; minimum is {self.cfg["min_core_points"]}')
                cores[name][pc]={'available':int(valid.sum()),'count':len(selected),'point_ids':selected.tolist(),
                    'families':list(families),'conditions_per_family':25,'fallback':False}
        self.cores=cores
        union=defaultdict(set)
        for panel,spec in self.cfg['panels'].items():
            core=cores[spec['core']]
            for s in self.panel_sequences(panel):union[s['sequence_id']].update(core[s['physical_context_id']]['point_ids'])
        self.sequence_point_ids={sid:sorted(ids) for sid,ids in union.items()}
        return cores

    def point_uv(self,s,point_ids):
        tr=self.track(s);index={int(p):i for i,p in enumerate(tr['point_id'])}
        try:idx=[index[int(p)] for p in point_ids]
        except KeyError as exc:raise ValueError(f'{s["sequence_id"]}: missing canonical point {exc.args[0]}')
        return tr['projected_uv'][-1,idx]

    def audit(self,hash_tracks=False):
        validation=json.loads((self.root/'validation_summary.json').read_text())
        if not validation.get('ok') or validation.get('geometry_only'):raise ValueError('V2 full validation is not passed/rendered')
        if len(self.contexts)!=24 or len(self.by_physical)!=24:raise ValueError('Expected exactly 24 V2 physical contexts')
        coverage=self.family_coverage();extended=self.extended_contexts()
        if len(extended)!=8:raise ValueError(f'Expected 8 six-family contexts, found {len(extended)}')
        if any('tx_d004' not in fs for fs in coverage.values()):raise ValueError('Every context must contain tx_d004')
        if set(x['motion_family_id'] for x in self.motion_families)!=set(EXPECTED_FAMILIES):raise ValueError('Unexpected motion family manifest')
        expected_conditions=set(itertools.product(range(-2,3),repeat=2))
        for pc,families in self.by_physical.items():
            for family,seqs in families.items():
                if len(seqs)!=25 or len({s['group_id'] for s in seqs})!=1:
                    raise ValueError(f'{pc}/{family}: expected one group_id and 25 sequences')
                context_id=seqs[0]['context_id']
                if not (self.root/'contexts'/context_id/family/'sequences').is_dir():
                    raise ValueError(f'Missing context payload directory: {context_id}/{family}')
        for group,seqs in self.by_group.items():
            if {(s['ego_level'],s['object_level']) for s in seqs}!=expected_conditions:raise ValueError(f'{group}: incomplete 5x5 condition grid')
            for s in seqs:
                if s['relative_level']!=s['object_level']-s['ego_level']:raise ValueError(f'{s["sequence_id"]}: invalid relative_level')
                ff=self.frames_for(s)
                if len(ff)!=8 or [x['frame_index'] for x in ff]!=list(range(8)):raise ValueError(f'{s["sequence_id"]}: invalid frames')
        if (len(self.by_group),len(self.sequences),len(self.frames),len(self.matched))!=(64,1600,12800,576):
            raise ValueError('Expected V2 totals 64/1600/12800/576')
        matched_by_group=defaultdict(list)
        for row in self.matched:matched_by_group[row['group_id']].append(row)
        for group,seqs in self.by_group.items():
            rows=matched_by_group[group]
            if {r['relative_level'] for r in rows}!=set(range(-4,5)):
                raise ValueError(f'{group}: matched-relative manifest does not cover r=-4..4')
            expected={s['sequence_id'] for s in seqs}
            actual={m['sequence_id'] for row in rows for m in row['members']}
            if actual!=expected:raise ValueError(f'{group}: matched-relative member IDs disagree with sequences')
        self.build_cores()
        fingerprints={name:digest(self.root/name) for name in [
            'validation_summary.json','manifests/sequences.jsonl','manifests/frames.jsonl',
            'manifests/motion_families.jsonl','manifests/contexts.jsonl','manifests/matched_relative_groups.jsonl','manifests/splits.json']}
        if hash_tracks:
            for s in self.sequences:fingerprints[s['tracks_path']]=digest(self.root/s['tracks_path'])
        panel_counts={p:len(self.panel_sequences(p)) for p in self.cfg['panels']}
        expected_panels={'core_x_confirmation':600,'xy_d004':400,'representation_specialization':400,'scale_locality':1200}
        if panel_counts!=expected_panels:raise ValueError(f'Unexpected E2 panel sequence counts: {panel_counts}')
        if sorted({float(x['delta_m']) for x in self.motion_families})!=sorted(self.cfg['scale_deltas_m']):
            raise ValueError('Configured scale list disagrees with motion-family manifest')
        return {'experiment':'E2','dataset_root':str(self.root),'physical_contexts':24,'groups':64,'sequences':1600,
            'frames':12800,'family_coverage':{k:list(v) for k,v in coverage.items()},'extension_context_ids':extended,
            'panel_sequence_counts':panel_counts,'cores':self.cores,'sequence_point_ids':self.sequence_point_ids,
            'input_manifest_sha256':fingerprints,'generation_validation_sha256':fingerprints['validation_summary.json'],
            'splits':json.loads((self.root/'manifests/splits.json').read_text())}


def validate_cache_metadata(metadata,expected):
    for key,value in expected.items():
        stored=json_safe(metadata.get(key));wanted=json_safe(value)
        if stored!=wanted:raise ValueError(f'Feature cache mismatch for {key}: {stored!r} != {wanted!r}')
    return True
