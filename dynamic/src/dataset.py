from pathlib import Path
from collections import defaultdict
import json
import itertools
import numpy as np
from .feature_io import read_jsonl, digest
from .spatial_sampling import SpatialTransform, boundary_distances, farthest_points, occupancy


def select_motion_families(sequences, family_ids=None):
    """Keep a homogeneous axis/scale run; V1 without a filter is unchanged."""
    if family_ids is not None:
        if not isinstance(family_ids, (list, tuple)) or not family_ids:
            raise ValueError('motion_family_ids must be a nonempty list')
        available = {s.get('motion_family_id') for s in sequences}
        unknown = set(family_ids) - available
        if unknown:
            raise ValueError(f'Unavailable motion_family_ids: {sorted(unknown)}')
        sequences = [s for s in sequences if s.get('motion_family_id') in family_ids]
    protocols = {(s.get('delta_m'), tuple(s.get('motion_axis_world', [1, 0, 0]))) for s in sequences}
    if len(protocols) > 1:
        raise ValueError('E1 requires one homogeneous motion family per run; set motion_family_ids: [tx_d004] or another single family')
    if not sequences:
        raise ValueError('No sequences selected')
    return sequences


class PilotDataset:
    def __init__(self, cfg):
        self.cfg=cfg; self.root=Path(cfg['dataset_root'])
        self.sequences=sorted(read_jsonl(self.root/'manifests/sequences.jsonl'),key=lambda s:s['sequence_id'])
        self.sequences=select_motion_families(self.sequences, cfg.get('motion_family_ids'))
        self.frames={f['frame_id']:f for f in read_jsonl(self.root/'manifests/frames.jsonl')}
        if cfg.get('motion_family_ids') is not None:
            selected_frames={fid for s in self.sequences for fid in s['frame_ids']}
            self.frames={fid:f for fid,f in self.frames.items() if fid in selected_frames}
        self.groups=defaultdict(list)
        for s in self.sequences: self.groups[s['group_id']].append(s)
        self.transform=SpatialTransform(input_hw=tuple(cfg['input_resolution']))
        self.core={};self.tracks={}

    def frames_for(self,s): return [self.frames[f] for f in s['frame_ids']]
    def track(self,s):
        if s['sequence_id'] not in self.tracks:
            with np.load(self.root/s['tracks_path']) as a:
                self.tracks[s['sequence_id']]={k:a[k] for k in a.files}
        return self.tracks[s['sequence_id']]
    def mask(self,s): return self.transform.mask(self.root/self.frames_for(s)[-1]['target_mask'])
    def point_uv(self,s):
        tr=self.track(s); ix={int(p):i for i,p in enumerate(tr['point_id'])}
        return tr['projected_uv'][-1,[ix[p] for p in self.core[s['group_id']]['point_ids']]]

    def audit(self):
        val=json.loads((self.root/'validation_summary.json').read_text()); assert val['ok'] and not val['geometry_only']
        schema=None; displacements=[];boundary=[];pool_counts=[];max_same_r=0.;max_projection=0.; fingerprints={}
        for name in ['config_used.yaml','validation_summary.json','manifests/sequences.jsonl','manifests/frames.jsonl','manifests/matched_relative_groups.jsonl','manifests/splits.json']:
            fingerprints[name]=digest(self.root/name)
        for g, seqs in self.groups.items():
            assert {(s['ego_level'],s['object_level']) for s in seqs}==set(itertools.product(range(-2,3),repeat=2))
            good=[]; ids0=None;xyz0=None
            for s in seqs:
                assert s['relative_level']==s['object_level']-s['ego_level']
                frames=self.frames_for(s);assert len(frames)==8 and [f['frame_index'] for f in frames]==list(range(8))
                t=self.track(s); schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in t.items()}
                ids=t['point_id'];xyz=t['xyz_object_local']
                if ids0 is None:ids0=ids;xyz0=xyz
                assert np.array_equal(ids,ids0) and np.array_equal(xyz,xyz0)
                for fi,f in enumerate(frames):
                    for key in ['rgb','target_mask','depth_npy','frame_metadata']:
                        assert (self.root/f[key]).is_file(),f[key]
                    assert f['image_size']==[512,512]
                    K=np.array(f['K']); cam=t['xyz_camera'][fi]; proj=cam@K.T;proj=proj[:,:2]/proj[:,2:]
                    max_projection=max(max_projection,float(np.max(np.abs(proj-t['projected_uv'][fi]))))
                uv=self.transform.to_input(t['projected_uv'][-1]);mask=self.mask(s)
                dist=boundary_distances(mask,uv)
                valid=t['visible'][-1]&t['in_front_of_camera'][-1]&t['in_image'][-1]&np.isfinite(uv).all(1)&(dist>=self.cfg['min_mask_boundary_distance_px'])
                valid &= (uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<512)&(uv[:,1]<512)
                good.append(valid)
                pool_counts.append({'sequence_id':s['sequence_id'],'clean_patches':int((occupancy(mask,16)>=self.cfg['clean_patch_occupancy']).sum())})
                fingerprints[s['tracks_path']]=digest(self.root/s['tracks_path'])
                # All source frames bound into provenance; no dataset writes.
                for f in frames:
                    for key in ['rgb','target_mask','depth_npy']:
                        fingerprints[f[key]]=digest(self.root/f[key])
            common=np.logical_and.reduce(good); selected=farthest_points(ids0[common],xyz0[common],self.cfg['max_core_points'])
            if len(selected)<self.cfg['min_core_points']:
                raise ValueError(f'{g}: only {len(selected)} common core points. Explicit metric-specific fallback plan required; no silent fallback.')
            self.core[g]={'available':int(common.sum()),'point_ids':selected.tolist(),'count':len(selected),'fallback':False,'conditions':25}
            for s in seqs:
                t=self.track(s);idx=np.flatnonzero(np.isin(t['point_id'],selected));uv=t['projected_uv'][-1]
                dist=boundary_distances(self.mask(s),self.transform.to_input(uv))[idx]
                boundary.extend({'group':g,'sequence_id':s['sequence_id'],'point_id':int(p),'distance_px':float(d)} for p,d in zip(t['point_id'][idx],dist))
                if (s['ego_level']==0) != (s['object_level']==0):
                    ds=np.linalg.norm(uv[idx]-t['projected_uv'][0,idx],axis=1)
                    displacements.append({'group':g,'e':s['ego_level'],'o':s['object_level'],'median_px':float(np.median(ds)),'min_px':float(ds.min()),'max_px':float(ds.max()),'median_patch_ratio':float(np.median(ds)/16)})
            for a,b in itertools.combinations(seqs,2):
                if a['relative_level']==b['relative_level']:
                    max_same_r=max(max_same_r,float(np.max(np.linalg.norm(self.point_uv(a)-self.point_uv(b),axis=1))))
        assert max_same_r<.25 and max_projection<.01
        return {'dataset_root':str(self.root),'scenes':len({s['scene_id'] for s in self.sequences}),'anchors':len(self.groups),'groups':len(self.groups),'sequences':len(self.sequences),'frames':len(self.frames),'delta_m':self.sequences[0]['delta_m'],'resolution':[512,512],'track_schema':schema,'core':self.core,'displacements':displacements,'boundary_distances':boundary,'clean_patch_counts':pool_counts,'generation_validation':{'ok':val['ok'],'checks':val['checks'],'sha256':fingerprints['validation_summary.json']},'live_max_same_r_final_uv_error_px':max_same_r,'live_max_track_reprojection_error_px':max_projection,'splits':json.loads((self.root/'manifests/splits.json').read_text()),'input_sha256':fingerprints}
