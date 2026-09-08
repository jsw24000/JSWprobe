"""Independent V2 protocol validation, reusing only axis-independent V1 raster/track checks."""
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from memory_scene_blender.scripts import validate_ego_object_x_factorial_dataset as old
from memory_scene_blender.object_translation.label_utils import corners_inside_room, validate_no_collision, aabb_corners
from memory_scene_blender.object_translation.manifest_utils import write_json

MANIFEST_IDS = dict(scenes='scene_id',targets='target_id',backgrounds='background_id',contexts='context_id',
    motion_families='motion_family_id',anchors='physical_context_id',base_cameras='physical_context_id',
    groups='group_id',sequences='sequence_id',frames='frame_id',matched_relative_groups='matched_relative_group_id',track_sets='track_set_id')
IDENTITY = ('context_id','physical_context_id','target_id','background_id','motion_family_id')
CANONICAL = ('point_id','xyz_object_local','normal_object_local','part_index','polygon_index','sampling_triangle_index')


def validate_dataset(root, geometry_only=False):
    root=Path(root).resolve(); state=old.ValidationState(); checks={}
    def run(name, fn):
        before=sum(state.error_counts.values())
        try: fn()
        except Exception as exc: state.error(name, f'{type(exc).__name__}: {exc}')
        checks[name]={'status':'failed' if sum(state.error_counts.values())>before else 'passed'}
    def require_files():
        for name in ['config_used.yaml','dataset_summary.json','context_plan.json','selection_report.json','provenance.json',
                     '.ego_object_factorial_v2_dataset','manifests/splits.json','README.md',
                     'reports/coverage_summary.json','reports/coverage_matrix.csv','reports/selection_summary.csv','reports/projected_motion_scale.csv']:
            assert (root/name).is_file(),name
        for name in MANIFEST_IDS: assert (root/'manifests'/f'{name}.jsonl').is_file(),name
    run('required_files',require_files)
    try:
        config=old._read_config(root/'config_used.yaml');summary=old._read_json(root/'dataset_summary.json')
        plan=old._read_json(root/'context_plan.json')
        m={name:old._read_jsonl(root/'manifests'/f'{name}.jsonl') for name in MANIFEST_IDS}
    except Exception as exc:
        state.error('schema',str(exc));m=None
    if m is not None:
        def schema():
            for name,rows in m.items():
                ids=[r[MANIFEST_IDS[name]] for r in rows]
                assert len(ids)==len(set(ids)),f'duplicate {name} ID'
                assert all(isinstance(i,str) and i for i in ids),name
            for s in m['sequences']:
                assert set((*IDENTITY,'panel_membership','delta_m','motion_axis_world','frame_ids','tracks_path','canonical_points_path')).issubset(s)
            for f in m['frames']:
                assert set((*IDENTITY,*(f'actual_{r}_{suffix}' for r in ('ego','object','relative') for suffix in ('amplitude_m','translation_world_m')))).issubset(f)
            assert {r['sequence_id'] for r in m['track_sets']}=={s['sequence_id'] for s in m['sequences']}
        run('schema_unique_ids',schema)
        def finite_paths():
            for path in itertools.chain(root.rglob('*.json'),root.rglob('*.jsonl'),[root/'config_used.yaml']):
                rows=old._read_jsonl(path) if path.suffix=='.jsonl' else [old._read_config(path)]
                assert old._json_is_finite(rows),str(path.relative_to(root))
            def visit(value,key=''):
                if isinstance(value,dict):
                    for k,v in value.items(): visit(v,k)
                elif isinstance(value,list):
                    for v in value: visit(v,key)
                elif isinstance(value,str) and (key.endswith(('_path','_dir')) or key in ('rgb','depth','depth_exr','depth_npy','normal','albedo','target_mask','instance','semantic','object_id','frame_metadata')):
                    p=Path(value)
                    assert not p.is_absolute() and '..' not in p.parts,value
                    assert (root/p).resolve().is_relative_to(root),value
            visit(m)
        run('json_finiteness_relative_paths',finite_paths)
        expected_pairs={(f'{cat}_v{v:02d}',f'bg_{(2*v+c+d)%6:03d}') for c,cat in enumerate(('chair','armchair','side_table','small_cabinet')) for v in range(3) for d in (0,1)}
        def crossing():
            planned=plan['contexts']
            assert len(planned)==24 and {(c['target_id'],c['background_id']) for c in planned}==expected_pairs
            assert set(Counter(c['target_id'] for c in planned).values())=={2}
            assert set(Counter(c['background_id'] for c in planned).values())=={4}
            for b in range(6): assert len({c['target_category'] for c in planned if c['background_id']==f'bg_{b:03d}'})==4
            ext=[c for c in planned if c['extension']]
            assert len(ext)==8 and set(Counter(c['target_category'] for c in ext).values())=={2}
            assert len({c['background_id'] for c in ext})==6
            for cidx,cat in enumerate(('chair','armchair','side_table','small_cabinet')):
                assert {c['target_variant'] for c in ext if c['target_category']==cat}=={cidx%2}
            assert len(plan['targets'])==12 and len(plan['backgrounds'])==6
            for c in planned:
                assert c['target_category'] in ('chair','armchair','side_table','small_cabinet') and c['target_variant'] in (0,1,2)
                assert c['target_id']==f"{c['target_category']}_v{c['target_variant']:02d}"
                assert c['context_id']==c['background_id']+'__'+c['target_id']
                for field,identity in [('canonical_sampling_seed',c['target_id']),('context_seed',c['context_id'])]:
                    expected_seed=int.from_bytes(hashlib.sha256(f"{config['seed']}:{identity}".encode()).digest()[:4],'big')
                    assert c[field]==expected_seed
            assert {r['target_id'] for r in m['targets']}=={r['target_id'] for r in m['contexts']}
            assert {r['background_id'] for r in m['backgrounds']}=={r['background_id'] for r in m['contexts']}
            assert {r['scene_id'] for r in m['scenes']}=={r['context_id'] for r in m['contexts']}
            assert m['scenes']==m['contexts']
            for name,key in [('targets','target_id'),('backgrounds','background_id')]:
                planned_rows={r[key]:r for r in plan[name]}
                assert all(r==planned_rows[r[key]] for r in m[name])
            assert plan['backgrounds']==config['backgrounds']
            assert len({(b['room_layout_index'],b['static_layout_id']%3) for b in plan['backgrounds']})==6
            actual=m['contexts']; expected=planned if config['mode_used']=='full' else [ext[0]]
            assert [c['context_id'] for c in actual]==[c['context_id'] for c in expected]
            assert plan['realized_contexts']==[{**e,'generated_motion_family_ids':a['generated_motion_family_ids']} for e,a in zip(expected,actual)]
            for a,e in zip(actual,expected):
                assert all(a[k]==v for k,v in e.items()),a['context_id']
        run('crossing_balance_and_realized_plan',crossing)
        groups={g['group_id']:g for g in m['groups']}; seqs=defaultdict(list);frames=old.group_frames(m['frames'])
        for s in m['sequences']: seqs[s['group_id']].append(s)
        def coverage_counts():
            family_expected={f't{a}_d00{d}':([int(a=='x'),int(a=='y'),0],d/100) for a in 'xy' for d in (2,4,6)}
            assert len(m['motion_families'])==6
            for f in m['motion_families']:
                axis,delta=family_expected[f['motion_family_id']]
                assert f['axis_world']==axis and f['delta_m']==delta
                assert config['motion_families'][f['motion_family_id']]==dict(axis_world=axis,delta_m=delta)
            assert set(groups)==set(seqs)
            for c in m['contexts']:
                gs=[g for g in groups.values() if g['context_id']==c['context_id']]
                expected=c['motion_family_ids'] if config['mode_used']=='full' or config.get('smoke_all_families') else config['effective_mode_settings']['motion_family_ids']
                assert len(gs)==len(expected) and {g['motion_family_id'] for g in gs}==set(expected)
                assert 'tx_d004' in expected
                for g in gs:
                    assert g['group_id']==g['physical_context_id']+'__'+g['motion_family_id']
                    axis,delta=family_expected[g['motion_family_id']]
                    assert g['delta_m']==delta and g['motion_axis_world']==axis
            n=len(groups)
            expected_counts=dict(contexts=24,groups=64,sequences=1600,frames=12800) if config['mode_used']=='full' else dict(contexts=1,groups=6 if config.get('smoke_all_families') else len(config['effective_mode_settings']['motion_family_ids']),sequences=25*n,frames=200*n)
            actual={k:len(m[k]) for k in expected_counts}
            assert actual==expected_counts and summary['counts']==actual and summary['complete']
            if config['mode_used']=='full':
                assert len(m['targets'])==12 and len(m['backgrounds'])==6
                assert sum(g['motion_family_id']=='tx_d004' for g in groups.values())==24
                assert sum(g['motion_family_id']!='tx_d004' for g in groups.values())==40
            assert len(m['anchors'])==len(m['contexts'])==len(m['base_cameras'])
            assert len(m['track_sets'])==len(m['sequences']) and len(m['matched_relative_groups'])==9*n
            assert config['effective_mode_settings']['num_frames']==8 and config['effective_mode_settings']['resolution']==[512,512]
            if config['mode_used']=='full': assert config['effective_mode_settings']['samples']==8
            assert set(frames)=={s['sequence_id'] for s in m['sequences']}
            state.metrics['counts']=actual
        run('motion_coverage_counts_no_partial_full',coverage_counts)
        def identity():
            canonical={}; cameras={r['physical_context_id']:r for r in m['base_cameras']};anchors={r['physical_context_id']:r for r in m['anchors']}
            for g in groups.values():
                pc=g['physical_context_id'];a=anchors[pc];cam=cameras[pc]
                assert g['anchor_root_xy']==a['anchor_root_xy'] and g['base_camera_id']==cam['base_camera_id']
                assert g['base_camera']['blender_camera_to_world']==cam['blender_camera_to_world']
                assert g['object_anchor_id']==a['object_anchor_id']
                assert pc==g['context_id']+'__'+g['object_anchor_id']+'__'+g['base_camera_id']
                expected_panels=['core_confirmation'] if g['motion_family_id']=='tx_d004' else []
                context=next(c for c in m['contexts'] if c['context_id']==g['context_id'])
                assert all(g[k]==context[k] for k in ('target_id','background_id','context_id','scene_id','canonical_points_path'))
                assert all(a[k]==cam[k]==g[k] for k in ('target_id','background_id','context_id'))
                if context['extension']:expected_panels.append('motion_extension')
                assert g['panel_membership']==expected_panels
                path=f"canonical_targets/{g['target_id']}/canonical_surface_points.npz"
                assert g['canonical_points_path']==path
                with np.load(root/path) as tr: payload={k:tr[k] for k in CANONICAL}
                assert all(np.isfinite(v).all() for v in payload.values())
                assert np.allclose(np.linalg.norm(payload['normal_object_local'],axis=1),1,atol=1e-7,rtol=0)
                if g['target_id'] in canonical:
                    assert all(np.array_equal(v,canonical[g['target_id']][k]) for k,v in payload.items())
                canonical[g['target_id']]=payload
                for s in seqs[g['group_id']]:
                    assert all(s[k]==g[k] for k in (*IDENTITY,'panel_membership','canonical_points_path','delta_m','motion_axis_world','object_anchor_id','base_camera_id'))
                    with np.load(root/s['tracks_path']) as tr:
                        assert all(np.array_equal(tr[k],payload[k]) for k in CANONICAL)
            target_rows={r['target_id']:r for r in m['targets']}
            for c in m['contexts']:
                assert c['canonical_points_path']==target_rows[c['target_id']]['canonical_points_path']
                assert c['canonical_sampling_seed']==target_rows[c['target_id']]['canonical_sampling_seed']
        run('canonical_identity_extension_pairing',identity)
        def geometry():
            max_xyz=0.;max_uv=0.
            # Saved Blender object transforms are float32 and V1 helpers serialize to 8 decimals.
            xyz_tol=max(2e-6,float(config['validation']['relative_equation_atol_m']))
            # Blender stores object transforms as float32; the previous 2e-7 boundary
            # rejected a valid endpoint at 2.114e-7 m after JSON serialization.
            object_tol=max(3e-7,float(config['validation']['displacement_atol_m']))
            for gid,ss in seqs.items():
                assert len(ss)==25 and {(s['ego_level'],s['object_level']) for s in ss}==set(itertools.product(range(-2,3),repeat=2))
                g=groups[gid];axis=np.array(g['motion_axis_world']);delta=g['delta_m']; reference={};first_zero=None
                base=np.array(g['base_camera']['blender_camera_to_world'])
                for s in ss:
                    e,o=s['ego_level'],s['object_level'];r=o-e
                    assert s['relative_level']==r
                    assert np.allclose([s['ego_amplitude_m'],s['object_amplitude_m'],s['relative_amplitude_m']],np.array([e,o,r])*delta,atol=1e-12,rtol=0)
                    ff=frames[s['sequence_id']]
                    assert s['num_frames']==8
                    assert len(ff)==8 and [f['frame_index'] for f in ff]==list(range(8))
                    assert s['frame_ids']==[f['frame_id'] for f in ff]
                    with np.load(root/s['tracks_path']) as tr:
                        xyz=tr['xyz_camera'];uv=tr['projected_uv']
                        assert np.isfinite(xyz).all() and np.isfinite(uv).all()
                        if r in reference:
                            dx=float(np.max(np.abs(xyz-reference[r][0])));du=float(np.max(np.abs(uv-reference[r][1])))
                            max_xyz=max(max_xyz,dx);max_uv=max(max_uv,du)
                            assert dx<xyz_tol and du<config['validation']['matched_uv_max_tolerance_px']
                        else:reference[r]=(xyz.copy(),uv.copy())
                        if r==0:
                            assert np.max(np.abs(xyz-xyz[0]))<xyz_tol
                            assert np.max(np.abs(uv-uv[0]))<config['validation']['compensated_uv_max_tolerance_px']
                    zero=np.array(ff[0]['object_to_world'])
                    if first_zero is None:first_zero=zero
                    assert np.allclose(zero,first_zero,atol=1e-7,rtol=0)
                    assert np.allclose(zero[:3,:3],np.eye(3),atol=1e-7,rtol=0) and abs(zero[2,3])<1e-7
                    assert np.allclose(zero[:2,3],g['anchor_root_xy'],atol=1e-7,rtol=0)
                    for i,f in enumerate(ff):
                        assert f['sequence_id']==s['sequence_id'] and f['group_id']==gid
                        assert f['image_size']==[512,512]
                        assert old._read_json(root/f['frame_metadata'])==f
                        assert all(f[k]==s[k] for k in IDENTITY)
                        alpha=i/7;assert abs(f['alpha_t']-alpha)<1e-12
                        for role,l in [('ego',e),('object',o),('relative',r)]:
                            assert abs(f[f'actual_{role}_amplitude_m']-alpha*l*delta)<1e-12
                            assert np.allclose(f[f'actual_{role}_translation_world_m'],axis*alpha*l*delta,atol=1e-12,rtol=0)
                        cam=np.array(f['blender_camera_to_world']);obj=np.array(f['object_to_world'])
                        expected_cam=base.copy();expected_cam[:3,3]+=axis*alpha*e*delta
                        expected_obj=zero.copy();expected_obj[:3,3]+=axis*alpha*o*delta
                        assert np.allclose(cam,expected_cam,atol=1e-7,rtol=0)
                        assert np.allclose(obj,expected_obj,atol=object_tol,rtol=0)
                        expected_bbox=np.array(ff[0]['target_bbox_corners_world'])+axis*alpha*o*delta
                        assert np.allclose(f['target_bbox_corners_world'],expected_bbox,atol=object_tol,rtol=0)
                        assert np.allclose(np.array(f['world_to_camera'])@np.array(f['camera_to_world']),np.eye(4),atol=1e-7,rtol=0)
                        assert np.allclose(f['world_to_camera'],np.diag([1,-1,-1,1])@np.linalg.inv(cam),atol=1e-7,rtol=0)
                        assert f['K']==g['base_camera']['intrinsics']['K']
                        k=np.array(f['K']);fx=512/(2*np.tan(np.deg2rad(g['base_camera']['fov_degrees'])/2))
                        assert np.allclose(k,[[fx,0,256],[0,fx,256],[0,0,1]],atol=1e-7,rtol=0)
                        assert f['look_at_recomputed'] is False
            state.metrics.update(max_same_r_xyz_error_m=max_xyz,max_same_r_uv_error_px=max_uv, effective_same_r_xyz_tolerance_m=xyz_tol, effective_object_displacement_tolerance_m=object_tol)
        run('interventions_endpoints_frame0_same_r_compensation',geometry)
        def matched():
            expected={(gid,r):{s['sequence_id'] for s in ss if s['relative_level']==r} for gid,ss in seqs.items() for r in range(-4,5)}
            actual={}
            for row in m['matched_relative_groups']:
                key=(row['group_id'],row['relative_level']);assert key not in actual
                actual[key]={s['sequence_id'] for s in row['members']}
                assert row['motion_family_id']==groups[row['group_id']]['motion_family_id']
                assert row['member_count']==len(row['members'])==len(actual[key])
            assert actual==expected
        run('matched_relative_family_isolation',matched)
        def sweeps():
            scenes={c['context_id']:c for c in m['contexts']}
            selection=old._read_json(root/'selection_report.json');assert selection['ok']
            assert {r['context_id'] for r in selection['contexts']}==set(scenes)
            for g in groups.values():
                c=scenes[g['context_id']];sweep=g['anchor_sweep'];assert sweep['valid']
                corners=np.array([p for s in seqs[g['group_id']] for f in frames[s['sequence_id']] for p in f['target_bbox_corners_world']])
                bbox=aabb_corners(corners.min(0),corners.max(0))
                assert corners_inside_room(bbox,c['room_dimensions'],config['selection']['wall_margin_m'])
                assert validate_no_collision(bbox,c['static_objects'],config['selection']['static_clearance_m'])
                selected=g['camera_selection'];assert selected['valid']
                assert selected['in_image_fraction']>=config['selection']['min_projected_point_in_image_fraction']
                assert selected['visible_surface_fraction']>=config['selection']['min_visible_surface_fraction']
                assert selected['bbox_margin_px']>=config['selection']['edge_margin_px']
                assert config['selection']['min_initial_mask_area_ratio']<=selected['initial_projected_bbox_area_ratio']<=config['selection']['max_initial_mask_area_ratio']
                evidence=next(r for r in selection['contexts'] if r['context_id']==g['context_id'])
                assert evidence['selected']==dict(anchor_id=g['object_anchor_id'],base_camera_id=g['base_camera_id'])
                candidate=next(r for r in evidence['candidates'] if r['anchor_id']==g['object_anchor_id'])
                assert set(candidate['sweeps'])==set(c['motion_family_ids'])
                assert all(sw['valid'] for sw in candidate['sweeps'].values())
                assert candidate['sweeps'][g['motion_family_id']]==g['anchor_sweep']
                for s in seqs[g['group_id']]:
                    with np.load(root/s['tracks_path']) as tr:
                        assert np.min(tr['in_image'].mean(1))>=config['selection']['min_projected_point_in_image_fraction']
                        assert np.min(tr['visible'].mean(1))>=config['selection']['min_visible_surface_fraction']
        run('sweep_room_collision_visibility',sweeps)
        # V1 helpers below are independent of axis and source reconstruction.
        def provenance():
            p=old._read_json(root/'provenance.json')
            assert p['scene_semantics']=='novel explicit target/background scenes; no source-output reconstruction'
            assert p['seed']==config['seed'] and p['blender_version']
            assert len(p['config_sha256'])==len(p['base_config_sha256'])==64
            assert p['generator_source_sha256'] and all(len(v)==64 for v in p['generator_source_sha256'].values())
        run('novel_scene_provenance',provenance)
        protocol=old._config_protocol(dict(config,motion=dict(levels=[-2,-1,0,1,2],axis_world=[1,0,0],delta_m=.04)),state)
        run('track_depth_reprojection',lambda:old.validate_tracks(root,m,{s['sequence_id']:s for s in m['sequences']},frames,protocol,geometry_only,state))
        run('common_visible_counterfactual_pairs',lambda:old.validate_counterfactual_pairs(seqs,{s['sequence_id']:root/s['tracks_path'] for s in m['sequences']},protocol,state))
        run('rendered_outputs',lambda:old.validate_rendered_outputs(root,config,m,frames,{},protocol,geometry_only,state))
        def raster_identity():
            from memory_scene_blender.ego_object_v2.rendering import axial_to_range, DEPTH_CONVENTION
            for ss in seqs.values():
                assert len({old._sha256_file(root/frames[s['sequence_id']][0]['rgb']) for s in ss})==1
            for f in m['frames']:
                assert all(f['render_convention'].get(k)==v for k,v in DEPTH_CONVENTION.items())
                native=np.load(root/f['native_depth_npy'],allow_pickle=False)
                depth=np.load(root/f['depth_npy'],allow_pickle=False)
                assert native.shape==depth.shape==(512,512) and np.isfinite(native).all() and (native>0).all()
                assert np.array_equal(depth,axial_to_range(native,f['K']))
                mask=old._load_mask(root/f['target_mask'])
                edge=np.zeros_like(mask);edge[:3]=True;edge[-3:]=True;edge[:,:3]=True;edge[:,-3:]=True
                assert not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
                assert (mask&edge).sum()/max(1,mask.sum())<=config['validation']['max_target_edge_touch_ratio']
        if geometry_only:
            checks['rendered_outputs']={'status':'skipped','reason':'geometry-only'}
            checks['frame0_rgb_and_actual_mask_edges']={'status':'skipped','reason':'geometry-only'}
        else: run('frame0_rgb_and_actual_mask_edges',raster_identity)
    for name,n in state.error_counts.items():
        if n: checks[name]={'status':'failed','error_count':n}
    result=dict(ok=not state.errors,geometry_only=geometry_only,checks=checks,metrics=state.metrics,
                errors=state.errors,warnings=state.warnings,error_counts=dict(state.error_counts))
    write_json(root/'validation_summary.json',result)
    return result
