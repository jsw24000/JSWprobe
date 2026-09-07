#!/usr/bin/env python3
"""Compact V2 contact sheets, bounded to two groups by default."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PIL import Image, ImageDraw
from memory_scene_blender.object_translation.manifest_utils import read_jsonl, write_json


def render_preview(root, max_groups=2):
    if max_groups < 1:
        raise ValueError('max_groups must be positive')
    root=Path(root);out=root/'previews';out.mkdir(exist_ok=True)
    groups=read_jsonl(root/'manifests/groups.jsonl')[:max_groups]
    sequences=read_jsonl(root/'manifests/sequences.jsonl')
    frames={f['frame_id']:f for f in read_jsonl(root/'manifests/frames.jsonl')}
    result=[]
    for g in groups:
        dest=out/(g['group_id']+'.jpg')
        if dest.exists(): raise FileExistsError(dest)
        canvas=Image.new('RGB',(960,1120),'white');draw=ImageDraw.Draw(canvas)
        for i,key in enumerate(('context_id','target_id','background_id','motion_family_id')):
            draw.text((8,5+18*i),f'{key}: {g[key]}',fill='black')
        by_condition={(s['ego_level'],s['object_level']):s for s in sequences if s['group_id']==g['group_id']}
        for row,condition in enumerate(((0,0),(1,0),(0,-1),(1,1))):
            s=by_condition[condition]
            for col,idx in enumerate((0,4,7)):
                f=frames[s['frame_ids'][idx]]
                with Image.open(root/f['rgb']) as im:
                    im=im.convert('RGB');im.thumbnail((250,230));canvas.paste(im,(col*320,row*255+110))
                draw.text((col*320+5,row*255+88),f'e={condition[0]} o={condition[1]} frame={idx}',fill='black')
        canvas.save(dest)
        result.append(dict(group_id=g['group_id'],path=dest.relative_to(root).as_posix()))
    write_json(out/'preview_summary.json',dict(groups=result,frame_indices=[0,4,7],conditions=[[0,0],[1,0],[0,-1],[1,1]]))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--max-groups',type=int,default=2)
    a=p.parse_args();print(json.dumps(render_preview(a.dataset_root,a.max_groups),indent=2))
