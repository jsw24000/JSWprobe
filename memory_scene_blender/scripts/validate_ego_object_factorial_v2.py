#!/usr/bin/env python3
"""Validate V2 manifests, geometry, physical identities and rendered evidence."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from memory_scene_blender.ego_object_v2.validation import validate_dataset
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--geometry-only',action='store_true')
    a=p.parse_args();r=validate_dataset(a.dataset_root,a.geometry_only)
    print(json.dumps(r,indent=2));sys.exit(0 if r['ok'] else 1)
