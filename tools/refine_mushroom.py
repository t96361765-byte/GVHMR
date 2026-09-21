"""Refine existing fixed-camera GVHMR results using an unpaired mushroom BVH."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hmr4d.utils.mushroom_refine import read_bvh, refine
from hmr4d.utils.export_smplx import save_smplx_animation, export_fbx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='Existing result folder or hmr4d_results.pt')
    parser.add_argument('--bvh',required=True)
    parser.add_argument('--config',required=True,help='Per-video stage and apparatus observations JSON')
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--iterations',type=int,default=800)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--export-fbx',action='store_true')
    parser.add_argument('--blender')
    parser.add_argument('--camera-motion',help='camera_motion.npz from independent background tracks')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    source=Path(args.input).resolve()
    if source.is_dir(): source=source/'hmr4d_results.pt'
    folder=source.parent;output=Path(args.output_root).resolve()/folder.name
    if output==folder or folder in output.parents:
        raise ValueError('Use a separate output root to preserve the original reconstruction')
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{output} is nonempty; choose another root or explicitly --overwrite')
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    manifest_path=folder/'input_manifest.json'
    if manifest_path.exists() and not json.loads(manifest_path.read_text()).get('static_cam',False) and not args.camera_motion:
        raise ValueError('This optimizer requires a fixed-camera source video')
    pred=torch.load(source,map_location='cpu',weights_only=True)
    kp=torch.load(folder/'preprocess/vitpose.pt',map_location='cpu',weights_only=True).numpy()
    if len(kp)!=len(pred['smpl_params_global']['body_pose']): raise ValueError('2D keypoint length mismatch')
    bvh=read_bvh(args.bvh)
    output.mkdir(parents=True,exist_ok=True)
    callback=lambda row:print(json.dumps(row),flush=True)
    camera_motion=np.load(args.camera_motion)['rotations'] if args.camera_motion else None
    result,arrays,metrics=refine(pred,kp,bvh,cfg,args.iterations,args.device,callback,camera_motion)
    if not np.isfinite(arrays['corrected_joints_zup']).all(): raise ValueError('Invalid corrected result')
    torch.save(result,output/'hmr4d_results.pt')
    save_smplx_animation(result,output/'smplx_neutral.npz',fps=cfg.get('fps',30))
    start,end=cfg.get('keep_range',[0,len(kp)])
    trimmed={k:({p:v[start:end] for p,v in val.items()} if isinstance(val,dict) else val[start:end]) for k,val in result.items()}
    torch.save(trimmed,output/'hmr4d_results_trimmed.pt')
    save_smplx_animation(trimmed,output/'smplx_neutral_trimmed.npz',fps=cfg.get('fps',30))
    np.savez_compressed(output/'diagnostics.npz',**arrays)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    provenance=dict(source=str(source),bvh=str(Path(args.bvh).resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        camera_motion=str(Path(args.camera_motion).resolve()) if args.camera_motion else None,
        bvh_sha256=hashlib.sha256(Path(args.bvh).read_bytes()).hexdigest(),iterations=args.iterations,device=args.device,
        source_frame_range_of_trimmed=[start,end],source_video=str(folder/'0_input_video.mp4'),
        world_coordinates='Y-up metres; apparatus centred at XZ=(0,0); floor Y=0',
        limitations=['Unpaired BVH is a weak phase/contact prior, not ground truth.',
                     'Apparatus is an approximate geometric fit, not a calibrated physical scan.',
                     'Collision loss uses a capped-body proxy and 437 surface samples; not a full collision-free guarantee.'])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='history'},indent=2),flush=True)
    if args.export_fbx: export_fbx(output/'smplx_neutral_trimmed.npz',output/'smplx_neutral_trimmed.fbx',args.blender)


if __name__=='__main__': main()
