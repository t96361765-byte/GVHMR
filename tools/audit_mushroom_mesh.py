"""Check full SMPL-X surfaces and render a fixed-view before/after comparison."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import cv2
from hmr4d.utils.body_model.smplx_lite import SmplxLite
from hmr4d.utils.mushroom_refine import Y_TO_Z, cycle_metrics
from hmr4d.utils.runtime_paths import ffmpeg_program


def apparatus_mesh(radius, top, dome):
    """Closed surface of the same capped cylinder used in the collision proxy."""
    n=96;theta=np.arange(n)*2*np.pi/n
    profile=[(0,0),(radius,0),(radius,top-dome)]
    profile += [(r,top-dome*(r/radius)**2) for r in np.linspace(radius,0,18)[1:]]
    v=np.array([[r*np.cos(t),r*np.sin(t),z] for r,z in profile for t in theta],dtype=np.float32)
    f=[]
    for i in range(len(profile)-1):
        for k in range(n):
            a=i*n+k;b=i*n+(k+1)%n;c=b+n;d=a+n
            f.extend([[a,b,c],[a,c,d]])
    return v,np.array(f,dtype=np.int64)


def audit(folder, render=True):
    folder=Path(folder);prov=json.loads((folder/'provenance.json').read_text());cfg=json.loads((folder/'config.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    arr=np.load(folder/'diagnostics.npz');src=torch.load(prov['source'],map_location='cpu',weights_only=True);dst=torch.load(folder/'hmr4d_results.pt',map_location='cpu',weights_only=True)
    assert hashlib.sha256(Path(prov['source']).read_bytes()).hexdigest()==prov['source_sha256'],'Source file changed'
    assert hashlib.sha256(Path(prov['bvh']).read_bytes()).hexdigest()==prov['bvh_sha256'],'Source BVH changed'
    assert torch.equal(src['smpl_params_global']['betas'],dst['smpl_params_global']['betas']),'Shape changed'
    assert torch.equal(dst['smpl_params_global']['body_pose'],dst['smpl_params_incam']['body_pose'])
    npz=np.load(folder/'smplx_neutral.npz')
    for key in ['body_pose','global_orient','transl','betas']:
        val=npz['betas_per_frame'] if key=='betas' else npz[key]
        assert np.array_equal(val,dst['smpl_params_global'][key].numpy()),f'NPZ mismatch: {key}'
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';model=SmplxLite().to(device).eval()
    def vertices(params):
        result=[]
        with torch.no_grad():
            for s in range(0,len(params['transl']),24):
                v=model(**{k:x[s:s+24].to(device) for k,x in params.items()})
                result.append(v.cpu().numpy())
        return np.concatenate(result)
    new=vertices(dst['smpl_params_global'])@Y_TO_Z.T
    old=vertices(src['smpl_params_global'])@Y_TO_Z.T
    cam=vertices(dst['smpl_params_incam'])
    transformed=np.einsum('tij,tvj->tvi',arr['camera_R_zup_to_camera'],new)+arr['camera_t'][:,None]
    transform_error=float(np.abs(transformed-cam).max())
    assert transform_error<5e-5,'Camera/world mesh inconsistency'
    legacy=cfg.get('evaluation_cycle_boundaries',arr['cycle_boundaries'].tolist())
    original_j=arr['original_joints_zup'];corrected_j=arr['corrected_joints_zup'];s,e=legacy[:2]
    registration=original_j[s:e,[20,21]].mean((0,1))-corrected_j[s:e,[20,21]].mean((0,1))
    old-=registration
    radius=metrics['apparatus']['radius_m'];top=metrics['apparatus']['top_m'];dome=metrics['apparatus']['dome_m'];ks,ke=cfg['keep_range']
    hand=(model.lbs_weights[:,[20,21]+list(range(25,55))].sum(-1).cpu().numpy()>=.35)
    def collision(v):
        v=v[ks:ke];r=np.linalg.norm(v[:,:,:2],axis=-1);cap=top-dome*(r/radius)**2
        depth=np.minimum(radius-r,np.minimum(cap-v[:,:,2],v[:,:,2]-.08))
        nonhand=np.maximum(depth[:,~hand],0);all_depth=np.maximum(depth,0)
        return dict(nonhand_max_depth_cm=float(nonhand.max()*100),
                    nonhand_p95_frame_max_depth_cm=float(np.percentile(nonhand.max(1),95)*100),
                    nonhand_vertex_fraction_over_1cm=float((nonhand>.01).mean()),
                    frames_with_nonhand_penetration_over_1cm=int((nonhand.max(1)>.01).sum()),
                    all_vertices_max_depth_cm=float(all_depth.max()*100),
                    ground_max_penetration_cm=float(max(0,-v[:,:,2].min())*100))
    def phase_stats(j):
        ss,se=cfg['circle_range'];f=j[:,[7,8]].mean(1)-j[:,[20,21]].mean(1);a=np.unwrap(np.arctan2(f[:,1],f[:,0]))
        return dict(turns_in_legacy_window=float((a[legacy[-1]]-a[legacy[0]])/(2*np.pi)),
                    pelvis_acceleration_p95_m_s2=float(np.percentile(np.linalg.norm(np.diff(j[ss:se,0],n=2,axis=0)*cfg['fps']**2,axis=-1),95)))
    report=dict(source_hashes_unchanged=True,source_shape_unchanged=True,npz_pt_exact_match=True,
        camera_world_full_mesh_max_coordinate_difference_m=transform_error,
        legacy_cycle_boundaries=legacy,legacy_original=cycle_metrics(original_j,legacy),legacy_corrected=cycle_metrics(corrected_j,legacy),
        full_mesh_proxy_collision_original=collision(old),full_mesh_proxy_collision_corrected=collision(new),
        phase_original=phase_stats(original_j),phase_corrected=phase_stats(corrected_j),
        comparison_registration_m=registration.tolist(),
        caveats=['The original motion is aligned once using the first cycle mean wrist position.',
                 'Collision depth is an axis-wise interior proxy against a fitted capped cylinder, not exact signed mesh distance.',
                 'Hand vertices are reported separately; fingers are unobserved SMPL-X mean hand poses.',
                 'Submillimetre cycle-mean residuals are fitted constraints, not real-world reconstruction accuracy.'])
    (folder/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    av,af=apparatus_mesh(radius,top,dome)
    with (folder/'mushroom_proxy_zup.obj').open('w') as stream:
        for v in av:stream.write('v '+' '.join(map(str,v))+'\n')
        for f in af:stream.write('f '+' '.join(str(i+1) for i in f)+'\n')
    if render:
        from hmr4d.utils.vis.renderer import Renderer
        class ReviewRenderer(Renderer):
            def create_renderer(self):
                super().create_renderer()
                self.renderer.rasterizer.raster_settings.max_faces_per_bin=50000
        width,height=640,720;fps=cfg['fps'];focal=850.
        renderer=ReviewRenderer(width,height,device=device,faces=np.concatenate([model.faces,af+len(new[0])]),focal_length=focal)
        eye=np.array([2.7,-3.8,2.55]);target=np.array([0.,0.,.65]);forward=target-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.stack([right,down,forward])
        colors=np.concatenate([np.tile([.58,.76,.88],(len(new[0]),1)),np.tile([.77,.48,.37],(len(av),1))]).astype(np.float32)
        colors=torch.from_numpy(colors[None]).to(device)
        cmd=[ffmpeg_program('ffmpeg'),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width*2}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p',str(folder/'world_comparison.mp4')]
        writer=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        samples=[];ids=np.linspace(ks,ke-1,8).astype(int).tolist()
        with torch.no_grad():
            for idx in range(ks,ke):
                panels=[]
                for verts,title in [(old[idx],'Original: one constant alignment'),(new[idx],'Refined: background camera + BVH prior')]:
                    combined=np.concatenate([verts,av]);c=(combined-eye)@R.T
                    im=renderer.render_mesh(torch.as_tensor(c,dtype=torch.float32,device=device),colors=colors)
                    cv2.rectangle(im,(0,0),(width,54),(245,245,245),-1)
                    cv2.putText(im,title,(12,22),0,.57,(30,30,30),1,cv2.LINE_AA)
                    cv2.putText(im,f'Source frame {idx}',(12,45),0,.5,(30,30,30),1,cv2.LINE_AA);panels.append(im)
                joined=np.hstack(panels).astype(np.uint8);writer.stdin.write(joined.tobytes())
                if idx in ids:samples.append(cv2.cvtColor(cv2.resize(joined,(640,360)),cv2.COLOR_RGB2BGR))
                if (idx-ks)%60==0:print('render',idx,flush=True)
        writer.stdin.close()
        if writer.wait()!=0:raise RuntimeError('Video encoding failed')
        cv2.imwrite(str(folder/'world_mesh_samples.jpg'),np.vstack([np.hstack(samples[i:i+2]) for i in range(0,len(samples),2)]))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--no-render',action='store_true')
    a=p.parse_args();audit(a.input,not a.no_render)
