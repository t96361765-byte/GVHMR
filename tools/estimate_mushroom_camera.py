"""Measure small camera rotations from static background tracks, independently of HMR."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def estimate(folder, roi, reference_frame, output):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pred=torch.load(folder/'hmr4d_results.pt',map_location='cpu',weights_only=True)
    K=pred['K_fullimg'][0].numpy();kinv=np.linalg.inv(K)
    boxes=torch.load(folder/'preprocess/bbx.pt',map_location='cpu',weights_only=True)['bbx_xyxy'].numpy()
    cap=cv2.VideoCapture(str(folder/'0_input_video.mp4'))
    frames=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    cap.release()
    ref=frames[reference_frame];mask=np.zeros_like(ref);x0,y0,x1,y1=roi;mask[y0:y1,x0:x1]=255
    for box in boxes:
        a,b,c,d=np.rint(box).astype(int);mask[max(0,b-20):d+20,max(0,a-20):c+20]=0
    points=cv2.goodFeaturesToTrack(ref,500,.01,8,mask=mask,blockSize=7)
    if points is None or len(points)<20:raise ValueError('Background ROI has fewer than 20 trackable features')
    def match(source, target, features):
        if features is None or len(features)<20:return None
        tracked,status,_=cv2.calcOpticalFlowPyrLK(source,target,features,None,winSize=(31,31),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
        if tracked is None:return None
        back,bs,_=cv2.calcOpticalFlowPyrLK(target,source,tracked,None,winSize=(31,31),maxLevel=4)
        if back is None:return None
        valid=status[:,0].astype(bool)&bs[:,0].astype(bool)&(np.linalg.norm(back[:,0]-features[:,0],axis=-1)<1)
        p=features[valid,0];q=tracked[valid,0]
        if len(p)<15:return None
        aff,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=2000)
        if aff is None or inliers is None or inliers.sum()<15:return None
        good=inliers[:,0].astype(bool);p=p[good];q=q[good]
        a=np.c_[p,np.ones(len(p))]@kinv.T;b=np.c_[q,np.ones(len(q))]@kinv.T
        a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
        u,s,vh=np.linalg.svd(b.T@a);r=u@np.diag([1,1,np.linalg.det(u@vh)])@vh
        projected=(np.c_[p,np.ones(len(p))]@kinv.T@r.T)@K.T;projected=projected[:,:2]/projected[:,2:]
        error=float(np.median(np.linalg.norm(projected-q,axis=-1)))
        if error>2.:return None
        return r,aff,len(p),error

    rotations=[];counts=[];errors=[];matrices=[];fallback_frames=[]
    consecutive_fallback=0
    for i,gray in enumerate(frames):
        result=match(ref,gray,points)
        fallback=result is None
        if fallback and i>0:
            # Replenish features only in the explicitly selected static background.
            fresh=cv2.goodFeaturesToTrack(frames[i-1],500,.01,8,mask=mask,blockSize=7)
            result=match(frames[i-1],gray,fresh)
        if result is None:raise ValueError(f'Insufficient reliable background tracks at frame {i}, including adjacent-frame recovery')
        r,aff,count,error=result
        if fallback:
            consecutive_fallback+=1
            if consecutive_fallback>30:
                raise ValueError(f'No reference-frame anchor for over 30 frames at {i}; choose another background/reference frame')
            r=r@rotations[-1]
            aff=(np.vstack([aff,[0,0,1]])@np.vstack([matrices[-1],[0,0,1]]))[:2]
            fallback_frames.append(i)
        else:consecutive_fallback=0
        rotations.append(r);counts.append(count);errors.append(error)
        matrices.append(aff)
    rvec=Rotation.from_matrix(np.array(rotations)).as_rotvec()
    smooth=gaussian_filter1d(rvec,.65,axis=0)
    matrices=np.array(matrices);rotation=Rotation.from_rotvec(smooth).as_matrix()
    np.savez_compressed(output/'camera_motion.npz',rotations=rotation,raw_rotations=np.array(rotations),affine=matrices,
                        inliers=np.array(counts),median_track_error_px=np.array(errors),reference_frame=reference_frame,roi=roi,
                        adjacent_recovery_frames=np.array(fallback_frames,dtype=int))
    report=dict(method='Reference-to-frame static-background LK tracks, similarity RANSAC, unit-ray SO(3) alignment',
                reference_frame=reference_frame,roi=roi,frames=len(frames),features=len(points),
                adjacent_recovery_frames=fallback_frames,
                recovery_note='Direct reference matching preferred; adjacent matches replenish features on failures, up to 30 consecutive frames. Recovery fit errors are local, not accumulated absolute errors.',
                minimum_inlier_count=int(min(counts)),median_rotation_fit_error_px=float(np.median(errors)),
                p95_rotation_fit_error_px=float(np.percentile(errors,95)),
                maximum_rotation_from_reference_deg=float(np.rad2deg(np.linalg.norm(smooth,axis=-1).max())),
                maximum_affine_translation_px=float(np.linalg.norm(matrices[:,:,2],axis=-1).max()),
                limitations='Small rotational jitter only. Translational parallax, rolling shutter, zoom and arbitrary moving cameras need a different estimator.')
    (output/'camera_motion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    vis=cv2.cvtColor(ref,cv2.COLOR_GRAY2BGR)
    for point in points[:,0]:cv2.circle(vis,tuple(np.rint(point).astype(int)),3,(0,255,0),-1)
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,0,255),2);cv2.imwrite(str(output/'background_tracks.jpg'),vis)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--roi',nargs=4,type=int,required=True);p.add_argument('--reference-frame',type=int,default=60)
    a=p.parse_args();estimate(a.input,a.roi,a.reference_frame,a.output)
