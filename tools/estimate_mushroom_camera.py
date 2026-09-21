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
    rotations=[];counts=[];errors=[];matrices=[]
    for i,gray in enumerate(frames):
        tracked,status,_=cv2.calcOpticalFlowPyrLK(ref,gray,points,None,winSize=(31,31),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
        back,bs,_=cv2.calcOpticalFlowPyrLK(gray,ref,tracked,None,winSize=(31,31),maxLevel=4)
        valid=status[:,0].astype(bool)&bs[:,0].astype(bool)&(np.linalg.norm(back[:,0]-points[:,0],axis=-1)<1)
        p=points[valid,0];q=tracked[valid,0]
        aff,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=2000)
        if aff is None or inliers.sum()<15:raise ValueError(f'Insufficient background tracks at frame {i}')
        good=inliers[:,0].astype(bool);p=p[good];q=q[good]
        a=np.c_[p,np.ones(len(p))]@kinv.T;b=np.c_[q,np.ones(len(q))]@kinv.T
        a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
        u,s,vh=np.linalg.svd(b.T@a);r=u@np.diag([1,1,np.linalg.det(u@vh)])@vh
        projected=(np.c_[p,np.ones(len(p))]@kinv.T@r.T)@K.T;projected=projected[:,:2]/projected[:,2:]
        rotations.append(r);counts.append(len(p));errors.append(np.median(np.linalg.norm(projected-q,axis=-1)))
        matrices.append(aff)
    rvec=Rotation.from_matrix(np.array(rotations)).as_rotvec()
    smooth=gaussian_filter1d(rvec,.65,axis=0)
    matrices=np.array(matrices);rotation=Rotation.from_rotvec(smooth).as_matrix()
    np.savez_compressed(output/'camera_motion.npz',rotations=rotation,raw_rotations=np.array(rotations),affine=matrices,
                        inliers=np.array(counts),median_track_error_px=np.array(errors),reference_frame=reference_frame,roi=roi)
    report=dict(method='Reference-to-frame static-background LK tracks, similarity RANSAC, unit-ray SO(3) alignment',
                reference_frame=reference_frame,roi=roi,frames=len(frames),features=len(points),
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
