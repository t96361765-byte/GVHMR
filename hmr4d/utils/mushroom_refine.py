"""Offline, fixed-camera mushroom refinement. Network weights are never changed.

The optimizer uses the source performer's shape and timing. An unpaired BVH
supplies periodic relative geometry and contact timing, not framewise targets.
All internal scene coordinates are metres, Z-up; exported SMPL-X stays Y-up.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from hmr4d.utils.body_model.smplx_lite import SmplxLite, SmplxLiteV437Coco17, batch_rigid_transform_v2

Y_TO_Z = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
BODY_MAP = {0: 'Hips', 1: 'LeftHip', 2: 'RightHip', 4: 'LeftKnee',
            5: 'RightKnee', 7: 'LeftAnkle', 8: 'RightAnkle',
            10: 'LeftToe', 11: 'RightToe', 12: 'Neck', 15: 'Head',
            16: 'LeftShoulder', 17: 'RightShoulder', 18: 'LeftElbow',
            19: 'RightElbow', 20: 'LeftWrist', 21: 'RightWrist'}


def read_bvh(path):
    """Preserve source hierarchy; honor declared intrinsic Euler channel order."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    mi = next(i for i, l in enumerate(lines) if l.strip() == 'MOTION')
    names, parents, offsets, channels, ends = [], [], [], [], {}
    active, in_end = -1, False
    for line in lines[:mi]:
        s = line.split()
        if not s:
            continue
        if s[0] in ('ROOT', 'JOINT'):
            names.append(s[1]); parents.append(active); offsets.append([0., 0, 0]); channels.append([])
            active = len(names) - 1
        elif s[:2] == ['End', 'Site']:
            in_end = True
        elif s[0] == 'OFFSET':
            if in_end:
                ends[names[active]] = list(map(float, s[1:]))
            else:
                offsets[active] = list(map(float, s[1:]))
        elif s[0] == 'CHANNELS':
            channels[active] = s[2:]
            if len(channels[active]) != int(s[1]):
                raise ValueError('Invalid BVH channel count')
        elif s[0] == '}':
            if in_end:
                in_end = False
            else:
                active = parents[active]
    n = int(lines[mi + 1].split(':')[1]); dt = float(lines[mi + 2].split(':')[1])
    values = np.fromstring(' '.join(lines[mi + 3:]), sep=' ')
    if dt <= 0 or len(values) != n * sum(map(len, channels)) or not np.isfinite(values).all():
        raise ValueError('Invalid BVH motion data')
    values = values.reshape(n, -1); positions = []; rotations = []; cursor = 0
    for j, ch in enumerate(channels):
        pos = np.tile(offsets[j], (n, 1)); angles, order = [], ''
        for c in ch:
            if c.endswith('position'):
                pos[:, 'XYZ'.index(c[0])] += values[:, cursor]
            elif c.endswith('rotation'):
                order += c[0]; angles.append(values[:, cursor])
            else:
                raise ValueError(f'Unsupported BVH channel {c}')
            cursor += 1
        rot = Rotation.from_euler(order, np.stack(angles, -1), degrees=True).as_matrix()
        if parents[j] >= 0:
            par = parents[j]
            pos = positions[par] + np.einsum('tij,tj->ti', rotations[par], pos)
            rot = rotations[par] @ rot
        positions.append(pos); rotations.append(rot)
    if not set(BODY_MAP.values()).issubset(names):
        raise ValueError('BVH is missing required named body joints')
    offsets = np.array(offsets)
    if offsets[names.index('Chest'), 2] <= 0 or not 0.1 < np.linalg.norm(offsets[names.index('LeftKnee')]) < 0.8:
        raise ValueError('This reference must be metre-valued and Z-up; convert other conventions explicitly')
    return dict(names=names, parents=parents, offsets=offsets, ends=ends,
                joints=np.stack(positions, 1), rotations=np.stack(rotations, 1), fps=1 / dt)


def phase_and_cuts(joints, start, end, wrists=(20, 21), ankles=(7, 8)):
    vector = joints[:, ankles].mean(1) - joints[:, wrists].mean(1)
    angle = np.unwrap(np.arctan2(vector[:, 1], vector[:, 0]))
    angle = gaussian_filter1d(angle, 0.6)
    direction = np.sign(np.median(np.diff(angle[start:end]))) or 1
    progress = direction * angle
    # Each boundary has the same apparatus-relative angle (positive X axis).
    levels = np.arange(np.ceil(progress[start] / (2*np.pi)), np.floor(progress[end-1] / (2*np.pi)) + 1) * 2*np.pi
    cuts = []
    for level in levels:
        hits = np.flatnonzero((progress[start:end-1] < level) & (progress[start+1:end] >= level)) + start + 1
        if len(hits):
            cuts.append(int(hits[0]))
    return angle, cuts, int(direction)


def fk(model, body_matrices, root_matrix, transl, betas):
    sk = model.get_skeleton(betas)
    full = torch.cat([root_matrix[:, None], body_matrices], 1)
    # Only the first 22 joints are needed for kinematic/contact losses.
    j, _ = batch_rigid_transform_v2(full, sk[:, :22], model.parents[:22])
    return j + transl[:, None]


def transform_camera(points, rotation, translation):
    if rotation.ndim == 3:
        return torch.einsum('tij,tnj->tni', rotation, points) + translation[:, None]
    return points @ rotation.T + translation


def project(points, rotation, translation, K):
    cam = transform_camera(points, rotation, translation)
    uv = cam[..., :2] / cam[..., 2:].clamp_min(.2)
    return uv * torch.stack([K[0, 0], K[1, 1]]) + K[:2, 2]


def periodic_reference(bvh, source_joints, cfg, phase):
    """Phase averaging repeats cycles, never stretches an entire BVH to the video."""
    names = bvh['names']; bj = bvh['joints']; fps = bvh['fps']
    wr = [names.index('LeftWrist'), names.index('RightWrist')]
    an = [names.index('LeftAnkle'), names.index('RightAnkle')]
    bs, be = cfg['bvh_circle_range']
    bphase, cuts, direction = phase_and_cuts(bj, bs, be, wr, an)
    if len(cuts) < 3:
        raise ValueError('BVH reference interval must contain at least two complete cycles')
    mapped = np.zeros((len(bj), 22, 3))
    for j, name in BODY_MAP.items():
        mapped[:, j] = bj[:, names.index(name)]
    ss, se = cfg['circle_range']
    leg = lambda x: np.linalg.norm(x[:, 1]-x[:, 4],axis=-1) + np.linalg.norm(x[:, 4]-x[:, 7],axis=-1)
    scale = np.median(leg(source_joints[ss:se])) / np.median(leg(mapped[bs:be]))
    center = mapped[cuts[0]:cuts[-1], [20,21]].mean((0,1))
    speed = np.linalg.norm(np.gradient(bj[:, wr], 1/fps, axis=0),axis=-1)
    contact = np.exp(-(speed / .45)**2)
    grid = np.linspace(0, 2*np.pi, 129)
    samples, contacts = [], []
    for s, e in zip(cuts[:-1],cuts[1:]):
        xp = direction * (bphase[s:e+1] - bphase[s])
        xp = np.maximum.accumulate(xp)
        # Preserve absolute horizontal phase, including the small crossing offset.
        yaw = -bphase[s]
        rz = Rotation.from_euler('z', yaw).as_matrix()
        rel = (mapped[s:e+1] - center) @ rz.T
        arr = np.stack([np.interp(grid, xp, col) for col in rel.reshape(len(rel),-1).T],-1).reshape(129,22,3)
        samples.append(arr)
        contacts.append(np.stack([np.interp(grid,xp,col) for col in contact[s:e+1].T],-1))
    template = np.mean(samples,0); ctemplate=np.mean(contacts,0)
    template[-1]=template[0]; ctemplate[-1]=ctemplate[0]
    query = np.mod(direction * phase, 2*np.pi)
    ref = np.stack([np.interp(query,grid,col) for col in template.reshape(129,-1).T],-1).reshape(len(phase),22,3)*scale
    contact=np.stack([np.interp(query,grid,col) for col in ctemplate.T],-1)
    # A reference can have two moving wrists briefly; ensure at least one soft support.
    contact /= np.maximum(contact.max(1,keepdims=True), .05)
    return ref, contact, dict(scale=float(scale), cycle_boundaries=cuts, direction=direction)


def cycle_metrics(joints, cuts):
    centers=np.array([joints[s:e,[20,21]].mean((0,1)) for s,e in zip(cuts[:-1],cuts[1:])])
    pelvis=np.array([joints[s:e,0].mean(0) for s,e in zip(cuts[:-1],cuts[1:])])
    return dict(wrist_centers_m=centers.tolist(), pelvis_centers_m=pelvis.tolist(),
                wrist_horizontal_first_last_cm=float(np.linalg.norm((centers[-1]-centers[0])[:2])*100),
                wrist_vertical_first_last_cm=float((centers[-1]-centers[0])[2]*100),
                pelvis_horizontal_first_last_cm=float(np.linalg.norm((pelvis[-1]-pelvis[0])[:2])*100))


def fit_camera(coco, kp, K, initial_rotation, initial_translation, frames):
    def residual(x):
        pts=coco[frames]@Rotation.from_rotvec(x[:3]).as_matrix().T+x[3:]
        uv=pts[...,:2]/np.maximum(pts[...,2:],.2)*K[[0,1],[0,1]]+K[:2,2]
        return ((uv[:,5:]-kp[frames,5:,:2])*np.sqrt(np.clip(kp[frames,5:,2:3],0,1))).ravel()
    result=least_squares(residual,np.r_[Rotation.from_matrix(initial_rotation).as_rotvec(),initial_translation],
                         loss='soft_l1',f_scale=15,max_nfev=100)
    return Rotation.from_rotvec(result.x[:3]).as_matrix(),result.x[3:]


def refine(prediction, kp, bvh, cfg, iterations=800, device='cuda', callback=None, camera_motion=None):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model=SmplxLiteV437Coco17().eval()
    # Dense surface coverage is needed near wrists/feet: the 437 preview vertices
    # miss narrow but deep penetrations. Retain the exact 132-vertex COCO regressor.
    dense=SmplxLite()
    extremity=dense.lbs_weights[:,[7,8,10,11,20,21]+list(range(25,55))].sum(-1)>.25
    vids=torch.unique(torch.cat([torch.arange(0,len(dense.v_template),int(cfg.get('surface_stride',3))),torch.where(extremity)[0]]))
    for name in ['v_template','shapedirs','lbs_weights']:
        setattr(model,name,torch.cat([getattr(model,name)[:132],getattr(dense,name)[vids]],0))
    model.posedirs=torch.cat([model.posedirs[:,:132],dense.posedirs[:,vids]],1)
    del dense
    model=model.to(device)
    def T(a): return torch.as_tensor(a,dtype=torch.float32,device=device)
    def N(a): return a.detach().cpu().numpy()
    pg={k:v.to(device) for k,v in prediction['smpl_params_global'].items()}
    pc={k:v.to(device) for k,v in prediction['smpl_params_incam'].items()}
    length=len(pg['body_pose']);fps=float(cfg.get('fps',30)); ss,se=cfg['circle_range']; ks,ke=cfg.get('keep_range',[0,length])
    if not 0<=ks<=ss<se<=ke<=length:
        raise ValueError('Invalid keep/circle frame ranges (zero-based, end exclusive)')
    B=T(Y_TO_Z);K=prediction['K_fullimg'][0].to(device)
    relative_camera=T(np.tile(np.eye(3),(length,1,1)) if camera_motion is None else camera_motion)
    if relative_camera.shape!=(length,3,3):raise ValueError('Background camera track length mismatch')
    rays=np.concatenate([kp[:,:,:2],np.ones((*kp.shape[:2],1))],-1)@np.linalg.inv(N(K)).T
    stabilized_rays=np.einsum('tji,tnj->tni',N(relative_camera),rays)
    stable_kp=kp.copy();stable_kp[:,:,:2]=stabilized_rays[:,:,:2]/stabilized_rays[:,:,2:]*N(K)[[0,1],[0,1]]+N(K)[:2,2]
    if not torch.allclose(prediction['K_fullimg'],prediction['K_fullimg'][:1].expand_as(prediction['K_fullimg'])):
        raise ValueError('Changing intrinsics are not supported by this fixed-camera refinement')
    beta=pg['betas']
    if not torch.allclose(beta,beta[:1].expand_as(beta),atol=1e-5):
        raise ValueError('Expected constant source shape')
    with torch.no_grad():
        source_v,source_c=model(**pg); cam_v,cam_c=model(**pc)
        body0=axis_angle_to_matrix(pg['body_pose'].reshape(length,21,3));root0=axis_angle_to_matrix(pg['global_orient'])
        j0=fk(model,body0,root0,pg['transl'],beta)@B.T
        source_c=source_c@B.T;source_v=source_v@B.T
    jn=N(j0);phase,cuts,direction=phase_and_cuts(jn,ss,se)
    if len(cuts)<3:
        raise ValueError('Circle range needs at least two complete revolutions for drift estimation')
    ref,contact,ref_info=periodic_reference(bvh,jn,cfg,phase)
    if direction!=ref_info['direction']:
        raise ValueError('Video and BVH turn in opposite directions; an explicit left/right mirror is needed')
    centers=np.array([jn[s:e,[20,21]].mean((0,1)) for s,e in zip(cuts[:-1],cuts[1:])])
    times=np.array([(s+e-1)/2 for s,e in zip(cuts[:-1],cuts[1:])])
    center=centers.mean(0)
    drift=np.stack([np.interp(np.arange(length),times,centers[:,i]) for i in range(3)],-1)-center
    # Normalize apparatus horizontal origin once, never align cycles independently.
    support_height=float(cfg.get('initial_wrist_height_m',.70))
    shift=np.array([center[0],center[1],center[2]-support_height])
    initial_j=jn-drift[:,None]-shift
    initial_c=N(source_c)-drift[:,None]-shift
    p0=initial_j[:,0]
    Rmean=Rotation.from_matrix(N(axis_angle_to_matrix(pc['global_orient'])@root0.transpose(-1,-2)@B.T)[ks:ke]).mean().as_matrix()
    tmean=(N(cam_c)[ks:ke]-initial_c[ks:ke]@Rmean.T).mean((0,1))
    rcam,tcam=fit_camera(initial_c,stable_kp,N(K),Rmean,tmean,np.arange(ks,ke))
    pelvis=torch.nn.Parameter(T(p0));cam_rot=torch.nn.Parameter(T(Rotation.from_matrix(rcam).as_rotvec()));cam_t=torch.nn.Parameter(T(tcam))
    root_delta=torch.nn.Parameter(torch.zeros(length,3,device=device));body_delta=torch.nn.Parameter(torch.zeros(length,21,3,device=device))
    # Apparatus scale is inferred in the source SMPL-X scale, not claimed metric calibration.
    radius=torch.nn.Parameter(T(float(cfg.get('initial_radius_m',.37))))
    top=torch.nn.Parameter(T(float(cfg.get('initial_top_m',.64))))
    wrist_offset=float(cfg.get('wrist_surface_offset_m',.055))
    ref += np.array([0,0,support_height])
    if cfg.get('apparatus_pixels'):
        # Visible hand position and speed override the unpaired BVH support timing.
        ap=np.asarray(cfg['apparatus_pixels']);wr=stable_kp[:,[9,10],:2]
        xmin,xmax=ap[1,0]-25,ap[2,0]+25;ymin,ymax=ap[0,1]-35,max(ap[1:3,1])+20
        distance=np.maximum.reduce([xmin-wr[:,:,0],wr[:,:,0]-xmax,ymin-wr[:,:,1],wr[:,:,1]-ymax,np.zeros(wr.shape[:2])])
        gate=np.exp(-(distance/20)**2)
        velocity=np.linalg.norm(np.gradient(gaussian_filter1d(wr,1,axis=0),axis=0)*fps,axis=-1)
        observed=gate*np.exp(-(velocity/160)**2)
        contact=(.25*contact+.75*observed)*gate
        contact[ss:se]/=np.maximum(contact[ss:se].max(1,keepdims=True),.15)
    contact[:ss]=0;contact[se:]=0
    for start,end,side in cfg.get('extra_hand_contacts',[]):
        contact[start:end,side]=1
    contact=T(contact);ref=T(ref);phase_t=T(phase)
    conf=np.clip(kp[:,:,2],0,1)**2
    conf[:ks]=0;conf[ke:]=0;conf[:,:5]*=.25
    for start,end,ids in cfg.get('occluded_keypoints',[]):
        conf[start:end,ids]*=.12
    conf=T(conf);kp_t=T(kp[:,:,:2]);circle=torch.zeros(length,device=device);circle[ss:se]=1
    ground_mask=torch.zeros(length,2,device=device)
    for start,end,side in cfg.get('ground_contacts',[]): ground_mask[start:end,side]=1
    foot_ids=[10,11]
    sk=model.get_skeleton(beta);rest_pelvis=sk[:,0]
    parents=model.parents[:22].tolist()
    # Keep the BVH prior on limb directions weak, with source limb lengths unchanged.
    edges=[(1,4),(4,7),(2,5),(5,8),(16,18),(18,20),(17,19),(19,21),(0,12)]
    refdir=torch.stack([torch.nn.functional.normalize(ref[:,b]-ref[:,a],dim=-1) for a,b in edges],1)
    # Sparse skinning for reliable COCO joints and surface collision proxies.
    with torch.no_grad():
        verts0=source_v-j0[:,:1];coco0=source_c-j0[:,:1];jrel0=j0-j0[:,:1]
    history=[];stage1=min(200,max(60,iterations//4))
    optimizer=torch.optim.Adam([{'params':[pelvis],'lr':.008},{'params':[cam_rot,cam_t],'lr':.002},
        {'params':[radius,top],'lr':.001},{'params':[root_delta,body_delta],'lr':.0025}])
    pixel_annotations=cfg.get('apparatus_pixels')
    pose_mask=torch.ones(21,3,device=device)
    pose_mask[[9,10,11,14]]=.4  # feet and head retain stronger preservation
    pose_sigma=torch.full((21,1),.20,device=device)
    pose_sigma[[19,20]]=.8  # wrist orientation is weakly observed; avoid twisting elbows to fix palms
    hand_weights=model.lbs_weights[132:,[20,21]+list(range(25,55))].sum(-1)
    collision_vertex_mask=hand_weights<.35
    base_lrs=[group['lr'] for group in optimizer.param_groups]
    translation_stage=None
    for step in range(iterations):
        full=step>=stage1
        decay=1.0 if step<iterations*.65 else max(.15,1-(step-iterations*.65)/(iterations*.4))
        for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr*decay
        optimizer.zero_grad(set_to_none=True)
        Rd=axis_angle_to_matrix(root_delta if full else root_delta.detach())
        Rz=Rd@(B@root0)
        body=axis_angle_to_matrix(body_delta*pose_mask if full else body_delta.detach())@body0
        Ry=B.T@Rz
        translation_y=pelvis@B-rest_pelvis
        if full:
            # r6d avoids an unnecessary matrix->axis-angle round trip in the loss.
            v,c=super(SmplxLiteV437Coco17,model).forward(
                matrix_to_rotation_6d(body).flatten(1),beta,matrix_to_rotation_6d(Ry),translation_y,rotation_type='r6d'),None
            c=torch.einsum('vj,tvc->tjc',model.smplx2coco17_interestd,v[:,:132])@B.T
            v=v[:,132:]@B.T
            j=fk(model,body,Ry,translation_y,beta)@B.T
        else:
            j=jrel0+pelvis[:,None];c=coco0+pelvis[:,None];v=verts0+pelvis[:,None]
        camera=axis_angle_to_matrix(cam_rot)
        cameras=relative_camera@camera
        camera_trans=torch.einsum('tij,j->ti',relative_camera,cam_t)
        uv=project(c,cameras,camera_trans,K)
        # Robust pixel residual, expressed in a reproducible pixel scale.
        r2=((uv-kp_t)/12).square().sum(-1)
        image_loss=((torch.sqrt(1+r2)-1)*conf).sum()/conf.sum()
        wrist=j[:,[20,21]];wr_r=wrist[:,:,:2].norm(dim=-1)
        dome=.21*radius
        surface=top-dome*(wr_r/radius).square().clamp(max=1.5)
        height_loss=(((wrist[:,:,2]-surface-wrist_offset)/.025).square()*contact).sum()/contact.sum().clamp_min(1)
        radial_loss=((torch.relu(wr_r-radius*.88)/.025).square()*contact).sum()/contact.sum().clamp_min(1)
        vel=(wrist[1:]-wrist[:-1])*fps
        cm=contact[1:]*contact[:-1]
        slip_loss=((vel/.25).square().sum(-1)*cm).sum()/cm.sum().clamp_min(1)
        cycle_centers=torch.stack([wrist[s:e].mean((0,1)) for s,e in zip(cuts[:-1],cuts[1:])])
        # Allow natural hand exchange and cycle variation; do not force each frame to the axis.
        center_loss=((cycle_centers[:,:2]-cycle_centers[:,:2].mean(0))/.015).square().mean()
        pelvis_centers=torch.stack([j[s:e,0].mean(0) for s,e in zip(cuts[:-1],cuts[1:])])
        pelvis_cycle_loss=((pelvis_centers[:,:2]-pelvis_centers[:,:2].mean(0))/.03).square().mean()
        axis_loss=(cycle_centers[:,:2].mean(0)/.08).square().mean()
        feet=j[:,foot_ids];ground_loss=(((feet[:,:,2]-.045)/.025).square()*ground_mask).sum()/ground_mask.sum().clamp_min(1)
        # Floor and capped-body proxy are soft; exact mesh penetration is audited separately.
        floor_loss=(torch.relu(-v[ks:ke,:,2]-.003)/.015).square().topk(20,dim=1).values.mean()
        vr=v[:,:,:2].norm(dim=-1);vtop=top-dome*(vr/radius).square()
        depth=torch.minimum(radius-vr,torch.minimum(vtop-v[:,:,2],v[:,:,2]-.08))
        penetration=(torch.relu(depth[:,collision_vertex_mask]-.005)/.015).square()
        collision_loss=penetration[ks:ke].topk(min(20,penetration.shape[1]),dim=1).values.mean()
        hand_penetration=(torch.relu(depth[:,~collision_vertex_mask]-.007)/.015).square()
        hand_collision_loss=hand_penetration[ks:ke].topk(20,dim=1).values.mean() if full else v.new_zeros(())
        footvec=j[:,[7,8]].mean(1)-wrist.mean(1)
        footdir=torch.nn.functional.normalize(footvec[:,:2],dim=-1)
        phaseref=torch.stack([torch.cos(phase_t),torch.sin(phase_t)],-1)
        phase_loss=(((footdir-phaseref)/.15).square().sum(-1)*circle).sum()/circle.sum()
        direction=torch.stack([torch.nn.functional.normalize(j[:,b]-j[:,a],dim=-1) for a,b in edges],1)
        bvh_loss=(((direction-refdir)/.5).square().sum(-1)*circle[:,None]).sum()/(circle.sum()*len(edges))
        # Preserve performer-specific articulation with small, smooth SO(3) corrections.
        pose_loss=(body_delta/pose_sigma).square().mean();root_loss=(root_delta/.24).square().mean()
        pos_delta=pelvis-T(p0)
        smooth=((pos_delta[2:]-2*pos_delta[1:-1]+pos_delta[:-2])/.008).square().mean()
        rot_smooth=((root_delta[2:]-2*root_delta[1:-1]+root_delta[:-2])/.025).square().mean()
        pose_smooth=((body_delta[2:]-2*body_delta[1:-1]+body_delta[:-2])/.025).square().mean()
        position_prior=(pos_delta/.35).square().mean()
        apparatus_prior=((radius-float(cfg.get('initial_radius_m',.37)))/.1).square()+((top-float(cfg.get('initial_top_m',.64)))/.15).square()
        apparatus_loss=pelvis.new_zeros(())
        if pixel_annotations:
            # Fixed apparatus centre-top, left/right cap silhouette, and base-front.
            viewx=camera[0,:2];viewx=viewx/viewx.norm()
            front=-camera[2,:2];front=front/front.norm()
            a=torch.cat([pelvis.new_zeros(2),top[None]])
            left=torch.cat([-viewx*radius,(top-dome)[None]])
            right=torch.cat([viewx*radius,(top-dome)[None]])
            base=torch.cat([front*radius*.94,pelvis.new_zeros(1)])
            ap=project(torch.stack([a,left,right,base]),camera,cam_t,K)
            apparatus_loss=((ap-T(pixel_annotations))/10).square().mean()
        loss=(image_loss+1.0*height_loss+.6*radial_loss+.07*slip_loss+2*center_loss+.1*axis_loss+.25*pelvis_cycle_loss
              +.8*ground_loss+.5*floor_loss+.8*collision_loss+.35*hand_collision_loss+.5*phase_loss+.08*bvh_loss
              +.22*pose_loss+.12*root_loss+.12*smooth+.08*rot_smooth+.08*pose_smooth
              +.04*position_prior+.03*apparatus_prior+.6*apparatus_loss)
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite refinement loss at iteration {step}')
        if step==stage1-1:
            translation_stage=dict(joints=N(j),coco_incam=N(transform_camera(c,cameras,camera_trans)),
                                   metrics=cycle_metrics(N(j),cuts))
        loss.backward();torch.nn.utils.clip_grad_norm_([pelvis,cam_rot,cam_t,root_delta,body_delta,radius,top],10)
        optimizer.step()
        with torch.no_grad(): radius.clamp_(.2,.65);top.clamp_(.35,1.0)
        if step%50==0 or step==iterations-1:
            scalar=lambda x:float(x.detach())
            row=dict(iteration=step,stage='pose' if full else 'translation',loss=scalar(loss),image=scalar(image_loss),
                     contact=scalar(height_loss),slip=scalar(slip_loss),center=scalar(center_loss),phase=scalar(phase_loss),
                     collision=scalar(collision_loss),radius=scalar(radius),top=scalar(top))
            history.append(row)
            if callback: callback(row)
    with torch.no_grad():
        Ry=B.T@axis_angle_to_matrix(root_delta)@B@root0
        body=axis_angle_to_matrix(body_delta*pose_mask)@body0
        params=dict(body_pose=matrix_to_axis_angle(body).flatten(1),global_orient=matrix_to_axis_angle(Ry),
                    transl=pelvis@B-rest_pelvis,betas=beta)
        v,c=model(**params);j=fk(model,body,Ry,params['transl'],beta)@B.T
        camera=axis_angle_to_matrix(cam_rot)
        cameras=relative_camera@camera
        camera_trans=torch.einsum('tij,j->ti',relative_camera,cam_t)
        # Transform the pelvis, not transl alone: SMPL-X rotates around its rest pelvis.
        cam_pelvis=transform_camera(pelvis[:,None],cameras,camera_trans)[:,0]
        incam=dict(body_pose=params['body_pose'],global_orient=matrix_to_axis_angle(cameras@B@Ry),
                   transl=cam_pelvis-rest_pelvis,betas=beta)
        vv,cc=model(**incam)
        transformed=transform_camera(v@B.T,cameras,camera_trans)
        consistency=float((transformed-vv).abs().max())
        arrays=dict(original_joints_zup=jn,corrected_joints_zup=N(j),original_coco_zup=N(source_c),
                    corrected_coco_zup=N(c@B.T),original_coco_incam=N(cam_c),corrected_coco_incam=N(cc),
                    corrected_surface_zup=N(v@B.T),original_surface_zup=N(source_v),surface_vertex_ids=vids.numpy(),
                    K=N(K),keypoints=kp,phase=phase,cycle_boundaries=np.array(cuts),contact=N(contact),
                    reference_joints_zup=N(ref),camera_R_zup_to_camera=N(cameras),camera_t=N(camera_trans),
                    base_camera_R_zup_to_camera=N(camera),base_camera_t=N(cam_t),
                    original_to_initial_shift=shift,root_delta=N(root_delta),body_delta=N(body_delta*pose_mask))
        def pixerr(q):
            uv=q[...,:2]/q[...,2:]*N(K)[[0,1],[0,1]]+N(K)[:2,2]
            error=np.linalg.norm(uv-kp[:,:,:2],axis=-1);weight=N(conf);weight[:,:5]=0
            return dict(weighted_mean_px=float((error*weight).sum()/weight.sum()),
                        median_confident_px=float(np.median(error[weight>.25])),p95_confident_px=float(np.percentile(error[weight>.25],95)))
        metrics=dict(original=cycle_metrics(jn,cuts),corrected=cycle_metrics(N(j),cuts),
            original_incam_reprojection=pixerr(N(cam_c)),corrected_reprojection=pixerr(N(cc)),
            camera_world_max_vertex_difference_m=consistency,
            root_correction_degrees=dict(median=float(np.median(np.linalg.norm(N(root_delta)[ks:ke],axis=-1))*180/np.pi),max=float(np.linalg.norm(N(root_delta)[ks:ke],axis=-1).max()*180/np.pi)),
            body_correction_degrees=dict(median=float(np.median(np.linalg.norm(N(body_delta*pose_mask)[ks:ke],axis=-1))*180/np.pi),max=float(np.linalg.norm(N(body_delta*pose_mask)[ks:ke],axis=-1).max()*180/np.pi)),
            apparatus=dict(center_xy_m=[0,0],radius_m=float(radius),top_m=float(top),dome_m=float(.21*radius),
                           calibration='Approximate fit in the original SMPL-X scale; no measured physical scale'),
            phase=dict(cuts=cuts,source_turns=float((phase[cuts[-1]]-phase[cuts[0]])/(2*np.pi)),bvh=ref_info),
            history=history)
        if translation_stage:
            arrays['translation_only_joints_zup']=translation_stage['joints']
            metrics['translation_only']=translation_stage['metrics']
            metrics['translation_only_reprojection']=pixerr(translation_stage['coco_incam'])
        metrics['camera_mode']='background_rotation' if camera_motion is not None else 'fixed'
    result={key:{k:v.detach().cpu() for k,v in value.items()} for key,value in [('smpl_params_global',params),('smpl_params_incam',incam)]}
    result['K_fullimg']=prediction['K_fullimg'].clone()
    # Omit stale net_outputs: they describe the pre-refinement motion, not this result.
    return result,arrays,metrics
