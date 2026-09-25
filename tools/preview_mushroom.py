"""Create comparable fixed-view plots and source-video skeleton overlays."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COCO_EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
BODY_EDGES=[(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]


def preview(folder):
    folder=Path(folder);a=np.load(folder/'diagnostics.npz');m=json.loads((folder/'metrics.json').read_text());cfg=json.loads((folder/'config.json').read_text());prov=json.loads((folder/'provenance.json').read_text())
    cuts=cfg.get('evaluation_cycle_boundaries',a['cycle_boundaries']);old=a['original_joints_zup'];new=a['corrected_joints_zup'];colors=['#2563eb','#16a34a','#f59e0b','#db2777','#9333ea','#0891b2']
    fig,axs=plt.subplots(2,2,figsize=(10,9))
    for col,(j,title) in enumerate([(old,'Original GVHMR world'),(new,'Refined world')]):
        centers=np.array([j[s:e,[20,21]].mean((0,1)) for s,e in zip(cuts[:-1],cuts[1:])]);shift=centers[0].copy();shift[2]=0
        for k,(s,e) in enumerate(zip(cuts[:-1],cuts[1:])):
            # Include the shared boundary sample to avoid a plotting-only gap.
            p=j[s:e+1,[7,8]].mean(1)-shift
            axs[0,col].plot(p[:,0],p[:,1],color=colors[k%len(colors)],label=f'Cycle {k+1}')
        axs[0,col].scatter(centers[:,0]-shift[0],centers[:,1]-shift[1],marker='x',c='black')
        axs[0,col].set(title=title,xlim=(-1.35,1.35),ylim=(-1.5,1.2),xlabel='X (m)',ylabel='Y (m)');axs[0,col].set_aspect('equal');axs[0,col].grid(alpha=.2)
        for i,label in enumerate(['X','Y','Z']):axs[1,col].plot(np.arange(len(centers))+1,(centers[:,i]-centers[0,i])*100,'o-',label=label)
        axs[1,col].set(xlabel='Complete cycle',ylabel='Mean wrist-center displacement (cm)',ylim=(-55,45));axs[1,col].grid(alpha=.2);axs[1,col].legend()
    axs[0,0].legend();fig.suptitle('Ankle midpoint paths and cycle-mean wrist centers (not center of mass)\nOne constant display translation per sequence; no per-cycle alignment')
    fig.tight_layout();fig.savefig(folder/'trajectory_comparison.png',dpi=170);plt.close(fig)
    K=a['K'];oldc=a['original_coco_incam'];newc=a['corrected_coco_incam']
    def uv(q):return q[...,:2]/q[...,2:]*K[[0,1],[0,1]]+K[:2,2]
    olduv=uv(oldc);newuv=uv(newc)
    cap=cv2.VideoCapture(prov['source_video']);fps=cfg['fps'];frames=[];ks,ke=cfg['keep_range']
    sampleids=np.linspace(ks,ke-1,12).astype(int).tolist()
    width,height=640,720
    writer=cv2.VideoWriter(str(folder/'reprojection_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width*2,height))
    for idx in range(len(old)):
        ok,img=cap.read()
        if not ok:break
        if not ks<=idx<ke:continue
        panels=[]
        for pts,color,title in [(olduv[idx],(0,140,255),'Original camera prediction'),(newuv[idx],(40,220,60),'Refined: same world motion')]:
            im=img.copy()
            for s,e in COCO_EDGES:cv2.line(im,tuple(pts[s].astype(int)),tuple(pts[e].astype(int)),color,4,cv2.LINE_AA)
            for pt in pts[5:]:cv2.circle(im,tuple(pt.astype(int)),5,color,-1)
            scale=min(width/im.shape[1],height/im.shape[0])
            resized=cv2.resize(im,(max(1,round(im.shape[1]*scale)),max(1,round(im.shape[0]*scale))))
            im=np.zeros((height,width,3),dtype=np.uint8)
            y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
            im[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
            cv2.rectangle(im,(0,0),(width,42),(20,20,20),-1)
            cv2.putText(im,f'{title} | frame {idx}',(12,28),0,.65,(255,255,255),1,cv2.LINE_AA);panels.append(im)
        joined=np.hstack(panels);writer.write(joined)
        if idx in sampleids:frames.append(cv2.resize(joined,(640,360)))
    cap.release();writer.release()
    if len(frames)==12:cv2.imwrite(str(folder/'reprojection_samples.jpg'),np.vstack([np.hstack(frames[i:i+3]) for i in range(0,12,3)]))
    radius=m['apparatus']['radius_m'];top=m['apparatus']['top_m'];dome=m['apparatus']['dome_m']
    fig=plt.figure(figsize=(15,10))
    for k,idx in enumerate(sampleids):
        ax=fig.add_subplot(3,4,k+1,projection='3d');j=new[idx]
        theta=np.linspace(0,2*np.pi,40);r=np.linspace(0,radius,12);tt,rr=np.meshgrid(theta,r)
        ax.plot_surface(rr*np.cos(tt),rr*np.sin(tt),top-dome*(rr/radius)**2,color='#d5a878',alpha=.5)
        for s,e in BODY_EDGES:ax.plot(*j[[s,e]].T,color='#187843',lw=2)
        ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(0,1.65),title=f'Frame {idx}');ax.set_box_aspect((2.2,2.2,1.65));ax.view_init(25,-65)
    fig.tight_layout();fig.savefig(folder/'world_samples.png',dpi=130);plt.close(fig)
    print(folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);preview(p.parse_args().input)
