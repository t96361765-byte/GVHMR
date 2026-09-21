"""Create a self-contained corrected review blend and FBX (run inside Blender)."""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

BODY_NAMES=['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
            'left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar',
            'right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']


def material(name,color):
    mat=bpy.data.materials.new(name);mat.diffuse_color=(*color,1);mat.use_nodes=True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*color,1)
    mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.7
    return mat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--addon',required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=Path(args.input).resolve()
    cfg=json.loads((folder/'config.json').read_text());data=np.load(folder/'diagnostics.npz');ks,ke=cfg['keep_range']
    sys.path.insert(0,str(Path(args.addon).resolve().parent))
    import smplx_blender_addon
    smplx_blender_addon.register()
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    result=bpy.ops.object.smplx_add_animation(filepath=str(folder/'smplx_neutral_trimmed.npz'),anim_format='SMPL-X',
        rest_position='SMPL-X',hand_reference='FLAT',target_framerate=int(cfg['fps']),keyframe_corrective_pose_weights=True)
    if 'FINISHED' not in result:raise RuntimeError('SMPL-X import failed')
    body=bpy.context.object;armature=body.parent;body.name='Corrected_SMPLX';armature.name='Corrected_Armature'
    body.data.materials.clear();body.data.materials.append(material('Body_blue',(.36,.62,.72)))
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=ke-ks;scene.render.fps=int(cfg['fps'])
    scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    errors=[]
    for frame in np.linspace(1,ke-ks,12).astype(int):
        scene.frame_set(int(frame));bpy.context.view_layer.update()
        positions=np.array([tuple(armature.matrix_world@armature.pose.bones[n].head) for n in BODY_NAMES])
        errors.append(float(np.linalg.norm(positions-data['corrected_joints_zup'][ks+frame-1],axis=-1).max()))
    if max(errors)>.003:raise RuntimeError(f'Blender/FK mismatch: {max(errors):.6f} m')
    bpy.ops.wm.obj_import(filepath=str(folder/'mushroom_proxy_zup.obj'),forward_axis='Y',up_axis='Z')
    apparatus=bpy.context.object;apparatus.name='Fitted_mushroom_proxy_Zup'
    apparatus.data.materials.clear();apparatus.data.materials.append(material('Mushroom_terracotta',(.58,.24,.18)))
    apparatus['description']='Approximate fitted video apparatus; not the exact geometry or scale of the source blend asset'
    bpy.ops.mesh.primitive_plane_add(size=6,location=(0,0,-.008));floor=bpy.context.object;floor.name='Ground'
    floor.data.materials.append(material('Ground_gray',(.34,.37,.40)))
    bpy.ops.object.camera_add(location=(2.7,-3.8,2.55));camera=bpy.context.object
    camera.rotation_euler=(Vector((0,0,.65))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.lens=52;scene.camera=camera
    for loc,energy,size in [((1,-3,5),850,5),((-3,1,3),600,4)]:
        bpy.ops.object.light_add(type='AREA',location=loc);light=bpy.context.object;light.data.energy=energy;light.data.shape='DISK';light.data.size=size
        light.rotation_euler=(Vector((0,0,.7))-light.location).to_track_quat('-Z','Y').to_euler()
    scene.world.color=(.3,.3,.3);scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1080;scene.render.resolution_y=1080;scene.render.resolution_percentage=100
    scene.frame_set(min(120,ke-ks));bpy.ops.object.select_all(action='DESELECT');body.select_set(True);bpy.context.view_layer.objects.active=body
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_location=(0,0,.7);area.spaces.active.region_3d.view_distance=3.8
                area.spaces.active.region_3d.view_rotation=camera.rotation_euler.to_quaternion();area.spaces.active.shading.color_type='MATERIAL'
    scene['source_frame_offset']=ks;scene['processing']='BVH-assisted offline refinement with measured background camera rotation'
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/f'{folder.name}_corrected.blend'))
    (folder/'blender_validation.json').write_text(json.dumps(dict(sampled_frames=12,max_joint_error_m=max(errors),fps=scene.render.fps,
        frame_start=1,frame_end=scene.frame_end,world_up='Z',source_frame_offset=ks),indent=2),encoding='utf-8')
    result=bpy.ops.object.smplx_export_fbx(filepath=str(folder/'smplx_neutral_trimmed.fbx'),export_shape_keys='SHAPE_POSECORRECTIVES',target_format='UNITY',animation_only=False)
    if 'FINISHED' not in result:raise RuntimeError('FBX export failed')
    print('MUSHROOM_BLENDER_EXPORT_OK',folder,flush=True)


if __name__=='__main__':main()
