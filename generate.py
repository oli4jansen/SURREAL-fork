import glob
import math
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from os import getenv, remove, getcwd
from os.path import dirname, exists, join, realpath
from pickle import load
from random import choice
import time
import uuid

import numpy as np

import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam
from mathutils import Euler, Matrix, Quaternion, Vector

# Config

BG_PATH = 'data/backgrounds/'
TEXTURES_PATH = 'data/textures/'
MOTION_PATH = 'data/motion/'
SMPL_PATH = 'data/smpl/'
TMP_PATH = 'tmp/'
SHAPE_PATH = 'data/shape/shape_data.pkl'
SHADER_PATH = 'data/shader/shader.osl'
OUTPUT_PATH = 'output'

RENDER_WIDTH = 400
RENDER_HEIGHT = 300

################
#    Utils     #
################

start_time = time.time()


def log(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))


def load_motions():
    motions = []
    # List all pickle files in the directory
    pattern = os.path.join(MOTION_PATH, '*.pkl')
    file_paths = glob(pattern)
    names = list(map(lambda f: f.split(
        '/')[-1].replace('.pkl', ''), file_paths))
    # Loop through files
    for file_path in file_paths:
        # Open the file, load pickle
        with open(file_path, 'rb') as file:
            motion = pickle.load(file)
            motions.append(motion)

    return names, motions


def get_body_model(gender):
    return os.path.join(SMPL_PATH, 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0])


################
#   Blender    #
################

part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
              'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
              'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
              'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
              'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
              'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand', 'bone_23': 'R_Hand'}

smpl_bones = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
              'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']


def get_main_scene():
    return bpy.data.scenes['Scene']


def get_bone_name(gender, bone):
    return gender[0] + '_avg_' + bone


def get_armature_name(person):
    return 'Person_%s_Armature' % person


def get_mesh_name(person):
    return 'Person_%s_Mesh' % person


# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues


def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return (cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

# transformation between pose and blendshapes


def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return (mat_rots, bshapes)


def init_scene(motion, shapes):
    scene = get_main_scene()
    # Set render system settings
    scene.render.engine = 'CYCLES'
    scene.render.use_antialiasing = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    # Delete the default cube
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # Get a random background image and load into Blender
    background_paths = glob(os.path.join(BG_PATH, '*'))
    background_img = bpy.data.images.load(random.choice(background_paths))

    persons = {}

    for key in list(motion.keys()):
        # Add the motion to the person object
        persons[key] = {}
        persons[key]['motion'] = motion[key]

        log('Going to init person %s' % key)

        # Load the images picked at domain randomisation into Blender
        gender = random.choice(['female', 'male'])

        log(gender)

        # Get a random clothing texture and load into Blender
        texture_paths = glob(os.path.join(TEXTURES_PATH, gender, '*'))
        clothing_img = bpy.data.images.load(random.choice(texture_paths))

        # Load the SMPL model as base for the scene
        bpy.ops.import_scene.fbx(
            filepath=get_body_model(gender), global_scale=100, axis_forward='Y', axis_up='Z')

        # Get armature using the default name
        armature = bpy.data.objects.get('Armature')
        if armature is None:
            raise Exception('Could not retrieve newly created armature')

        armature.name = get_armature_name(key)

        mesh = [obj for obj in list(bpy.data.objects)
                if obj.type == 'MESH'][-1]
        mesh.name = get_mesh_name(key)
        # body_model_name = mesh.name
        # body_model = mesh

        # Autosmooth creates artifacts so turn it off
        mesh.data.use_auto_smooth = False
        # Clear existing animation data
        mesh.data.shape_keys.animation_data_clear()
        armature.animation_data_clear()

        n_sh_bshapes = len([k for k in mesh.data.shape_keys.key_blocks.keys()
                            if k.startswith('Shape')])

        # Get a body shape from CAESAR
        shape = random.choice(shapes['%sshapes' % gender][:, :n_sh_bshapes])

        # Create a material with the clothing texture to apply to the human body model
        material = create_material(key, clothing_img)
        # Assign the existing spherical harmonics material to the body model
        mesh.active_material = material

        # # unblocking both the pose and the blendshape limits
        for k in mesh.data.shape_keys.key_blocks.keys():
            if not k in bpy.data.shape_keys['Key'].key_blocks:
                print('%s is not in shape_keys' % k)
            else:
                bpy.data.shape_keys['Key'].key_blocks[k].slider_min = -10
                bpy.data.shape_keys['Key'].key_blocks[k].slider_max = 10

        deselect_all()
        mesh.select = True
        bpy.context.scene.objects.active = mesh

        pelvis = armature.pose.bones[get_bone_name(gender, 'Pelvis')]
        orig_pelvis_loc = (armature.matrix_world.copy() *
                           pelvis.head.copy()) - Vector((-1., 1., 1.))

        scene.objects.active = armature
        orig_trans = np.asarray(pelvis.location).copy()

        # armature.animation_data_clear()

        persons[key]['gender'] = gender
        persons[key]['shape'] = shape
        persons[key]['clothing_img'] = clothing_img
        persons[key]['material'] = material
        persons[key]['orig_pelvis_loc'] = orig_pelvis_loc
        persons[key]['orig_trans'] = orig_trans
        persons[key]['armature'] = armature
        persons[key]['mesh'] = mesh

    # Set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob

    # TODO: make camera somewhat random
    # You can get camera matrix from Blender Python console (bpy.data.objects['Camera'].matrix_world)
    matrix_world = Matrix(((-1.,  0.,  0.,  1.25),
                           (0., -0.2,  1.,  10),
                           (0., -1., -0.2, -2.5),
                           (0.,  0.,  0.,  1.)))
    # cam_ob.matrix_world = cam_ob.matrix_world * mathutils.Matrix.Translation((0.0, 1.0, 0.0))
    cam_ob.matrix_world = matrix_world

    cam_ob.data.angle = math.radians(30)
    cam_ob.data.lens = 50
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    cam_ob.animation_data_clear()

    orig_cam_loc = cam_ob.location.copy()

    # Allows the background image to come through
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_material_index = True

    # Set render size
    scn.render.resolution_x = RENDER_WIDTH
    scn.render.resolution_y = RENDER_HEIGHT
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    # Create the Composite Nodes graph, combining foreground (human bodies) with background image
    create_composite_nodes(background_img)

    return scene, persons, cam_ob, orig_cam_loc


def deselect_all():
    for ob in bpy.data.objects.values():
        ob.select = False
    bpy.context.scene.objects.active = None

# creation of the spherical harmonics material, using an OSL script


def create_material(person, clothing_img):
    deselect_all()
    material = bpy.data.materials.get('Material')
    if material is None:
        material = bpy.data.materials.new(
            name='Material_For_Person_%s' % person)
    else:
        material.name = 'Material_For_Person_%s' % person

    # Copy the base shader file to temporary directory
    tmp_shader_path = join(getcwd(), TMP_PATH, 'shader_%s.osl' % person)
    os.system('cp %s %s' % (SHADER_PATH, tmp_shader_path))

    # Clear the default nodes
    material.use_nodes = True
    for n in material.node_tree.nodes:
        material.node_tree.nodes.remove(n)

    # Build new tree with nodes that contain the texture image, script
    uv = material.node_tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = material.node_tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'

    uv_im = material.node_tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    uv_im.image = clothing_img

    rgb = material.node_tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    script = material.node_tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    # using the same file from multiple jobs causes white texture
    script.filepath = tmp_shader_path
    script.update()

    # The emission node makes it independent of the scene lighting
    emission = material.node_tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    # Link the nodes together
    material.node_tree.links.new(uv.outputs[2], uv_im.inputs[0])
    material.node_tree.links.new(uv_im.outputs[0], script.inputs[0])
    material.node_tree.links.new(script.outputs[0], emission.inputs[0])
    material.node_tree.links.new(emission.outputs[0], mat_out.inputs[0])

    # This has something to do with lighting
    material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (
        0, 0)

    return material


def create_composite_nodes(background_img):
    tree = get_main_scene().node_tree

    # Start with an empty tree
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create a node for the foreground
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400

    # Create a node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    bg_im.location = -300, 30
    bg_im.image = background_img

    # Create a node for mixing foreground and background
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.location = 40, 30
    mix.use_alpha = True

    # Create a node for the final output
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30

    # Merge foreground and background nodes
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs['Image'], mix.inputs[2])

    # bg+fg image
    tree.links.new(mix.outputs[0], composite_out.inputs[0])

# apply trans pose and shape to character


def apply_trans_pose_shape(trans, pose, person, frame=None):
    # Get the armature and mesh objects for this person
    armature = person['armature']
    mesh = person['mesh']
    gender = person['gender']
    shape = person['shape']

    # Set the location of the first bone to the translation parameter
    bone_name = get_bone_name(gender, 'Pelvis')
    armature.pose.bones[bone_name].location = trans
    if not frame is None:
        # Insert a keyframe
        bone_name = get_bone_name(gender, 'root')
        armature.pose.bones[bone_name].keyframe_insert('location', frame=frame)

    # Transform pose into rotation matrices (for pose) and pose blendshapes
    rotation_matrices, blendshapes = rodrigues2bshapes(pose)

    # Set the pose of each bone to the quaternion specified by pose
    for bone_index, mrot in enumerate(rotation_matrices):
        bone_name = get_bone_name(gender, smpl_bones[bone_index])
        bone = armature.pose.bones[bone_name]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if not frame is None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for i, blendshape in enumerate(blendshapes):
        key = 'Pose%03d' % i

        mesh.data.shape_keys.key_blocks[key].value = blendshape
        if not frame is None:
            mesh.data.shape_keys.key_blocks[key].keyframe_insert(
                'value', index=-1, frame=frame)

    # apply shape blendshapes
    for i, shape_elem in enumerate(shape):
        key = 'Shape%03d' % i
        mesh.data.shape_keys.key_blocks[key].value = shape_elem
        if not frame is None:
            mesh.data.shape_keys.key_blocks[key].keyframe_insert(
                'value', index=-1, frame=frame)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(scene, cam_ob, reg_ivs, joint_reg, person):

    orig_trans = person['orig_trans']
    shape = person['shape']
    mesh = person['mesh']
    armature = person['armature']

    scene.objects.active = armature
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    # empty array to hold vertices to regress from
    reg_vs = np.empty((len(reg_ivs), 3))
    # zero the pose and trans to obtain joint positions in zero pose

    apply_trans_pose_shape(orig_trans, np.zeros(72), person)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = mesh.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    armature.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    armature.hide = True
    for bone_index in range(24):
        bone_name = get_bone_name(person['gender'], smpl_bones[bone_index])
        bb = armature.data.edit_bones[bone_name]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[bone_index]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return shape


################
#     Main     #
################

def main():
    seed = str(start_time)
    random.seed(seed)
    log('Seed is %s' % seed)

    # Load the extracted motions from disk
    log('Loading motion files')
    motion_names, motions = load_motions()

    log('Loading CAESAR shape data')
    with open(SHAPE_PATH, 'rb') as file:
        shapes = pickle.load(file)
    joint_regressor = shapes['joint_regressor']
    regression_verts = shapes['regression_verts']

    # Loop through all motions
    for motion_index, (motion_name, motion) in enumerate(zip(motion_names, motions)):
        log('Processing motion %s' % motion_index)

        # Create a filename for the final render and use it to
        render_filename = '%s.mp4' % (motion_name + '_' + seed)
        render_temp_path = join(TMP_PATH, render_filename)
        log('Filename will be %s' % render_filename)

        # Initialise Blender
        log('Initialising Blender')
        # scene, body_model, body_model_name, arm_ob, cam_ob = init_scene(persons)
        scene, persons, cam_ob, orig_cam_loc = init_scene(motion, shapes)

        nr_frames = max([len(motion[person]['frame_ids'])
                         for person in motion.keys()])

        # TODO: figure out why this is needed
        for person in persons.values():
            curr_shape = reset_joint_positions(scene,
                                               cam_ob, regression_verts, joint_regressor, person)

        log('Did joint position reset for all persons')

        # LOOP TO CREATE 3D ANIMATION
        for frame in range(nr_frames):
            # Set a new keyframe
            scene.frame_set(frame)

            for person in persons.values():
                person_motion = person['motion']

                pred_cam = list(person_motion['pred_cam'][frame])

                # translation = Vector([pred_cam[0], pred_cam[1], pred_cam[2]])
                # translation = Vector([pred_cam[1], pred_cam[0], pred_cam[2]]) # Pretty good
                # translation = Vector([pred_cam[0], pred_cam[2], pred_cam[1]])
                # translation = Vector([pred_cam[1], pred_cam[2], pred_cam[0]])
                # translation = Vector([pred_cam[2], pred_cam[1], pred_cam[0]])
                # translation = Vector([pred_cam[2], pred_cam[0], pred_cam[1]])

                translation = Vector(
                    [pred_cam[0], -1 * pred_cam[2], pred_cam[1]])

                # print('Frame ' + str(frame))
                # print(pred_cam)

                pose = person_motion['pose'][frame]
                # apply the translation, pose and shape to the character
                apply_trans_pose_shape(translation, pose, person, frame)

                # arm_ob.pose.bones[body_model_name+'_root'].rotation_quaternion = rot_quat
                # arm_ob.pose.bones[body_model_name+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
                # dict_info['zrot'][iframe] = random_zrot

                scene.update()

        log('Finished setting keyframes')

        # random light
        sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
        # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
        sh_coeffs[0] = .5 + .9 * np.random.rand()
        sh_coeffs[1] = -.7 * np.random.rand()

        for person in persons.values():
            material = person['material']
            for index, coeff in enumerate(sh_coeffs):
                inp = index + 1
                material.node_tree.nodes['Script'].inputs[inp].default_value = coeff

        log('Finished setup')

        # iterate over the keyframes and render
        for frame in range(nr_frames):
            log("Rendering frame %d" % frame)

            scene.frame_set(frame)
            scene.render.filepath = join(render_temp_path, '%04d.png' % frame)
            bpy.ops.render.render(write_still=True)

            for person in persons.values():
                bone_name = get_bone_name(person['gender'], 'root')
                bone = person['armature'].pose.bones[bone_name]
                bone.rotation_quaternion = Quaternion((1, 0, 0, 0))

        cmd_ffmpeg = 'ffmpeg -y -r 30 -i %s -c:v h264 -pix_fmt yuv420p -crf 23 %s' % (
            join(render_temp_path, '%04d.png'), join(OUTPUT_PATH, render_filename))
        log("Generating RGB video (%s)" % cmd_ffmpeg)
        os.system(cmd_ffmpeg)

        # TODO: clear TMP_PATH


if __name__ == '__main__':
    main()
