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
import logging

import numpy as np

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view as world2cam
from mathutils import Euler, Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

# Config

BG_PATH = 'data/backgrounds/'
TEXTURES_PATH = 'data/textures/'
MOTION_PATH = 'data/motion/'
SMPL_PATH = 'data/smpl/'
TMP_PATH = 'tmp/'
SHAPE_PATH = 'data/shape/shape_data.pkl'
SHADER_PATH = 'data/shader/shader.osl'
OUTPUT_PATH = 'output'

# Scene constraints
MIN_NR_PERSONS = 3
MAX_NR_PERSONS = 5
MAX_Z = -5.5
MIN_Z = -50

# Render settings
RENDER_WIDTH = 640
RENDER_HEIGHT = 360
FRAMES_PER_SECOND = 25

# Target size of the output dataset
TARGET_SIZE = 1

################
#    Utils     #
################

start_time = time.time()


class InfoFilter(logging.Filter):
    def filter(self, rec):
        print(rec)
        return True


def log(message):
    elapsed_time = time.time() - start_time
    # print("[%.2f s] %s" % (elapsed_time, message))
    logging.info("[%.2fs] %s" % (elapsed_time, message))


def load_motions():
    motions = []
    # List all pickle files in the directory
    pattern = os.path.join(MOTION_PATH, '**/*.pkl')
    file_paths = glob(pattern)
    names = list(map(lambda f: f.split(
        '/')[-1].replace('.pkl', ''), file_paths))
    # Loop through files
    for file_path in file_paths:
        # Open the file, load pickle
        with open(file_path, 'rb') as file:
            motion = pickle.load(file)
            motions.append(motion)

    return dict(zip(names, motions))

    # return names, motions


def get_body_model(gender):
    return os.path.join(SMPL_PATH, 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0])


logfile = os.path.join(TMP_PATH, 'blender_render.log')


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


# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

# transformation between pose and blendshapes


def rodrigues2bshapes(pose):
    joint_rotations = np.asarray(pose).reshape(24, 3)

    mat_rots = [Rodrigues(rot) for rot in joint_rotations]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return (mat_rots, bshapes)


def init_scene(motion, shapes):
    scene = get_main_scene()
    # Set render system settings
    scene.render.engine = 'CYCLES'
    scene.render.fps = FRAMES_PER_SECOND
    scene.render.use_antialiasing = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    # Delete the default blender cube if it is in the scene
    bpy.ops.object.select_all(action='DESELECT')
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete(use_global=False)

    # Get a random background image and load into Blender
    background_paths = glob(os.path.join(BG_PATH, '*'))
    background_img = bpy.data.images.load(random.choice(background_paths))

    nr_frames = min([len(motion[person]['frame_ids'])
                     for person in motion.keys()])

    persons = {}

    for key in list(motion.keys()):
        # Add the motion to the person object
        persons[key] = {}
        persons[key]['motion'] = motion[key]

        log('Going to init person %s' % key)

        # Load the images picked at domain randomisation into Blender
        gender = random.choice(['female', 'male'])

        # Get a random clothing texture and load into Blender
        texture_paths = glob(os.path.join(TEXTURES_PATH, gender, '*'))
        clothing_img = bpy.data.images.load(random.choice(texture_paths))

        old = list(bpy.data.objects)

        # Load the SMPL model as base for the scene
        bpy.ops.import_scene.fbx(
            filepath=get_body_model(gender), global_scale=100, axis_forward='Y', axis_up='Z')

        new = list(bpy.data.objects)
        delta = [x for x in new if not x in old]

        # Get armature using the default name
        if len(delta) < 2:
            raise Exception('Missed one new object')

        mesh = [obj for obj in list(delta)
                if obj.type == 'MESH'][-1]

        armature = [obj for obj in list(delta)
                    if obj.type == 'ARMATURE'][-1]

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

        # TODO: give random position somewhere here!!!

        orig_pelvis_loc = (armature.matrix_world.copy() *
                           pelvis.head.copy()) - Vector((-1., 1., 1.))

        scene.objects.active = armature
        orig_trans = np.asarray(pelvis.location).copy()

        persons[key]['gender'] = gender
        persons[key]['shape'] = shape
        persons[key]['clothing_img'] = clothing_img
        persons[key]['material'] = material
        persons[key]['orig_pelvis_loc'] = orig_pelvis_loc
        persons[key]['orig_trans'] = orig_trans
        persons[key]['armature'] = armature
        persons[key]['mesh'] = mesh
        persons[key]['movement'] = {
            'x': np.zeros(nr_frames),
            'z': np.zeros(nr_frames),
            'speed_x': np.random.normal(0, 0.01),
            'speed_z': np.random.normal(0, 0.01)
        }

    # Set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')

    cam_ob = bpy.data.objects['Camera']
    # TODO: scn vs scene?
    scn = bpy.context.scene
    scn.objects.active = cam_ob

    # bpy.data.objects['Camera'].data.type = 'ORTHO'
    # bpy.data.objects['Camera'].data.ortho_scale = 3

    cam_ob.animation_data_clear()
    cam_ob.data.angle = math.radians(30)
    cam_ob.data.lens = 40
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    cam_ob.matrix_world = np.eye(4)

    cam_ob.matrix_world = Matrix(((1.0, 0.0, 0.0, 0.0),
                                  (0.0, 0.9914448261260986,
                                   0.13052618503570557, 0.75),
                                  (0.0, -0.13052618503570557,
                                   0.9914448261260986, 0.0),
                                  (0.0, 0.0, 0.0, 1.0)))

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

    # frame = cam_ob.data.view_frame(scene)

    # # move from object-space into world-space
    # frame = [cam_ob.matrix_world * v for v in frame]

    # print(frame)

    # Create the Composite Nodes graph, combining foreground (human bodies) with background image
    create_composite_nodes(background_img)

    return scene, persons, cam_ob, nr_frames


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


def apply_trans_pose_shape(trans, scale, pose, person, frame=None):
    # Get the armature and mesh objects for this person
    armature = person['armature']
    mesh = person['mesh']
    gender = person['gender']
    shape = person['shape']

    # Set the location of the first bone to the translation parameter
    pelvis = armature.pose.bones[get_bone_name(gender, 'Pelvis')]
    # pelvis.location = trans

    # root_bone.scale = scale

    root = armature.pose.bones[get_bone_name(gender, 'root')]
    if frame is not None:
        root.keyframe_insert('location', frame=frame)

    # armature.location = trans
    # armature.scale = scale

    # armature.rotation = Euler((-2.1415, 0, 0))

    # if not frame is None:
    #     # Insert a keyframe
    #     bone_name = get_bone_name(gender, 'root')
    #     armature.pose.bones[bone_name].keyframe_insert('location', frame=frame)

    # Transform pose into rotation matrices (for pose) and pose blendshapes
    rotation_matrices, blendshapes = rodrigues2bshapes(pose)

    # Set the pose of each bone to the quaternion specified by pose
    for bone_index, rotation_matrix in enumerate(rotation_matrices):

        bone_name = get_bone_name(gender, smpl_bones[bone_index])
        bone = armature.pose.bones[bone_name]

        bone.rotation_quaternion = Matrix(rotation_matrix).to_quaternion()

        if not frame is None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    # for i, blendshape in enumerate(blendshapes):
    #     key = 'Pose%03d' % i

    #     mesh.data.shape_keys.key_blocks[key].value = blendshape
    #     if not frame is None:
    #         mesh.data.shape_keys.key_blocks[key].keyframe_insert(
    #             'value', index=-1, frame=frame)

    # # apply shape blendshapes
    # for i, shape_elem in enumerate(shape):
    #     key = 'Shape%03d' % i
    #     mesh.data.shape_keys.key_blocks[key].value = shape_elem
    #     if not frame is None:
    #         mesh.data.shape_keys.key_blocks[key].keyframe_insert(
    #             'value', index=-1, frame=frame)


def max_x_from_z(z):
    return z / 2.85


def is_roughly_in_view(x, y, z):
    if not y is 1:
        return False
    if abs(x) <= abs(max_x_from_z(z)):
        return False
    return True


# reset the joint positions of the character according to its new shape
def reset_joint_positions(scene, cam_ob, reg_ivs, joint_reg, person):

    shape = person['shape']
    mesh = person['mesh']
    armature = person['armature']

    scene.objects.active = armature

    # Rotate up and one up (TODO: why are estimated poses upside down???)
    armature.rotation_euler = Euler((np.pi, 0, 0))

    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    # empty array to hold vertices to regress from
    reg_vs = np.empty((len(reg_ivs), 3))

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

def distance(first, second):
	locx = second[0] - first[0]
	locy = second[1] - first[1]
	locz = second[2] - first[2]

	return math.sqrt((locx)**2 + (locy)**2 + (locz)**2) 

def overlap(scene, obj_name):

    subject_bm = bmesh.new()
    subject_bm.from_mesh(scene.objects[obj_name].data)
    subject_bm.transform(scene.objects[obj_name].matrix_world)
    subject_tree = BVHTree.FromBMesh(subject_bm)
    subject_center = scene.objects[obj_name].parent.location

    # check every object for intersection with every other object
    for obj in scene.objects.keys():
        if obj == obj_name or scene.objects[obj].type != 'MESH':
            continue

        if distance(subject_center, scene.objects[obj].parent.location) < 0.25:
            return True

        object_bm = bmesh.new()

        object_bm.from_mesh(scene.objects[obj].data)
        object_bm.transform(scene.objects[obj].matrix_world)
        object_tree = BVHTree.FromBMesh(object_bm)

        intersection = object_tree.overlap(subject_tree)

        # if list is empty, no objects are touching
        if intersection != []:
            log(obj + " and " + obj_name + " are touching!")
            return True

    return False


def random_walk(scene, person, frame, tries):
    if frame == 0:
        z_0 = max(MAX_Z - abs(np.random.normal(0, 5)), MIN_Z)
        person['movement']['z'][0] = z_0
        person['movement']['x'][0] = np.random.rand() * abs(max_x_from_z(z_0)) * 2 - abs(max_x_from_z(z_0))
        # person['movement']['x'][0] = 0
        # filling the coordinates with random variables
    else:
        if tries > 5:
            # When a person bumps into someone else, the speed is reduced and direction is reversed
            person['movement']['speed_z'] = -0.5 * person['movement']['speed_z']
            person['movement']['speed_x'] = -0.5 * person['movement']['speed_x']

        if tries > 100:
            log('Nr tries = %s' % str(tries))

        delta_z = np.random.normal(person['movement']['speed_z'], 0.01 * (tries % 10 + 1))
        delta_x = np.random.normal(person['movement']['speed_x'], 0.01 * (tries % 10 + 1))

        new_z = person['movement']['z'][frame - 1] + delta_z
        new_x = person['movement']['x'][frame - 1] + delta_x

        # Prevent people from walking off screen
        max_x = abs(max_x_from_z(new_z))
        if new_x > 0 and new_x > max_x:
            person['movement']['speed_x'] /= 2
            new_x = max_x
        if new_x < 0 and new_x < -1 * max_x:
            person['movement']['speed_x'] /= 2
            new_x = -1 * max_x

        # Prevent people from disappearing into the distance
        if new_z > MAX_Z:
            person['movement']['speed_z'] = 0
            new_z = MAX_Z
        if new_z < MIN_Z:
            new_z = MIN_Z
            person['movement']['speed_z'] = 0

        # TODO: misschien analyseren of een persoon in de buurt komt van een ander persoon en dan snelheid laten afnemen?

        person['movement']['speed_x'] = np.random.normal(person['movement']['speed_x'], 0.001)
        person['movement']['speed_z'] = np.random.normal(person['movement']['speed_z'], 0.001)

        person['movement']['z'][frame] = new_z
        person['movement']['x'][frame] = new_x

    return Vector((person['movement']['x'][frame], 0.75, person['movement']['z'][frame]))


################
#     Main     #
################

def main():
    log_format = '[synth_motion] [ln%(lineno)d] %(message)s'
    logging.basicConfig(level='INFO', format=log_format)

    # sys.stdout = open(os.devnull, 'w')

    seed = str(start_time)
    random.seed(seed)
    log('Seed is %s' % seed)

    # Load the extracted motions from disk
    log('Loading motion files')
    raw_motions = load_motions()
    log('Loaded %s raw motion files' % len(raw_motions))

    samples = {}

    log('Target size is %s' % TARGET_SIZE)

    # Combine motions until we have reached the target size of the synthetic dataset
    for sample in range(0, TARGET_SIZE):
        # Determine the number of persons in this sample
        nr_persons = random.choice(range(MIN_NR_PERSONS, MAX_NR_PERSONS + 1))

        # Determine the motion files to use for this sample
        motion_names = random.sample(raw_motions.keys(), nr_persons)
        # Extract the corresponding motion data
        motion_data = {n: raw_motions[n] for n in motion_names}

        # Unique identier for this sample contains motion_names for debugging purposes
        sample_identifier = str(sample) + '_' + '_'.join(motion_names)
        # Save data to samples dict
        samples[sample_identifier] = motion_data

    log('Loading CAESAR shape data')
    with open(SHAPE_PATH, 'rb') as file:
        shapes = pickle.load(file)
    joint_regressor = shapes['joint_regressor']
    regression_verts = shapes['regression_verts']

    # Loop over all samples and create videos
    for sample_index, (sample_id, motion) in enumerate(samples.items()):
        log('Processing sample %s' % sample_index)

        # Create a filename for the final render and use it to
        render_filename = '%s.mp4' % (sample_id + '_' + seed)
        render_temp_path = join(TMP_PATH, render_filename)
        log('Filename will be %s' % render_filename)

        # Initialise Blender
        scene, persons, cam_ob, nr_frames = init_scene(motion, shapes)

        # TODO: figure out why this is needed
        for person in persons.values():
            curr_shape = reset_joint_positions(scene,
                                               cam_ob, regression_verts, joint_regressor, person)

        # Set motion and position keyframes to create the animation
        for frame in range(nr_frames):

            # Set a new keyframe
            scene.frame_set(frame)

            # Loop over the persons in the scene
            for key, person in persons.items():

                # Transform pose into rotation matrices (for pose) and pose blendshapes
                rotation_matrices, blendshapes = rodrigues2bshapes(
                    person['motion']['pose'][frame])

                # Set the pose of each bone to the quaternion specified by pose
                for bone_index, rotation_matrix in enumerate(rotation_matrices):
                    bone = person['armature'].pose.bones[get_bone_name(
                        person['gender'], smpl_bones[bone_index])]
                    bone.rotation_quaternion = Matrix(
                        rotation_matrix).to_quaternion()
                    bone.keyframe_insert(
                        'rotation_quaternion', frame=frame)
                    # bone.keyframe_insert('location', frame=frame)

                # apply pose blendshapes
                # for i, blendshape in enumerate(blendshapes):
                #     key = 'Pose%03d' % i

                #     mesh.data.shape_keys.key_blocks[key].value = blendshape
                #     if not frame is None:
                #         mesh.data.shape_keys.key_blocks[key].keyframe_insert(
                #             'value', index=-1, frame=frame)

                # # apply shape blendshapes
                # for i, shape_elem in enumerate(shape):
                #     key = 'Shape%03d' % i
                #     mesh.data.shape_keys.key_blocks[key].value = shape_elem
                #     if not frame is None:
                #         mesh.data.shape_keys.key_blocks[key].keyframe_insert(
                #             'value', index=-1, frame=frame)


                # mesh = [c for c in person['armature'].children if c.type == 'MESH'][0]

                # person['random_walk'] = { 'x': x, 'z': z }

                # orig_cam = list(
                #     person['motion']['orig_cam'][frame_offset_index])

                # scale_x = orig_cam[0]
                # scale_y = orig_cam[1]
                # trans_x = orig_cam[2]
                # trans_y = orig_cam[3]

                walk = True
                walk_i = 0
                while walk:
                    walk_i += 1
                    if walk_i > 1:
                        log('Frame %s - Walk blocked for %s' % (str(frame), person['mesh'].name))

                    # get a new location from the random walk algorithm
                    new_location = random_walk(scene, person, frame, walk_i)
                    # Set person location keyframe
                    person['armature'].location = new_location
                    # Update scene
                    scene.update()
                    # Calculate overlap and use it to determine if walk should be extended
                    # Overlap can only be calculated when scene is updated, not within random walk algorithm
                    walk = overlap(scene, person['mesh'].name)
                person['armature'].keyframe_insert('location', frame=frame)


        # Random light
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

        area_i = list(map(
            lambda x: x.type, bpy.data.window_managers[0].windows[0].screen.areas)).index('VIEW_3D')
        area = bpy.data.window_managers[0].windows[0].screen.areas[area_i]

        space_i = list(map(lambda x: x.type, area.spaces)).index('VIEW_3D')
        space = area.spaces[space_i]

        space.region_3d.view_matrix = Matrix(((1, 0, 0, 0),
                                              (0, 1, 0, 0),
                                              (0, 0, 1, -5),
                                              (0, 0, 0, 1)))

        # iterate over the keyframes and render
        # for frame in range(min(nr_frames, 200)):
        #     log("Rendering video %d/%d, frame %d/%d" %
        #         (sample_index + 1, len(samples), frame + 1, nr_frames))

        #     scene.frame_set(frame)
        #     scene.render.filepath = join(render_temp_path, '%04d.png' % frame)

        #     bpy.ops.render.render(write_still=True)

        #     # Why is this here?
        #     for person in persons.values():
        #         bone_name = get_bone_name(person['gender'], 'root')
        #         bone = person['armature'].pose.bones[bone_name]
        #         bone.rotation_quaternion = Quaternion((1, 0, 0, 0))

        # cmd_ffmpeg = 'ffmpeg -y -r %s -i %s -c:v h264 -pix_fmt yuv420p -crf 23 %s' % (
        #     FRAMES_PER_SECOND, join(render_temp_path, '%04d.png'), join(OUTPUT_PATH, render_filename))
        # log("Generating RGB video (%s)" % cmd_ffmpeg)
        # os.system(cmd_ffmpeg)

        # log("Saved RGB video")
        # # TODO: clear TMP_PATH
        # bpy.ops.wm.read_factory_settings()


if __name__ == '__main__':
    main()
