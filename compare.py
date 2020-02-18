import glob
import math
import os
import pickle
import random
import subprocess
import sys
import time
import shutil
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

BG_IMAGE = 'data/backgrounds/compare.png'
TEXTURES_PATH = 'data/textures/'
MOTION_PATH = 'data/motion/'
SMPL_PATH = 'data/smpl/'
TMP_PATH = 'tmp/'
SHAPE_PATH = 'data/shape/shape_data.pkl'
SHADER_PATH = 'data/shader/shader.osl'
OUTPUT_PATH = 'output'

# Scene constraints
MIN_NR_PERSONS = 1
MAX_NR_PERSONS = 4
MAX_Z = -5.5
MIN_Z = -50

# Render settings
RENDER_WIDTH = 640
RENDER_HEIGHT = 360
FRAMES_PER_SECOND = 25

# Target size of the output dataset
TARGET_SIZE = 4

################
#    Utils     #
################

start_time = time.time()

def log(message):
    elapsed_time = time.time() - start_time
    # print("[%.2f s] %s" % (elapsed_time, message))
    logging.info("[%.2fs] %s" % (elapsed_time, message))


def get_body_model(gender):
    return os.path.join(SMPL_PATH, 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0])


################
#   Blender    #
################

smpl_bones = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
              'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']


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


def init_scene(motion_a, motion_b, shapes):
    # Set render system settings
    bpy.context.scene.render.engine = 'CYCLES'

    # Delete the default blender cube if it is in the scene
    bpy.ops.object.select_all(action='DESELECT')
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete(use_global=False)

    background_img = bpy.data.images.load(BG_IMAGE)

    assert(len(motion_a['frame_ids']) == len(motion_b['frame_ids']))

    nr_frames = len(motion_a['frame_ids'])
                
    persons = { 0: {}, 1: {} }

    for key, motion in enumerate([motion_a, motion_b]):
        deselect_all()
        # Add the motion to the person object
        persons[key]['motion'] = motion

        # Load the images picked at domain randomisation into Blender
        gender = 'male'

        clothing_img = bpy.data.images.load('data/textures/male/grey_male_0013.jpg')

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

        # Hide the armature artifacts
        armature.hide_set(True)
        # Rotate up and one up (TODO: why are estimated poses upside down???)
        armature.rotation_euler = Euler((np.pi, 0, 0))
        if key == 0:
          armature.location = Vector((-1.5, 1, -7))
        else:
          armature.location = Vector((1.5, 1, -7))

        # Autosmooth creates artifacts so turn it off
        mesh.data.use_auto_smooth = False
        # Clear existing animation data
        # mesh.data.shape_keys.animation_data_clear()
        # armature.animation_data_clear()

        n_sh_bshapes = len([k for k in mesh.data.shape_keys.key_blocks.keys()
                            if k.startswith('Shape')])

        # Get a body shape from CAESAR
        shape = shapes['%sshapes' % gender][:, :n_sh_bshapes][0]

        # Create a material with the clothing texture to apply to the human body model
        material = create_material(key, clothing_img)
        # Assign the existing spherical harmonics material to the body model
        mesh.active_material = material

        for i, shape_elem in enumerate(shape):
            k = 'Shape%03d' % i
            mesh.data.shape_keys.key_blocks[k].slider_min = -10
            mesh.data.shape_keys.key_blocks[k].slider_max = 10
            mesh.data.shape_keys.key_blocks[k].value = shape_elem

        deselect_all()
        # mesh.select_set(True)
        # bpy.context.view_layer.objects.active = mesh

        pelvis = armature.pose.bones[get_bone_name(gender, 'Pelvis')]

        orig_pelvis_loc = (armature.matrix_world.copy() @
                           pelvis.head.copy()) - Vector((-1., 1., 1.))

        bpy.context.view_layer.objects.active = armature
        orig_trans = np.asarray(pelvis.location).copy()

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
    bpy.context.view_layer.objects.active = cam_ob

    cam_ob.animation_data_clear()
    cam_ob.data.angle = math.radians(30)
    cam_ob.data.lens = 40
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    cam_ob.matrix_world = np.eye(4)

    cam_ob.matrix_world = Matrix(((1.0, 0.0, 0.0, 0.0),
                                  (0.0, 1.0, 0.09, 1.0),
                                  (0.0, -0.09, 1.0, 0.0),
                                  (0.0, 0.0, 0.0, 1.0)))

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.fps = FRAMES_PER_SECOND
    bpy.context.scene.cycles.shading_system = True
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.film_transparent = True

    # Set render size
    bpy.context.scene.render.resolution_x = RENDER_WIDTH
    bpy.context.scene.render.resolution_y = RENDER_HEIGHT
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Create the Composite Nodes graph, combining foreground (human bodies) with background image
    create_composite_nodes(background_img)

    return persons, cam_ob, nr_frames


def deselect_all():
    # TODO: check where this can be removed
    for ob in bpy.data.objects.values():
        ob.select_set(False)
    bpy.context.view_layer.objects.active = None

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
    uv_xform.inputs[1].default_value = (0, 0, 1)\

    uv_xform.operation = 'NORMALIZE'
    # TODO: in 2.79 stond hier nog average maar deze optie is verwijderd. Uitzoeken wat voor effect dit heeft.
    # uv_xform.operation = 'AVERAGE'

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
    tree = bpy.context.scene.node_tree

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

    tree.links.new(mix.outputs[0], composite_out.inputs[0])


################
#     Main     #
################

def main():
    # Set up logging format
    log_format = '[synth_motion] %(message)s'
    logging.basicConfig(level='INFO', format=log_format)
    # Set seed for random generator
    # seed = str(start_time)
    # random.seed(seed)
    # log('[init] Seed is %s' % seed)

    motions = []
    file_paths = [('mma_smooth_smplify/vibe_output_1.pkl', 'mma_not-smooth_not-smplify/vibe_output_1.pkl')]
    # Loop through files
    for a, b in file_paths:
        t = [None, None]
        # Open the file, load pickle
        with open(MOTION_PATH + a, 'rb') as file:
            t[0] = pickle.load(file)
        with open(MOTION_PATH + b, 'rb') as file:
            t[1] = pickle.load(file)
        motions.append(t)

    pairs = dict(zip(file_paths, motions))
    assert(len(pairs) > 0)

    # Load CEASAR shape data
    with open(SHAPE_PATH, 'rb') as file:
        shapes = pickle.load(file)

    for index, ((name_a, name_b), (motion_a, motion_b)) in enumerate(pairs.items()):
        # Create a filename for the final render and use it to
        render_filename = 'compare-%s.mp4' % (name_a + '-' + name_b)
        render_temp_path = join(TMP_PATH, render_filename)

        # Initialise Blender
        persons, cam_ob, nr_frames = init_scene(motion_a, motion_b, shapes)

        # Set motion and position keyframes to create the animation
        for frame in range(nr_frames):
            # Set the frame pointer
            bpy.context.scene.frame_set(frame)

            # Loop over the persons in the scene
            for key, person in persons.items():
                deselect_all()
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
                for i, blendshape in enumerate(blendshapes):
                    key = 'Pose%03d' % i

                    person['mesh'].data.shape_keys.key_blocks[key].value = blendshape
                    if not frame is None:
                        person['mesh'].data.shape_keys.key_blocks[key].keyframe_insert(
                            'value', index=-1, frame=frame)

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

        #     bpy.context.scene.frame_set(frame)
        #     bpy.context.scene.render.filepath = join(
        #         render_temp_path, '%04d.png' % frame)

        #     bpy.ops.render.render(write_still=True)

        #     # Why is this here?
        #     for person in persons.values():
        #         bone_name = get_bone_name(person['gender'], 'root')
        #         bone = person['armature'].pose.bones[bone_name]
        #         bone.rotation_quaternion = Quaternion((1, 0, 0, 0))

        # # Combine the frames into a video using ffmpeg
        # cmd_ffmpeg = 'ffmpeg -y -r %s -i %s -c:v h264 -pix_fmt yuv420p -crf 23 %s' % (
        #     FRAMES_PER_SECOND, join(render_temp_path, '%04d.png'), join(OUTPUT_PATH, render_filename))

        # os.system(cmd_ffmpeg)

        # # Clear temporary folder contents
        # for filename in os.listdir(TMP_PATH):
        #     file_path = os.path.join(TMP_PATH, filename)
        #     try:
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
        #     except Exception as e:
        #         log('Failed to clear temporary folder (%s)' % e)


if __name__ == '__main__':
    main()
