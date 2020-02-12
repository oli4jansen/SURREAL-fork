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


def deselect_all():
    for ob in bpy.data.objects.values():
        ob.select_set(False)
    bpy.context.view_layer.objects.active = None



################
#     Main     #
################

def main():

    log('Loading CAESAR shape data')
    with open(SHAPE_PATH, 'rb') as file:
        shapes = pickle.load(file)

    joint_regressor = shapes['joint_regressor']
    regression_verts = shapes['regression_verts']




    # Set render system settings
    bpy.context.scene.render.engine = 'CYCLES'

    # Delete the default blender cube if it is in the scene
    bpy.ops.object.select_all(action='DESELECT')
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete(use_global=False)



    gender = 'female'


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

    armature.hide_set(True)

    # Autosmooth creates artifacts so turn it off
    mesh.data.use_auto_smooth = False
    # Clear existing animation data
    mesh.data.shape_keys.animation_data_clear()
    armature.animation_data_clear()

    n_sh_bshapes = len([k for k in mesh.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # Get a body shape from CAESAR
    shape = random.choice(shapes['%sshapes' % gender][:, :n_sh_bshapes])


    shape = [0,-10,0,0,0,0,0,0,0,0]

    # # unblocking both the pose and the blendshape limits
    for k in mesh.data.shape_keys.key_blocks.keys():
        if not k in bpy.data.shape_keys['Key'].key_blocks:
            print('%s is not in shape_keys' % k)
        else:
            bpy.data.shape_keys['Key'].key_blocks[k].slider_min = -10
            bpy.data.shape_keys['Key'].key_blocks[k].slider_max = 10

    for i, shape_elem in enumerate(shape):
        key = 'Shape%03d' % i
        mesh.data.shape_keys.key_blocks[key].value = shape_elem

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
                                  (0.0, 0.9914448261260986,
                                   0.13052618503570557, 0.75),
                                  (0.0, -0.13052618503570557,
                                   0.9914448261260986, 0.0),
                                  (0.0, 0.0, 0.0, 1.0)))


    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.fps = FRAMES_PER_SECOND
    bpy.context.scene.cycles.shading_system = True
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.film_transparent = True

    bpy.context.scene.render.resolution_x = RENDER_WIDTH
    bpy.context.scene.render.resolution_y = RENDER_HEIGHT
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'



if __name__ == '__main__':
    main()
