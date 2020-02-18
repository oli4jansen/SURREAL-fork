import logging
import math
import os
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from glob import glob
from os import getcwd, getenv, remove
from os.path import dirname, exists, join, realpath
from pickle import load
from random import choice

import numpy as np

import bmesh
import bpy
from bpy_extras.object_utils import world_to_camera_view as world2cam
from mathutils import Euler, Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

# Data path locations
BG_PATH = 'data/backgrounds/'
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
MAX_FRAMES = 200

# Target size of the output dataset
TARGET_SIZE = 10

SMPL_BONE_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
                   'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']


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


class Motion():
    """
    The Motion class is instantiated for every person in the captured motion data.
    It handles the loading of the SMPL model, texturing the mesh, animating the armature and applying a random walk.
    """

    def __init__(self, synthesiser, pred_cam, orig_cam, verts, pose, betas, joints3d, joints2d, bboxes, frame_ids):
        self.synthesiser = synthesiser
        self.identifier = random.getrandbits(32)

        self.pred_cam = pred_cam
        self.orig_cam = orig_cam
        self.verts = verts
        self.pose = pose
        self.betas = betas
        self.joints3d = joints3d
        self.joints2d = joints2d
        self.bboxes = bboxes
        self.frame_ids = frame_ids

        self.nr_frames = len(frame_ids)
        self.start_frame = 0
        self.end_frame = len(frame_ids) - 1

        self.movement = {
            'x': np.zeros(self.nr_frames),
            'z': np.zeros(self.nr_frames),
            'speed_x': np.random.normal(0, 0.01),
            'speed_z': np.random.normal(0, 0.01)
        }

    def crop(self, start_frame, end_frame):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def load(self):
        self._set_gender()
        self._import_body_model()
        self._set_texture()

    def _set_gender(self):
        self.gender = random.choice(['female', 'male'])

    def _set_texture(self):
        material = bpy.data.materials.new(
            name='Material_%s' % self.identifier)

        # Copy the base shader file to temporary directory
        tmp_shader_path = join(
            getcwd(), TMP_PATH, 'shader_%s.osl' % self.identifier)
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

        # Get a random clothing texture and load into Blender
        texture_paths = glob(os.path.join(TEXTURES_PATH, self.gender, '*'))
        clothing_img = bpy.data.images.load(random.choice(texture_paths))

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

        # Assign the existing spherical harmonics material to the body model
        self.mesh.active_material = material

        # Set the shader coefficients taken from synthesiser
        for index, coeff in enumerate(self.synthesiser.shader_coeffs):
            material.node_tree.nodes['Script'].inputs[index +
                                                      1].default_value = coeff

    def _import_body_model(self):
        filepath = join(
            SMPL_PATH, 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % self.gender[0])
        # Copy list of all objects that are in Blender BEFORE importing
        old = list(bpy.data.objects)
        # Import SMPL FBX file
        bpy.ops.import_scene.fbx(
            filepath=filepath, global_scale=100, axis_forward='Y', axis_up='Z')
        # Copy list of all objects that are in Blender AFTER importing
        new = list(bpy.data.objects)
        delta = [x for x in new if not x in old]
        # Get armature using the default name
        if len(delta) < 2:
            raise Exception('Missed one new object')

        self.mesh = [obj for obj in list(delta)
                     if obj.type == 'MESH'][-1]

        self.armature = [obj for obj in list(delta)
                         if obj.type == 'ARMATURE'][-1]

        # Hide the armature artifacts
        self.armature.hide_set(True)
        # Rotate up and one up (TODO: why are estimated poses upside down???)
        self.armature.rotation_euler = Euler((np.pi, 0, 0))

        # Autosmooth creates artifacts so turn it off
        self.mesh.data.use_auto_smooth = False
        # Clear existing animation data
        # mesh.data.shape_keys.animation_data_clear()
        # armature.animation_data_clear()

        n_sh_bshapes = len([k for k in self.mesh.data.shape_keys.key_blocks.keys()
                            if k.startswith('Shape')])

        assert(n_sh_bshapes == 10)

        # Get a body shape from CAESAR
        self.shape = random.choice(
            self.synthesiser.shapes['%sshapes' % self.gender][:, :n_sh_bshapes])

        for i, shape_elem in enumerate(self.shape):
            k = 'Shape%03d' % i
            self.mesh.data.shape_keys.key_blocks[k].slider_min = -10
            self.mesh.data.shape_keys.key_blocks[k].slider_max = 10
            self.mesh.data.shape_keys.key_blocks[k].value = shape_elem

    def _get_bone_name(self, bone):
        return self.gender[0] + '_avg_' + bone


class Synthesiser():

    def __init__(self):
        # Set up logging format
        log_format = '[synth_motion] %(message)s'
        logging.basicConfig(level='INFO', format=log_format)

        # Load CEASAR shape data
        with open(SHAPE_PATH, 'rb') as file:
            self.shapes = pickle.load(file)

    def run(self):
        # Collect motion data from disk
        self.motion_data = self._collect_motion_data()
        # Generate samples by combining motion data
        self.samples = self._generate_samples(self.motion_data)

        # Loop over all samples and create videos
        for sample_id, motion_list in self.samples.items():
            self._synthesise(sample_id, motion_list)

    def _get_number_of_frames(self, motion):
        return min([m.nr_frames for m in motion.values()])

    def _synthesise(self, sample_id, motion_list):
        self._reset_blender()

        # Determine the number of frames for this sample based on the motion data in it
        nr_frames = self._get_number_of_frames(motion_list)

        # Crop all motions
        # TODO: maybe not always cut off the end but also (sometimes?) a part of the beginning
        for m in motion_list.values():
            m.crop(0, nr_frames - 1)

        # Initialise Blender
        self._init_scene(nr_frames, motion_list)

        # Set motion and position keyframes to create the animation
        self._animate(nr_frames, motion_list)

        # Render this sample
        self._render(sample_id, nr_frames)

        # Clean Blender for the next sample
        self._clean()

    def _reset_blender(self):
        # Set render system settings
        bpy.context.scene.render.engine = 'CYCLES'

        # Delete the default blender cube if it is in the scene
        bpy.ops.object.select_all(action='DESELECT')

        for obj in bpy.data.objects:
            # logging.info(obj.type)
            if obj.type == 'MESH' or obj.type == 'ARMATURE':
                bpy.data.objects.remove(obj)
            

    def _init_scene(self, nr_frames, motion_list):
        # Get a random background image and load into Blender
        background_paths = glob(os.path.join(BG_PATH, '*'))
        self.background_img = bpy.data.images.load(
            random.choice(background_paths))

        # Random light coefficients, to be used for Blender material shader.
        # Partly taken from SURREAL but takes background image into account
        self.shader_coeffs = .7 * (2 * np.random.rand(9) - 1)
        self.shader_coeffs[0] = .5 + .6 * \
            statistics.mean(self.background_img.pixels) + .15 * np.random.rand()
        self.shader_coeffs[1] = -.7 * np.random.rand()

        # Load motion data into Blender
        for key in list(motion_list.keys()):
            motion_list[key].load()

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

        # Create the Composite Nodes graph, combining foreground (human models) with background image
        self._create_composite_nodes()

        return cam_ob

    def _create_composite_nodes(self):
        assert(self.background_img != None)
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
        bg_im.image = self.background_img

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

    def _collect_motion_data(self):
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
                # Wrap into Motion object
                motions.append(Motion(self, **motion))

        raw_motions = dict(zip(names, motions))
        assert(len(raw_motions) > 0)
        return raw_motions

    def _generate_samples(self, motion_data):
        samples = {}
        # Combine motions until we have reached the target size of the synthetic dataset
        for sample in range(0, TARGET_SIZE):
            # The number of persons in the sample is capped by the nummer of raw motions and MAX_NR_PERSONS flag
            nr_persons_range = range(MIN_NR_PERSONS, min(
                len(motion_data), MAX_NR_PERSONS) + 1)
            # Determine the number of persons in this sample
            nr_persons = random.choice(nr_persons_range)
            # Determine the motion files to use for this sample
            motion_names = random.sample(motion_data.keys(), nr_persons)
            motion_names.sort()
            # Unique identier for this sample contains motion_names for debugging purposes
            sample_id = str(sample) + '_' + \
                '_'.join(motion_names) + '_' + str(int(time.time()))

            # Save data to samples dict
            samples[sample_id] = {
                n: motion_data[n] for n in motion_names}
        return samples

    def _animate(self, nr_frames, motion_list):
        for frame in range(nr_frames):
            # Set the frame pointer
            bpy.context.scene.frame_set(frame)

            # Loop over the persons in the scene
            for key, motion in motion_list.items():
                # Transform pose into rotation matrices (for pose) and pose blendshapes
                rotation_matrices, blendshapes = rodrigues2bshapes(
                    motion.pose[frame])

                # Set the pose of each bone to the quaternion specified by pose
                for bone_index, rotation_matrix in enumerate(rotation_matrices):
                    bone = motion.armature.pose.bones[motion._get_bone_name(
                        SMPL_BONE_NAMES[bone_index])]
                    bone.rotation_quaternion = Matrix(
                        rotation_matrix).to_quaternion()
                    bone.keyframe_insert(
                        'rotation_quaternion', frame=frame)

                # apply pose blendshapes
                for i, blendshape in enumerate(blendshapes):
                    key = 'Pose%03d' % i

                    motion.mesh.data.shape_keys.key_blocks[key].value = blendshape
                    if not frame is None:
                        motion.mesh.data.shape_keys.key_blocks[key].keyframe_insert(
                            'value', index=-1, frame=frame)

                walk = True
                walk_i = 0
                while walk:
                    walk_i += 1
                    # get a new location from the random walk algorithm
                    new_location = self._random_walk(motion, frame, walk_i)
                    # Set person location keyframe
                    motion.armature.location = new_location
                    # Update scene
                    bpy.context.view_layer.update()
                    # Calculate overlap and use it to determine if walk should be extended
                    # Overlap can only be calculated when scene is updated, not within random walk algorithm
                    walk = self._overlap(motion.mesh.name)
                motion.armature.keyframe_insert('location', frame=frame)

    def _max_x_from_z(self, z):
        return z / 2.85

    def _is_roughly_in_view(self, x, y, z):
        if abs(x) <= abs(self._max_x_from_z(z)):
            return False
        return True

    def _distance(self, first, second):
        locx = second[0] - first[0]
        locy = second[1] - first[1]
        locz = second[2] - first[2]

        return math.sqrt((locx)**2 + (locy)**2 + (locz)**2)


    def _overlap(self, obj_name):

        subject_bm = bmesh.new()
        subject_bm.from_mesh(bpy.context.scene.objects[obj_name].data)
        subject_bm.transform(bpy.context.scene.objects[obj_name].matrix_world)
        subject_tree = BVHTree.FromBMesh(subject_bm)
        subject_center = bpy.context.scene.objects[obj_name].parent.location

        # check every object for intersection with every other object
        for obj in bpy.context.scene.objects.keys():
            if obj == obj_name or bpy.context.scene.objects[obj].type != 'MESH':
                continue

            if self._distance(subject_center, bpy.context.scene.objects[obj].parent.location) < 0.35:
                return True

            object_bm = bmesh.new()

            object_bm.from_mesh(bpy.context.scene.objects[obj].data)
            object_bm.transform(bpy.context.scene.objects[obj].matrix_world)
            object_tree = BVHTree.FromBMesh(object_bm)

            intersection = object_tree.overlap(subject_tree)

            # if list is empty, no objects are touching
            if intersection != []:
                return True

        return False


    def _random_walk(self, motion, frame, tries):
        if frame == 0:
            z_0 = max(MAX_Z - abs(np.random.normal(0, 5)), MIN_Z)
            motion.movement['z'][0] = z_0
            motion.movement['x'][0] = np.random.rand(
            ) * abs(self._max_x_from_z(z_0)) * 2 - abs(self._max_x_from_z(z_0))
        else:
            if tries > 5:
                # When a person bumps into someone else, the speed is reduced and direction is reversed
                motion.movement['speed_z'] = - \
                    0.5 * motion.movement['speed_z']
                motion.movement['speed_x'] = - \
                    0.5 * motion.movement['speed_x']

            if tries > 100:
                logging.info('Nr tries = %s' % str(tries))

            delta_z = np.random.normal(
                motion.movement['speed_z'], 0.01 * (tries % 10 + 1))
            delta_x = np.random.normal(
                motion.movement['speed_x'], 0.01 * (tries % 10 + 1))

            new_z = motion.movement['z'][frame - 1] + delta_z
            new_x = motion.movement['x'][frame - 1] + delta_x

            # Prevent people from walking off screen
            max_x = abs(self._max_x_from_z(new_z))
            if new_x > 0 and new_x > max_x:
                motion.movement['speed_x'] /= 2
                new_x = max_x
            if new_x < 0 and new_x < -1 * max_x:
                motion.movement['speed_x'] /= 2
                new_x = -1 * max_x

            # Prevent people from disappearing into the distance
            if new_z > MAX_Z:
                motion.movement['speed_z'] = 0
                new_z = MAX_Z
            if new_z < MIN_Z:
                new_z = MIN_Z
                motion.movement['speed_z'] = 0

            # TODO: misschien analyseren of een persoon in de buurt komt van een ander persoon en dan snelheid laten afnemen?

            motion.movement['speed_x'] = np.random.normal(
                motion.movement['speed_x'], 0.001)
            motion.movement['speed_z'] = np.random.normal(
                motion.movement['speed_z'], 0.001)

            motion.movement['z'][frame] = new_z
            motion.movement['x'][frame] = new_x

        return Vector((motion.movement['x'][frame], 0.5, motion.movement['z'][frame]))

    def _render(self, sample_id, nr_frames):
        self._render_images(sample_id, nr_frames)
        self._combine_images_as_video(sample_id)

    def _render_images(self, sample_id, nr_frames):
        # iterate over the keyframes and render
        for frame in range(min(nr_frames, MAX_FRAMES)):
            bpy.context.scene.frame_set(frame)
            filepath = join(TMP_PATH, '%04d.png' % frame)
            bpy.context.scene.render.filepath = filepath

            bpy.ops.render.render(write_still=True)

    def _combine_images_as_video(self, sample_id):
        render_filename = str(sample_id) + '.mp4'
        # Combine the frames into a video using ffmpeg
        cmd_ffmpeg = 'ffmpeg -y -r %s -i %s -c:v h264 -pix_fmt yuv420p -crf 23 %s' % (
            FRAMES_PER_SECOND, join(TMP_PATH, '%04d.png'), join(OUTPUT_PATH, render_filename))

        os.system(cmd_ffmpeg)

    def _clean(self):
        # Clear temporary folder contents
        for filename in os.listdir(TMP_PATH):
            file_path = os.path.join(TMP_PATH, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.info('Failed to clear temporary folder (%s)' % e)


if __name__ == '__main__':
    synthesiser = Synthesiser()
    synthesiser.run()
