import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
from torch.nn import Module
from tqdm import trange
import numpy as np

from manopth.manolayer import ManoLayer

from pytransform3d import rotations
from pytransform3d import transformations as pt

import sapien
from sapien import internal_renderer as R
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dataset import YCB_CLASSES

from URDF_units import yourdfpy as urdf

from config import (
    MANO_DIR,
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
    RetargetingConfig,
)
from retarget_from_video import SeqRetargeting
from utils.utils import compute_smooth_shading_normal_np

class MANOLayer(Module):
    """Wrapper layer for manopth ManoLayer."""

    def __init__(self, side, betas):
        """Constructor.
        Args:
          side: MANO hand type. 'right' or 'left'.
          betas: A numpy array of shape [10] containing the betas.
        """
        super(MANOLayer, self).__init__()

        self._side = side
        self._betas = betas
        self._mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side=self._side,
            mano_root=str(MANO_DIR / "models"),
            use_pca=True,
        )

        b = torch.from_numpy(self._betas).unsqueeze(0)
        f = self._mano_layer.th_faces
        self.register_buffer("b", b)
        self.register_buffer("f", f)

        v = (
            torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(0, 1)).permute(
                2, 0, 1
            )
            + self._mano_layer.th_v_template
        )
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer("root_trans", r)

    def forward(self, p, t):
        """Forward function.
        Args:
          p: A tensor of shape [B, 48] containing the pose.
          t: A tensor of shape [B, 3] containing the trans.
        Returns:
          v: A tensor of shape [B, 778, 3] containing the vertices.
          j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)
        v /= 1000
        j /= 1000
        return v, j

class HandDatasetSAPIENViewer:
    def __init__(self, headless=False, use_ray_tracing=False):
        # Setup
        if not use_ray_tracing:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        else:
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")

        # Scene
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)

        # Lighting
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_directional_light(
            np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
        )
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        # Add ground
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # Viewer
        if not headless:
            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.window.resize(1280, 720)  # window size 

            viewer.set_camera_xyz(1.5, 0, 1)
            viewer.set_camera_rpy(0, -0.8, 3.14)
            viewer.control_window.toggle_origin_frame(False)
            self.viewer = viewer
        else:
            self.camera = scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
            self.camera.set_local_pose(sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061]))

        self.headless = headless

        # Create table
        white_diffuse = sapien.render.RenderMaterial()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
        white_diffuse.set_roughness(0.9)
        builder = scene.create_actor_builder()
        builder.add_box_collision(
            sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02])
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, -0.02]),
            half_size=np.array([0.5, 2.0, 0.02]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))

        # Caches
        sapien.render.set_log_level("error")
        self.scene = scene
        self.internal_scene: R.Scene = scene.render_system._internal_scene
        self.context: R.Context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(
            np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0
        )

        self.mano_layer: Optional[MANOLayer] = None
        self.mano_face: Optional[np.ndarray] = None
        self.camera_pose: Optional[sapien.Pose] = None
        self.objects: List[sapien.Entity] = []
        self.nodes: List[R.Node] = []

    def clear_all(self):
        for actor in self.objects:
            self.scene.remove_actor(actor)
        for _ in range(len(self.objects)):
            actor = self.objects.pop()
            self.scene.remove_actor(actor)
        self.clear_node()
        self.mano_layer = None

    def clear_node(self):
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()
            self.internal_scene.remove_node(node)

    def load_object_hand(self, data: Dict):
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]
        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]
        for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
            self._load_ycb_object(ycb_id, ycb_mesh_file)

        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        pose_vec = pt.pq_from_transform(extrinsic_mat)
        self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()

    def _load_ycb_object(self, ycb_id, ycb_mesh_file):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(ycb_mesh_file)
        actor = builder.build_static(name=YCB_CLASSES[ycb_id])
        self.objects.append(actor)

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
        # pose parameters all zero, no hand is detected
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        if not use_camera_frame:
            camera_mat = self.camera_pose.to_transformation_matrix()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)

        return vertex, joint

    def _update_hand(self, vertex):
        self.clear_node()
        normal = compute_smooth_shading_normal_np(vertex, self.mano_face)
        mesh = self.context.create_mesh_from_array(vertex, self.mano_face, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)

    def render_dexycb_data(self, data: Dict, fps=10):
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        frame_num = hand_pose.shape[0]

        if self.headless:
            video_path = Path(__file__).parent.resolve() / "data/human_hand_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        step_per_frame = int(60 / fps)
        for i in trange(frame_num):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, _ = self._compute_hand_geometry(hand_pose_frame)
            if vertex is not None:
                self._update_hand(vertex)
            for k in range(len(self.objects)):
                pos_quat = object_pose_frame[k]
                pose = self.camera_pose * sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )
                self.objects[k].set_pose(pose)
            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()

class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless=False,
        use_ray_tracing=False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem and "qb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = (
                Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            )
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # Loop rendering
        step_per_frame = int(60 / fps)
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = self.camera_pose * sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(
                        pose_offsets[copy_ind] * pose
                    )

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()

