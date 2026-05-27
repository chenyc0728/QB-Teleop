import pickle
from pathlib import Path

import cv2
import time
import tqdm
import numpy as np
from typing import Optional, List, Union, Dict
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from pytransform3d import rotations

from optimizer import Optimizer, LPFilter

from config import (
    OPERATOR2MANO,
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    RetargetingConfig,
)
from detector.single_hand_detector import SingleHandDetector

class SeqRetargeting:
    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: Optional[LPFilter] = None,
    ):
        self.optimizer = optimizer
        robot = self.optimizer.robot

        # Joint limit
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        # Temporal information
        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(
            np.float32
        )
        self.accumulated_time = 0
        self.num_retargeting = 0

        # Filter
        self.filter = lp_filter

        # Warm started
        self.is_warm_started = False

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        is_mano_convention: bool = False,
    ):
        """
        Initialize the wrist joint pose using analytical computation instead of retargeting optimization.
        This function is specifically for position retargeting with the flying robot hand, i.e. has 6D free joint
        You are not expected to use this function for vector retargeting, e.g. when you are working on teleoperation

        Args:
            wrist_pos: position of the hand wrist, typically from human hand pose
            wrist_quat: quaternion of the hand wrist, the same convention as the operator frame definition if not is_mano_convention
            hand_type: hand type, used to determine the operator2mano matrix
            is_mano_convention: whether the wrist_quat is in mano convention
        """
        # This function can only be used when the first joints of robot are free joints

        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos: {wrist_pos} is not a 3-dim vector.")
        if len(wrist_quat) != 4:
            raise ValueError(f"Wrist quat: {wrist_quat} is not a 4-dim vector.")

        operator2mano = OPERATOR2MANO[hand_type] if is_mano_convention else np.eye(3)
        robot = self.optimizer.robot
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = (
            rotations.matrix_from_quaternion(wrist_quat) @ operator2mano.T
        )
        target_wrist_pose[:3, 3] = wrist_pos

        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]

        # Set the dummy joints angles to zero
        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0

        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist

        euler = rotations.euler_from_matrix(
            target_root_pose[:3, :3], 0, 1, 2, extrinsic=False
        )
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])

        # Find the dummy joints
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        tic = time.perf_counter()

        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=np.clip(
                self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            ),
        )
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        robot_qpos[self.optimizer.idx_pin2target] = qpos

        if self.optimizer.adaptor is not None:
            robot_qpos = self.optimizer.adaptor.forward_qpos(robot_qpos)

        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def set_qpos(self, robot_qpos: np.ndarray):
        target_qpos = robot_qpos[self.optimizer.idx_pin2target]
        self.last_qpos = target_qpos

    def get_qpos(self, fixed_qpos: Optional[np.ndarray] = None):
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2target] = self.last_qpos
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        return robot_qpos

    def verbose(self):
        min_value = self.optimizer.opt.last_optimum_value()
        print(
            f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s"
        )
        print(f"Last distance: {min_value}")

    def reset(self):
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names

def retarget_video(
    retargeting: SeqRetargeting, video_path: str, hand_type: str, output_path: str, config: RetargetingConfig, yaml_config_path: str
):
    cap = cv2.VideoCapture(video_path)
    data = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        detector = SingleHandDetector(hand_type = hand_type, selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(
                    rgb
                )
                if num_box == 0:
                    continue

                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                qpos = retargeting.retarget(ref_value)
                data.append(qpos)
                pbar.update(1)

        meta_data = dict(
            yaml_config_path=yaml_config_path,
            dof=len(retargeting.optimizer.robot.dof_joint_names),
            joint_names=retargeting.optimizer.robot.dof_joint_names,
            urdf_path=config.urdf_path,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(dict(data=data, meta_data=meta_data), f)

        retargeting.verbose()
        cap.release()
        cv2.destroyAllWindows()

def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    print("Calling render_by_sapien...")
    # Generate rendering config
    use_rt = headless
    if not use_rt:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
    else:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    # Config is loaded only to find the urdf path and robot name
    urdf_path = meta_data["urdf_path"]
    # config_path = meta_data["config_path"]
    # config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    print("SAPIEN scene initialized in headless mode.") if headless else print("SAPIEN scene initialized with viewer.")

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))
    # cam.set_local_pose(sapien.Pose([0, 0, 1.0], [1, 0, 0, 0]))


    # Viewer
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.window.resize(1280, 720)
        viewer.control_window.show_origin_frame = True
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())

    else:
        viewer = None
    record_video = output_video_path is not None

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    robot_name = Path(urdf_path).stem
    
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5
    elif "vcl" in robot_name:
        loader.scale = 1.7
    elif "qb" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name and "qb" not in robot_name:
        filepath = str(urdf_path).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(urdf_path)
    filepath = str(filepath)
    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "vcl" in robot_name:
        # robot.set_pose(sapien.Pose([0, 0, -0.13], [0.7071, 0, -0.7071, 0]))
        robot.set_pose(sapien.Pose([0, -0.3, 0.2], [0.5, -0.5, 0.5, -0.5]))
    elif "qb" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))


    # Video recorder
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (cam.get_width(), cam.get_height())
        )

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    for qpos in tqdm.tqdm(data):
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if not headless:
            for _ in range(2):
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()

    scene = None

def main(
    robot_name: RobotName,
    video_path: str,
    output_pkl_path: str,
    output_video_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    headless: bool = False,
):
    yaml_config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    config = RetargetingConfig.load_from_file(yaml_config_path)
    retargeting = config.build()

    retarget_video(retargeting, video_path, hand_type.name.capitalize(), output_pkl_path, config, yaml_config_path)

    pickle_data = np.load(output_pkl_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]
    render_by_sapien(meta_data, data, output_video_path, headless = headless)

# RobotName: allegro; shadow; svh; leap; ability; inspire; panda
# RetargetingType: vector; position; dexpilot
# HandType: left; right
if __name__ == "__main__":
    main(
        robot_name=RobotName.shadow,
        video_path = r"D:\study\VScodes\dex-retargeting\example\vector_retargeting\data\human_hand_video.mp4",
        output_pkl_path = r"D:\study\VScodes\Retargeting\data\shadow_video1.pkl",
        output_video_path = r"D:\study\VScodes\Retargeting\data\shadow_video1.mp4",
        retargeting_type=RetargetingType.vector,
        hand_type=HandType.right,
    )
