from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d.euler
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from hand_detector.hand_monitor import Record3DSingleHandMotionControl
from hand_detector.record3d_app_realsense import RealsenseApp
from test_hand_pose import MoCapDetector, landmarks_to_pixel_array
from detector.single_hand_detector import SingleHandDetector
from coord_converter import SAPIEN2MEDIAPIPE
from mediapipe.framework.formats import landmark_pb2
import cv2
import scipy
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import time

default_camera_mat = np.array([
    [614.450317, 0., 332.668884],
    [0., 614.965996, 246.103592],
    [0., 0., 1.]])

class SingleHandDepthDetector:
    SUPPORT_HAND_MODE = ["Right", "Left"]
    SUPPORT_DETECTOR = ["Mediapipe", "MoCap"]
    def __init__(self, hand_mode: str, detector = "Mediapipe", show_hand=True, virtual_video_file="", need_init=True):
        if hand_mode not in self.SUPPORT_HAND_MODE:
            raise ValueError(
                f"Mode {hand_mode} is invalid. Current {len(self.SUPPORT_HAND_MODE)} mode are supported: "
                f"{self.SUPPORT_HAND_MODE} ")
        if detector == "Mocap": detector = "MoCap" 
        if detector not in self.SUPPORT_DETECTOR:
            raise ValueError(
                f"Mode {detector} is invalid. Current {len(self.SUPPORT_DETECTOR)} mode are supported: "
                f"{self.SUPPORT_DETECTOR} ")

        # Camera app 获取相机内参
        self.camera = RealsenseApp(file=virtual_video_file)
        self.camera.connect_to_device()
        self.camera_mat = self.camera.camera_intrinsics
        self.focal_length = self.camera.camera_intrinsics[0, 0]
        print("Camera Intrinsics:", self.camera_mat)

        # Hand detection 
        # mediapipe_hand_type = "right" if hand_mode == "right_hand" else "left"
        if detector == "Mediapipe":
            self.detector_name = "Mediapipe"
            self.detector = SingleHandDetector(hand_type=hand_mode)
        elif detector == "MoCap":
            self.detector_name = "MoCap"
            self.detector = MoCapDetector(hand_type=hand_mode)

        # Offset based bbox estimation 3D位移
        self.offset = {"left_hand": np.zeros(3, dtype=np.float32), "right_hand": np.zeros(3, dtype=np.float32)}

        self.num_box = 0
        self.joint_pos = np.zeros([21,3])
        self.keypoint_2d = np.zeros([21,3])
        self.wrist_rot = np.eye(3) # camera2hand
        self.wrist_pos_world = np.array([0, 0, 0])
    
    def draw_skeleton_on_image(
        self, image, keypoint_2d, style="white", show_coord=False, keypoint_3d=None
    ):
        return self.detector.draw_skeleton_on_image(image, keypoint_2d, style=style, show_coord=show_coord, keypoint_3d=keypoint_3d)
    
    def compute_3d_offset(self, keypoint_2d, joint_pos, wrist_rot,  depth: np.ndarray, pred_output=None):
        """
        由预测得到的图像中关键点坐标+深度信息获取真实世界的三维坐标
            :keypoint_2d: [21,3]的NormalizedLandmarkList {x,y,z}
            :joint_pos: [21,3]的MANO手部21个关键点的3D坐标, 手腕为原点
            :wrist_rot: [3,3]的Camera2Hand旋转矩阵
            :depth: 深度图
            :pred_output: FrankMocap预测输出
        返回相机坐标系下的偏移
        """
        height, width = depth.shape
        # 获取图像中关键点位置
        if self.detector_name == "Mediapipe":
            keypoint_img = landmarks_to_pixel_array(keypoint_2d, (height, width))
        elif self.detector_name == "MoCap":
            keypoint_img = keypoint_2d
        # 将mano手部关节坐标转换回opencv相机坐标系
        joint_pos_smplx = joint_pos @ wrist_rot
        # 去整+防止越界
        if pred_output is None:
            mask_int = np.rint(keypoint_img[:, :2]).astype(int)
        else:
            mask_int = np.rint(pred_output["pred_vertices_img"][:, :2]).astype(int)  
        mask_int = np.clip(mask_int, [0, 0], [width - 1, height - 1])
        # mask_array = np.zeros([height, width])
        # mask_array[mask_int[:, 0], mask_int[:, 1]] = 1
        # # 膨胀
        # mask_array = self.dilate_mask(mask_array,5)
        # # 还原为点列表
        # mask_int = np.array(np.nonzero(mask_array)).T

        # Image space vertices 21个关节的深度信息, 注意y,x的顺序
        depth_vertices = depth[mask_int[:, 1], mask_int[:, 0]]
        depth_median = np.nanmedian(depth_vertices)
        # 清除离群点
        depth_valid_mask = np.nonzero(np.abs(depth_vertices - depth_median) < 0.2)[0]
        valid_vertex_depth = depth_vertices[depth_valid_mask]

        # Hand frame vertices
        v_smpl = joint_pos_smplx[depth_valid_mask] if pred_output is None else pred_output["pred_vertices_smpl"][depth_valid_mask]
        # 取相机坐标系下手部坐标的z(深度)
        z_smpl = v_smpl[:, 2]
        # 按z排序，返回从小到大的索引
        z_near_to_far_order = np.argsort(z_smpl)

        # Filter depth with same pixel pos to the front position
        # 21个关节中，有效筛选后按对应z由小到大排列
        valid_mask_int = mask_int[depth_valid_mask, :][z_near_to_far_order, :]
        mask_int_encoding = valid_mask_int[:, 0] * 1e5 + valid_mask_int[:, 1]
        _, unique_indices = np.unique(mask_int_encoding, return_index=True)
        front_indices = z_near_to_far_order[unique_indices]

        # Calculate mean depth from image space and hand frame
        mean_depth_image = np.mean(valid_vertex_depth[front_indices])
        mean_depth_smpl = np.mean(z_smpl[front_indices])
        depth_offset = mean_depth_image - mean_depth_smpl

        offset_img = keypoint_img[0, 0:2] - self.camera_mat[0:2, 2]
        offset = np.concatenate([offset_img / self.focal_length * depth_offset, [depth_offset]])

        return offset
    
    def compute_wrist_pos_world(self, depth, pred_output=None):
        """
        获取包括深度信息的腕部位置
        """
        offset_mano = self.compute_3d_offset(self.keypoint_2d, self.joint_pos, self.wrist_rot, depth, pred_output=pred_output)
        offset_sapien = offset_mano #@ SAPIEN2MEDIAPIPE
        self.offset = offset_sapien
        self.wrist_pos_world += self.offset
        return self.wrist_pos_world

    def detect(self):
        """
        RGB-D 检测人手
        """
        rgb, depth = self.camera.fetch_rgb_and_depth()
        if rgb is None:
            return None, None, 0, None, None, None, None
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.detector_name == "Mediapipe":
            self.num_box, self.joint_pos, self.keypoint_2d, self.wrist_rot, self.wrist_pos_world = self.detector.detect(image_bgr)
        elif self.detector_name == "MoCap":
            self.num_box, self.joint_pos, self.keypoint_2d, self.wrist_rot, self.wrist_pos_world, pred_output = self.detector.detect(image_bgr, MocapOutput=True)
        # wrist_pos MANO坐标系中始终为0？
        self.wrist_pos_world = np.array([0, 0, 0],dtype=float)
        if self.num_box != 0:
            if self.detector_name == "Mediapipe":
                self.compute_wrist_pos_world(depth)
            elif self.detector_name == "MoCap" and pred_output is not None:
                self.compute_wrist_pos_world(depth, pred_output=pred_output)
        return image_bgr, depth, self.num_box, self.joint_pos, self.keypoint_2d, self.wrist_rot, self.wrist_pos_world

class DepthDetector:
    SUPPORT_HAND_MODE = ["Right", "Left"]
    SUPPORT_DETECTOR = ["Mediapipe", "MoCap"]
    def __init__(self, hand_mode: str, camera_mat: np.ndarray = default_camera_mat, detector = "Mediapipe", show_hand=True, need_init=True):
        if hand_mode not in self.SUPPORT_HAND_MODE:
            raise ValueError(
                f"Mode {hand_mode} is invalid. Current {len(self.SUPPORT_HAND_MODE)} mode are supported: "
                f"{self.SUPPORT_HAND_MODE} ")
        if detector == "Mocap": detector = "MoCap" 
        if detector not in self.SUPPORT_DETECTOR:
            raise ValueError(
                f"Mode {detector} is invalid. Current {len(self.SUPPORT_DETECTOR)} mode are supported: "
                f"{self.SUPPORT_DETECTOR} ")

        # Camera app 获取相机内参
        self.camera_mat = camera_mat
        self.focal_length = camera_mat[0, 0]
        print("Camera Intrinsics:", self.camera_mat)

        # Hand detection 
        # mediapipe_hand_type = "right" if hand_mode == "right_hand" else "left"
        if detector == "Mediapipe":
            self.detector_name = "Mediapipe"
            self.detector = SingleHandDetector(hand_type=hand_mode)
        elif detector == "MoCap":
            self.detector_name = "MoCap"
            self.detector = MoCapDetector(hand_type=hand_mode)

        # Offset based bbox estimation 3D位移
        self.offset = {"left_hand": np.zeros(3, dtype=np.float32), "right_hand": np.zeros(3, dtype=np.float32)}

        self.num_box = 0
        self.joint_pos = np.zeros([21,3])
        self.keypoint_2d = np.zeros([21,3])
        self.wrist_rot = np.eye(3) # camera2hand
        self.wrist_pos_world = np.array([0, 0, 0])
    
    def draw_skeleton_on_image(
        self, image, keypoint_2d, style="white", show_coord=False, keypoint_3d=None
    ):
        return self.detector.draw_skeleton_on_image(image, keypoint_2d, style=style, show_coord=show_coord, keypoint_3d=keypoint_3d)
    
    def compute_3d_offset(self, keypoint_2d, joint_pos, wrist_rot,  depth: np.ndarray, pred_output=None):
        """
        由预测得到的图像中关键点坐标+深度信息获取真实世界的三维坐标
            :keypoint_2d: [21,3]的NormalizedLandmarkList {x,y,z}
            :joint_pos: [21,3]的MANO手部21个关键点的3D坐标, 手腕为原点
            :wrist_rot: [3,3]的Camera2Hand旋转矩阵
            :depth: 深度图
            :pred_output: FrankMocap预测输出
        返回相机坐标系下的偏移
        """
        height, width = depth.shape
        # 获取图像中关键点位置
        if self.detector_name == "Mediapipe":
            keypoint_img = landmarks_to_pixel_array(keypoint_2d, (height, width))
        elif self.detector_name == "MoCap":
            keypoint_img = keypoint_2d
        # 将mano手部关节坐标转换回opencv相机坐标系
        joint_pos_smplx = joint_pos @ wrist_rot
        # 去整+防止越界
        if pred_output is None:
            mask_int = np.rint(keypoint_img[:, :2]).astype(int)
        else:
            mask_int = np.rint(pred_output["pred_vertices_img"][:, :2]).astype(int)            
        mask_int = np.clip(mask_int, [0, 0], [width - 1, height - 1])
        # mask_array = np.zeros([height, width])
        # mask_array[mask_int[:, 0], mask_int[:, 1]] = 1
        # # 膨胀
        # mask_array = self.dilate_mask(mask_array,5)
        # # 还原为点列表
        # mask_int = np.array(np.nonzero(mask_array)).T

        # Image space vertices 21个关节的深度信息, 注意y,x的顺序
        depth_vertices = depth[mask_int[:, 1], mask_int[:, 0]]
        depth_median = np.nanmedian(depth_vertices)
        # 清除离群点
        depth_valid_mask = np.nonzero(np.abs(depth_vertices - depth_median) < 0.2)[0]
        valid_vertex_depth = depth_vertices[depth_valid_mask]

        # Hand frame vertices
        v_smpl = joint_pos_smplx[depth_valid_mask] if pred_output is None else pred_output["pred_vertices_smpl"][depth_valid_mask]
        # 取相机坐标系下手部坐标的z(深度)
        z_smpl = v_smpl[:, 2]
        # 按z排序，返回从小到大的索引
        z_near_to_far_order = np.argsort(z_smpl)

        # Filter depth with same pixel pos to the front position
        # 21个关节中，有效筛选后按对应z由小到大排列
        valid_mask_int = mask_int[depth_valid_mask, :][z_near_to_far_order, :]
        mask_int_encoding = valid_mask_int[:, 0] * 1e5 + valid_mask_int[:, 1]
        _, unique_indices = np.unique(mask_int_encoding, return_index=True)
        front_indices = z_near_to_far_order[unique_indices]

        # Calculate mean depth from image space and hand frame
        mean_depth_image = np.mean(valid_vertex_depth[front_indices])
        mean_depth_smpl = np.mean(z_smpl[front_indices])
        depth_offset = mean_depth_image - mean_depth_smpl

        offset_img = keypoint_img[0, 0:2] - self.camera_mat[0:2, 2]
        offset = np.concatenate([offset_img / self.focal_length * depth_offset, [depth_offset]])

        return offset
    
    def compute_wrist_pos_world(self, depth, pred_output=None):
        """
        获取包括深度信息的腕部位置
        """
        offset_mano = self.compute_3d_offset(self.keypoint_2d, self.joint_pos, self.wrist_rot, depth, pred_output=pred_output)
        offset_sapien = offset_mano #@ SAPIEN2MEDIAPIPE
        self.offset = offset_sapien
        self.wrist_pos_world += self.offset
        return self.wrist_pos_world

    def detect(self, rgb, depth):
        """
        RGB-D 检测人手
        """
        if rgb is None:
            return None, None, 0, None, None, None, None
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.detector_name == "Mediapipe":
            self.num_box, self.joint_pos, self.keypoint_2d, self.wrist_rot, self.wrist_pos_world = self.detector.detect(image_bgr)
        elif self.detector_name == "MoCap":
            self.num_box, self.joint_pos, self.keypoint_2d, self.wrist_rot, self.wrist_pos_world, pred_output = self.detector.detect(image_bgr, MocapOutput=True)
        # wrist_pos MANO坐标系中始终为0？
        self.wrist_pos_world = np.array([0, 0, 0],dtype=float)
        if self.num_box != 0:
            if self.detector_name == "Mediapipe":
                self.compute_wrist_pos_world(depth)
            elif self.detector_name == "MoCap" and pred_output is not None:
                self.compute_wrist_pos_world(depth, pred_output=pred_output)
            # self.wrist_pos_world = self.wrist_pos_world
        return self.num_box, self.joint_pos, self.keypoint_2d, self.wrist_rot, self.wrist_pos_world

def main(detector_name:str="Mediapipe", file:str="", record_data=False):
    from loguru import logger
    import os
    import time
    if detector_name == "Mediapipe":
        detector = SingleHandDepthDetector(hand_mode="Right",detector="Mediapipe",virtual_video_file=file)
    elif detector_name == "MoCap" or "Mocap":
        detector = SingleHandDepthDetector(hand_mode="Right",detector="MoCap",virtual_video_file=file)
    else:
        logger.error("⛓️‍💥 Unsupported Detector!")
        return
    if not os.path.exists(file):
        logger.info("📷 Capture video stream from camera.")
    else:
        logger.info("🎬 Capture video stream from RGB video file.")
    # record_data = False
    if record_data: logger.info("✅ Start Recording Data...")
    data_list = []
    while True:
        image_bgr, depth, num_box, joint_pos, keypoint_2d, wrist_rot, wrist_pos = detector.detect()
        if image_bgr is None:
            logger.info("🎞️  RGB video ended, preparing to exit...")
            break
        if num_box != 0:
            image_bgr = detector.draw_skeleton_on_image(image_bgr, keypoint_2d, show_coord=True)
            # image_bgr = detector.draw_skeleton_on_image(image_bgr, keypoint_2d, show_coord=True, keypoint_3d=joint_pos@wrist_rot@SAPIEN2MEDIAPIPE.T+wrist_pos)
        
        # print wrist pos in Real World
        text = f"Wrist Pos in Camera:({wrist_pos[0]:.5f},{wrist_pos[1]:.5f},{wrist_pos[2]:.5f})"
        cv2.putText(image_bgr, text, (5, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        wrist_pos = wrist_pos @ SAPIEN2MEDIAPIPE
        text = f"Wrist Pos in Simulation:({wrist_pos[0]:.5f},{wrist_pos[1]:.5f},{wrist_pos[2]:.5f})"
        cv2.putText(image_bgr, text, (5, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("realtime_hand_detect_demo", image_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("🤖 Q pressed, preparing to exit...")
            break
        elif key == ord("x"):
            record_data = True
            logger.info("✅ Start Recording Data...")
        elif key == ord("z"):
            record_data = False
            logger.info("🛑 End Recording Data...")

        # 记录当前帧时间（绝对时间戳）
        timestamp = time.time()
        # 根据检测结果组装数据
        if joint_pos is not None:
            # 检测成功
            current_joint_pos = joint_pos
            # 将 keypoint_2d 转换为像素坐标
            h, w = image_bgr.shape[:2]
            current_keypoint_2d = landmarks_to_pixel_array(keypoint_2d, (h, w)) if detector_name == "Mediapipe" else keypoint_2d
            current_wrist_rot = wrist_rot
            current_wrist_pos = np.expand_dims(wrist_pos, axis=0)
        else:
            # 检测失败，joint_pos 和 keypoint_2d 为 None
            current_joint_pos = None
            current_keypoint_2d = None
            current_wrist_rot = None
            current_wrist_pos = None
        
        if record_data:    
            data_list.append({
            'timestamp': timestamp,
            'joint_pos': current_joint_pos,
            'keypoint_2d': current_keypoint_2d,
            'wrist_rot': current_wrist_rot,
            'wrist_pos': current_wrist_pos
        })
            
    if data_list:
        # 定义txt输出路径
        output_path = f"data/with_depth/hand_data_{time.strftime('%Y%m%d_%H%M%S')}_{detector_name}.txt"
        
        # 打开txt文件并写入
        with open(output_path, 'w', encoding='utf-8') as f:           
            # 写入逐帧数据
            for frame_idx, frame_data in enumerate(data_list):
                f.write(f"\n--- 第 {frame_idx+1} 帧 ---\n")
                for key, value in frame_data.items():
                    f.write(f"{key}: ")
                    # 处理numpy数组（格式化输出）
                    if isinstance(value, np.ndarray):
                        # 数组转为易读的字符串（保留4位小数，换行）
                        f.write("\n")
                        np.savetxt(f, value, fmt="%.4f", delimiter=",")
                    # 处理None值
                    elif value is None:
                        f.write("None (未检测到手部)\n")
                    # 处理普通数值（如timestamp、qpos）
                    else:
                        f.write(f"{value}\n")
        
        logger.info(f"💾 数据已保存到TXT文件: {output_path}")

def set_sapien_viewer(urdf_path=r"D:\study\VScodes\Retargeting\assets\robots\hands\qb\qb_right.urdf"):
    # Sapien 仿真场景搭建
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    # Setup 场景与材质
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting 光照
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera 相机与可视化
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture 机械臂模型加载，大小位姿适配
    loader = scene.create_urdf_loader()
    filepath = str(Path(urdf_path))
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(urdf_file=filepath)
    return scene, viewer, robot

def main_TO_EDIT():
    from loguru import logger
    # # 设置相机
    # app = RealsenseApp(file=r"D:\study\Grasp\B4Counting\video_rgb.mp4")
    # app.connect_to_device()

    # scene, viewer, robot = set_sapien_viewer(r"assets\robots\hands\qb\qb_right.urdf")

    # # Perception
    # detector = MoCapDetector(hand_type="Right")
    # # motion_control = Record3DSingleHandMotionControl(hand_mode="right_hand", show_hand=True, virtual_video_file=r"D:\study\Grasp\B4Counting\video_rgb.mp4")

    # # Init
    # create_robot = False
    # steps = 0
    # env_init_pos = np.array([-0.4, 0, 0.2])
    # rgb, depth = app.fetch_rgb_and_depth()
    # scene.step()

    # # Press "q" on the keyboard to exit the teleoperation when you finish
    # # The demonstration data will be automatically saved
    # while True:
    #     record_data = False
    #     data_list = []
    #     while True:
    #         rgb, depth = app.fetch_rgb_and_depth()
    #         image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    #         num_box, joint_pos, keypoint_2d, wrist_rot, wrist_pos = detector.detect(image_bgr)
    #         if num_box != 0:
    #             image_bgr = detector.draw_skeleton_on_image(image_bgr, keypoint_2d, show_coord=True, keypoint_3d=joint_pos@wrist_rot@SAPIEN2MEDIAPIPE.T+wrist_pos)
    #         cv2.imshow("realtime_hand_detect_demo", image_bgr)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             logger.info("🤖 Q pressed, preparing to exit...")
    #             break
    #         elif key == ord("x"):
    #             record_data = True
    #             logger.info("✅ Start Recording Data...")
    #         elif key == ord("z"):
    #             record_data = False
    #             logger.info("🛑 End Recording Data...")




    #     for _ in range(2):
    #         viewer.render()
    #     steps += 1

    #     if not motion_control.initialized:
    #         success, motion_data = motion_control.step()
    #         rgb = motion_data["rgb"]
    #         if not success:
    #             continue

    #         rotate_pose = sapien.Pose(q=[0.9238, 0, 0.3826, 0], p=[0.2, 0, -0.1])
    #         robot.set_pose(sapien.Pose(env_init_pos) * rotate_pose)
    #     else:
    #         if not create_robot:
    #             zero_joint_pos = motion_control.compute_hand_zero_pos()
    #             robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, np.pi / 2, 0)))
    #             create_robot = True


    #         success, motion_data = motion_control.step()
    #         rgb = motion_data["rgb"]

    #         if not success:
    #             continue

    #         root_joint_qpos = motion_control.compute_operator_space_root_qpos(motion_data)
    #         root_joint_qpos *= 1
    #         robot.set_pose(sapien.Pose(env_init_pos) ,root_joint_qpos)


    #         # if np.abs(robot.get_qpos().mean()) < 1e-5:
    #         #     robot.set_qpos(robot_qpos)

rot=np.array(
[[-0.9845,  0.0571, -0.1656],
 [-0.0352, -0.9906, -0.1322],
 [-0.1716, -0.1243,  0.9773]]
)

if __name__ == '__main__':
    # print('main')
    main(detector_name="Mocap",record_data=False,file=r"")


    # scene, viewer, robot = set_sapien_viewer(urdf_path=r"D:\study\VScodes\Retargeting\assets\robots\hands\qb\qb_right.urdf")
    # angle = 0
    # offset = 1
    # while True:
    #     # angle+=np.pi/8
    #     # offset += 0.1 # (offset-1)%2/2
    #     # angle = np.pi/2
    #     # quat = R.from_euler('xz',[np.pi/2,np.pi/2]).as_quat()
    #     rot = R.from_euler('zx',[np.pi/2,-np.pi/2]).as_matrix()
    #     print(rot)
    #     quat = R.from_matrix(rot).as_quat()
    #     quat = R.from_matrix(R.from_quat(quat).as_matrix() @ np.array([[-1,0,0],[0,-1,0],[0,0,1]])).as_quat()
    #     robot.set_pose(sapien.Pose([0, 0, 0], [quat[3],quat[0],quat[1],quat[2]]))
    #     # robot.set_pose(sapien.Pose([0, 0, -0.2], [0, 0, 0, 1]))
    #     for _ in range(2):
    #         viewer.render()
    #     time.sleep(1)
    


    # arr = np.array([
    # [176.4984,458.9069],
    # [251.9666,429.9290],
    # [305.2307,359.2115],
    # [331.4523,290.8489],
    # [363.5584,238.3885],
    # [251.5649,265.1935],
    # [278.7316,180.4878],
    # [294.5190,123.3591],
    # [305.4508,73.5699],
    # [200.6781,256.1936],
    # [210.4465,155.9750],
    # [218.0534,87.8013],
    # [221.9628,30.2069],
    # [153.9060,269.9189],
    # [140.5004,178.2453],
    # [137.0884,114.3905],
    # [135.8648,59.3106],
    # [108.9392,300.9490],
    # [83.6165,231.1305],
    # [70.4492,182.6259],
    # [61.5546,134.6606]])
    # joint_pos=np.array([
    #     [0.0000,0.0000,0.0000],
    #     [0.0092,0.0314,0.0300],
    #     [0.0160,0.0499,0.0582],
    #     [0.0244,0.0601,0.0935],
    #     [0.0197,0.0651,0.1225],
    #     [-0.0000,0.0265,0.1000],
    #     [0.0064,0.0318,0.1308],
    #     [0.0072,0.0334,0.1521],
    #     [0.0291,0.0369,0.1748],
    #     [-0.0000,0.0000,0.0996],
    #     [0.0018,-0.0017,0.1418],
    #     [0.0144,-0.0015,0.1681],
    #     [0.0307,0.0002,0.1968],
    #     [0.0101,-0.0199,0.0941],
    #     [0.0155,-0.0267,0.1263],
    #     [0.0274,-0.0281,0.1520],
    #     [0.0402,-0.0284,0.1766],
    #     [0.0169,-0.0357,0.0756],
    #     [0.0181,-0.0448,0.0989],
    #     [0.0220,-0.0509,0.1222],
    #     [0.0327,-0.0542,0.1429]
    # ])
    # depth = np.zeros([480,640])
    # camera = RealsenseApp(file=r"D:\study\Grasp\B4Counting\video_rgb.mp4")
    # camera.connect_to_device()
    # camera_mat = camera.camera_intrinsics
    # print(camera_mat)
    # focal_length = camera.camera_intrinsics[0, 0]
    # print("Camera Intrinsics:", camera_mat)
    # height, width = depth.shape
    # # 获取图像中关键点位置
    # keypoint_img = arr
    # joint_pos_smplx = joint_pos @ FRANKMOCAP2MANO
    # # 去整+防止越界
    # mask_int = np.rint(keypoint_img).astype(int)
    # mask_int = np.clip(mask_int, [0, 0], [width - 1, height - 1])
    # # mask_array = np.zeros([height, width])
    # # mask_array[mask_int[:, 0], mask_int[:, 1]] = 1
    # # # 膨胀
    # # mask_array = self.dilate_mask(mask_array,5)
    # # # 还原为点列表
    # # mask_int = np.array(np.nonzero(mask_array)).T
    # # Image space vertices 21个关节的深度信息, 注意y,x的顺序
    # depth_vertices = depth[mask_int[:, 1], mask_int[:, 0]]
    # depth_median = np.nanmedian(depth_vertices)
    # # 清除离群点
    # depth_valid_mask = np.nonzero(np.abs(depth_vertices - depth_median) < 0.2)[0]
    # valid_vertex_depth = depth_vertices[depth_valid_mask]

    # # Hand frame vertices
    # v_smpl = joint_pos_smplx[depth_valid_mask]
    # # 取SMPLX手部坐标的z(深度)
    # z_smpl = v_smpl[:, 2]
    # # 按z排序，返回从小到大的索引
    # z_near_to_far_order = np.argsort(z_smpl)

    # # Filter depth with same pixel pos to the front position
    # # 21个关节中，有效筛选后按对应z由小到大排列
    # valid_mask_int = mask_int[depth_valid_mask, :][z_near_to_far_order, :]
    # mask_int_encoding = valid_mask_int[:, 0] * 1e5 + valid_mask_int[:, 1]
    # _, unique_indices = np.unique(mask_int_encoding, return_index=True)
    # front_indices = z_near_to_far_order[unique_indices]

    # # Calculate mean depth from image space and hand frame
    # mean_depth_image = np.mean(valid_vertex_depth[front_indices])
    # mean_depth_smpl = np.mean(z_smpl[front_indices])
    # depth_offset = mean_depth_image - mean_depth_smpl

    # offset_img = keypoint_img[0, 0:2] - camera_mat[0:2, 2]
    # offset = np.concatenate([offset_img / focal_length * depth_offset, [depth_offset]])