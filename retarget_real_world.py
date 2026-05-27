import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import pickle
import os

import cv2
import numpy as np
from loguru import logger

from scipy.spatial.transform import Rotation as R

from coord_converter import *
CAMERA2WORLD = MEDIAPIPE2SAPIEN3
from motion_control import PinRobot, PinRobotController, is_pose_changed, clamp_rotation

# assembly urdf path
arm_path = r"assets/robots/assembly/xarm7_qbr/qbr.urdf"
mesh_path = r"assets/robots/assembly/xarm7_qbr"
# robot arm joint names
arm_joint_names=[
        "joint1","joint2","joint3","joint4","joint5","joint6","joint7"
    ]
arm_DoF = 7
hand_links = [
    "base_link",
    "link1", "link2", "link3", "link4", "link5",
    "link6", "link7", "link8", "link9", "link10", "link11"
]

from config import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    RetargetingConfig
)

from detector.single_hand_detector import SingleHandDetector
from teleop_hand import DepthDetector
from hand_detector.record3d_app_realsense import RealsenseApp

from real_world_controller import XArm7QB
import threading

from pynput import keyboard
# 键盘按下
current_key = None
def on_press(key):
    global current_key
    try:
        # 普通字母、数字、符号键
        char = key.char
        current_key = char.lower()  # 统一小写，不分大小写
    except AttributeError:
        # 特殊键（如方向键、功能键等）
        if key == keyboard.Key.up:
            current_key = "UP"
        elif key == keyboard.Key.down:
            current_key = "DOWN"
        elif key == keyboard.Key.left:
            current_key = "LEFT"
        elif key == keyboard.Key.right:
            current_key = "RIGHT"
        elif key == keyboard.Key.ctrl:
            current_key = "CTRL"
        elif key == keyboard.Key.space:
            current_key = "SPACE"
        else:
            current_key = None  # 其他特殊键不处理

def on_release(key):
    # 松开清空（可选）
    global current_key
    current_key = None

# 启动监听
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True  # 设置为守护线程，主程序退出时自动结束监听
listener.start()

exit_flag = multiprocessing.Value('b', False)

video_file = r""
detector_name = "Mediapipe" # "MoCap" or "Mediapipe"

default_camera_mat = np.array([
    [614.450317, 0., 332.668884],
    [0., 614.965996, 246.103592],
    [0., 0., 1.]])

def landmarks_to_pixel_array(keypoint_2d, img_shape):
    """将 MediaPipe NormalizedLandmarkList 转换为像素坐标数组 (21, 2)"""
    if keypoint_2d is None:
        return None
    h, w = img_shape[:2]
    points = []
    # 通过 .landmark 属性访问每个 landmark
    for lm in keypoint_2d.landmark:
        x = lm.x * w
        y = lm.y * h
        points.append([x, y])
    return np.array(points, dtype=np.float32)

# 传递并保存RobotName、RetargetingType、HandType 等元数据
def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str,
                      robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType,
                       camera_mat: np.ndarray=default_camera_mat, detector_type: str = "Mediapipe",
                       headless: bool = False, multi_cam: bool = False, record_video: bool = False,
                       enable_real_robot: bool = True):
    global current_key, exit_flag
    
    USING_PRESET_POSES = False
    # RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    # 加载指定的重定向配置文件
    logger.info(f"Start retargeting with config {config_path}")
    # 解析机械臂 URDF 路径、关节配置等，建立重定向优化序列
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    # 初始化手部检测器
    # detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    detector = DepthDetector(hand_mode=hand_type, camera_mat=camera_mat, detector=detector_type)

    config = RetargetingConfig.load_from_file(config_path)
    urdf_path = arm_path


    # Different robot loader may have different orders for joints
    # 从重定向配置中读取预定义的关节名称列表
    # 需要拼接 arm + hand
    retargeting_joint_names = list(arm_joint_names) + retargeting.joint_names

    # 初始化机器人位置，使其不至于陷入地面，尽量靠近实际位姿
    # 这里按照Pinocchio模型的关节顺序初始化位置，后续会通过retargeting_to_sapien映射到正确的机械臂和手部关节位置
    init_qpos = np.zeros(7+11)
    # init_qpos[:7]=[-np.pi/4,-0.32,0.06,1.49,-0.3,0.78,-0.29] # 机械臂初始位置，单位为弧度
    init_qpos[:7]=[0,0.21,0,0.9,0,-0.75,0]
    
    # 建立机器人Pinocchio模型和控制器
    pin_robot = PinRobot(urdf_path, mesh_path)
    pin_joint_names = pin_robot.joint_names
    controller = PinRobotController(pin_robot, initial_qpos=init_qpos, hand_links=hand_links, ee_link="base_link", wrist_alpha=0.65)
    init_pose = controller.pose

    hand_DoF = pin_robot.model.nq - arm_DoF
    # 重定向关节索引→Pinocchio关节索引映射数组
    retargeting_to_pin = np.array(
        [retargeting_joint_names.index(name) for name in pin_joint_names]
    ).astype(int)
    # Pinocchio关节索引→重定向关节索引映射数组
    pin_to_retargeting = np.array(
        [pin_joint_names.index(name) for name in retargeting_joint_names]
    ).astype(int)

    # 初始化轨迹策略
    last_goal_pose = None      # 上一次的目标位姿
    pose_threshold = 0.01      # 位姿变化阈值（位置变化0.05米或角度变化10度）
    rot_threshold = 10             # 角度变化阈值（单位：度）
    init_depth = 0.3 # 初始深度，需要根据实际情况调整

    # 手部位置原点，避免过于靠近底座
    # 即相机外参中的平移向量，也可以根据实际情况调整
    # 手部位置原点，避免过于靠近底座，设置为机械臂末端初始位置
    origin_pos = controller.pose.translation

    # 在 机器人加载完成后保存元数据
    metadata = {
        'robot_name': robot_name,          # 枚举值转为字符串
        'retargeting_type': retargeting_type.value,
        'hand_type': hand_type,
        'retargeting_joint_names': retargeting_joint_names,
        'pin_joint_names': pin_joint_names,
    }

    data_list = []   # 用于存储每帧数据

    # 测试腕部姿态
    id = 0
    quat_list = np.array([
        [0, 0, 0, 1],
        [0.5, -0.5, -0.5, 0.5],
        [0, 0, -0.707, 0.707],
        [0, 0, 0.707, 0.707],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [-0.707, 0, 0, 0.707],
        [0, 0.707, 0, 0.707],
        [0, -0.707, 0, 0.707],
        [0.5, -0.5, 0.5, -0.5]
    ])
    this_quat =quat_list[id%10]
    if USING_PRESET_POSES:
        logger.info(f"📝 Using Wrist Quat: {this_quat}")

    # 默认深度
    init_depth = 0.3
    depth = init_depth
    init_position = [0,0,init_depth]
    is_init = False
    last_r = R.from_euler('zyx', [-90, 0, 180], degrees=True)
    
    # 在循环开始前记录起始时间
    start_time = time.time()
    record_data = False
    take_photo = False

    # 限制腕部平移范围，避免机械臂末端过于靠近底座或伸出可达范围
    wrist_pos_limits = ([0.4, -0.14, 0.1], [0.65, 0.14, 0.55])
    object_id = 1
    i = 0

    # 真实机器人控制相关变量
    real_robot = None
    control_thread = None
    real_arm_qpos_lock = threading.Lock()
    real_hand_qpos_lock = threading.Lock()
    latest_arm_qpos = init_qpos[:7]
    latest_hand_qpos = np.zeros(6)
    reset_flag = False
    arm_pose_lock = False
    
    def real_robot_control_loop():
        """后台线程：以固定频率发送关节角到真实机器人"""
        nonlocal real_robot
        nonlocal reset_flag
        last_time = time.perf_counter()
        while not exit_flag.value:
            if reset_flag:
                if real_robot is not None:
                    real_robot.reset()
                reset_flag = False

            # 控制频率
            now = time.perf_counter()
            dt = now - last_time
            if dt < real_robot.control_dt:  # 50Hz
                time.sleep(real_robot.control_dt - dt)
            last_time = time.perf_counter()
            
            # 读取最新目标关节角
            with real_arm_qpos_lock:
                arm_qpos = latest_arm_qpos.copy()
            with real_hand_qpos_lock:
                hand_qpos = latest_hand_qpos.copy()
            
            if real_robot is not None:
                try:
                    if real_robot.use_arm:
                        real_robot.control_arm_qpos(arm_qpos)
                    if real_robot.use_hand:
                        real_robot.control_hand_qpos(hand_qpos)
                except Exception as e:
                    logger.error(f"Real robot control error: {e}")
                    # 出现错误后停止真实机器人控制
                    if real_robot is not None:
                        real_robot.stop()
                    break

    if enable_real_robot:
        logger.info("Initializing real robot (xArm7 + QB Hand)...")
        try:
            real_robot = XArm7QB(
                arm_ip_address="192.168.1.204",   # 根据实际修改
                hand_port="COM10",                 # 根据实际修改
                use_arm=True,
                use_hand=True,                    # 如果不需要手部可改为 False
                initial_arm_qpos=init_qpos[:7].tolist()
            )
            real_robot.start()   # 启动机械臂内部控制线程（如果有）
            # 启动我们自己的控制线程
            control_thread = threading.Thread(target=real_robot_control_loop, daemon=True)
            control_thread.start()
            logger.info("Real robot controller started.")
            if real_robot.use_arm:
                init_arm_pos = real_robot.get_arm_qpos()  # 实际机械臂位置
                with real_arm_qpos_lock:
                    latest_arm_qpos[:] = init_arm_pos
        except Exception as e:
            logger.error(f"Failed to initialize real robot: {e}")
            enable_real_robot = False

    try:
        while True:
            try:
                rgb, depth = queue.get(timeout=5)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Empty:
                logger.error(
                    "Fail to fetch image from camera in 5 secs. Please check your web camera device."
                )
                return

            # joint_pos：人手关节的 3D 空间位置数组
            # keypoint_2d：人手关节的 2D 像素坐标
            _, joint_pos, keypoint_2d, wrist_rot, wrist_pos = detector.detect(rgb, depth)
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            if bgr is None:
                logger.info("🎞️  RGB video ended, preparing to exit...")
                exit_flag.value = True
                break
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # text = f"Wrist Pos in Camera:({wrist_pos[0]:.5f},{wrist_pos[1]:.5f},{wrist_pos[2]:.5f})"
            # cv2.putText(bgr, text, (5, 30),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # text = f"Wrist Pos in Simulation:({wrist_pos[2]+origin_pos[0]:.5f},{-wrist_pos[0]+origin_pos[1]:.5f},{-wrist_pos[1]+origin_pos[2]:.5f})"
            # cv2.putText(bgr, text, (5, 60),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("realtime_retargeting_demo", cv2.flip(bgr, 1)) # 左右翻转以匹配镜像输入
            cv2.waitKey(1)
            
            # 在这里获取方向键
            if current_key == "q":
                logger.info("🤖 Q pressed, preparing to exit...")
                current_key = None
                break
            elif current_key == "x" and not record_data:
                record_data = True
                start_session = time.strftime('%Y%m%d_%H%M%S')
                logger.info("✅ Start Recording Data...")
                current_key = None
            elif current_key == "z" and record_data:
                record_data = False
                logger.info("🛑 End Recording Data...")
                current_key = None
            elif current_key == "e" and USING_PRESET_POSES:
                id += 1
                this_quat = quat_list[id%10]
                logger.info(f"📝 Using Wrist Quat: {this_quat}")
                current_key = None
            elif current_key == "f":
                USING_PRESET_POSES = not USING_PRESET_POSES
                logger.info(f"📝 Using Pre-set Wrist Quat is set {USING_PRESET_POSES}")
                if USING_PRESET_POSES:
                    logger.info(f"📝 Using Wrist Quat: {this_quat}")
                current_key = None
            elif current_key == "r" or current_key == "t":
                controller.reset(init_qpos)
                depth = init_depth
                if current_key == "t":
                    object_id = (object_id % 21) + 1
                origin_pos = controller.pose.translation
                is_init = False
                logger.info("🔧 Controller reset...")
                reset_flag = True
                current_key = None
            elif current_key == "SPACE":
                origin_pos[2] += 0.01
                origin_pos[2] = min(origin_pos[2], wrist_pos_limits[1][2]) # 限制z轴最大值
                logger.info(f"⬆️  Move Up (+z): {origin_pos[2]:.2f}m")
                current_key = None
            elif current_key == "c":
                origin_pos[2] -= 0.01
                origin_pos[2] = max(origin_pos[2], wrist_pos_limits[0][2]) # 限制z轴最小值
                logger.info(f"⬇️  Move Down (-z): {origin_pos[2]:.2f}m")
                current_key = None
            elif current_key == "UP":
                origin_pos[0] += 0.01
                origin_pos[0] = min(origin_pos[0], wrist_pos_limits[1][0]) # 限制x轴最大值
                logger.info(f"⤴️  Move Forward (+x): {origin_pos[0]:.2f}m")
                current_key = None
            elif current_key == "DOWN":
                origin_pos[0] -= 0.01
                origin_pos[0] = max(origin_pos[0], wrist_pos_limits[0][0]) # 限制x轴最小值
                logger.info(f"⤵️  Move Backward (-x): {origin_pos[0]:.2f}m")
                current_key = None                
            elif current_key == "LEFT":
                origin_pos[1] += 0.01
                origin_pos[1] = min(origin_pos[1], wrist_pos_limits[1][1]) # 限制y轴最大值
                logger.info(f"⬅️  Move Left (+y): {origin_pos[1]:.2f}m")
                current_key = None
            elif current_key == "RIGHT":
                origin_pos[1] -= 0.01
                origin_pos[1] = max(origin_pos[1], wrist_pos_limits[0][1]) # 限制y轴最小值
                logger.info(f"➡️  Move Right (-y): {origin_pos[1]:.2f}m")
                current_key = None 
            elif current_key == "v":
                take_photo = True
                logger.info("📸 Taking photo...")
                current_key = None
            elif current_key == "i":
                is_init = False
                logger.info("🔄 Re-initializing origin position...")
                current_key = None   
            elif current_key == "y":
                if data_list:
                    output_path = f"data/teleop_data/{object_id}/Sequence_{start_session}.pkl"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        pickle.dump({'metadata': metadata, 'object_name': object_id, 'data': data_list}, f)
                    logger.info(f"💾 Data saved to {output_path}")
                    data_list.clear()  # 清空数据列表，准备下一轮记录
                else:
                    logger.warning("❌ No data to save. Press 'X' to start recording data.")
                record_data = False
                current_key = None
            elif current_key == "g":
                data_list.clear()
                record_data = False
                import shutil
                shutil.rmtree(f"data/teleop_data/{object_id}/{start_session}", ignore_errors=True)  # 删除整个目录及其内容
                logger.info("🗑️ Data cleared. Press 'X' to start recording new data.")
                current_key = None
            elif current_key == "l" and last_goal_pose is not None:
                arm_pose_lock = True
                logger.info(f"Arm pose lock is set{arm_pose_lock}")
                current_key = None
            elif current_key == "u":
                arm_pose_lock = False
                logger.info(f"Arm pose lock is set{arm_pose_lock}")
                current_key = None
            elif current_key == "h":
                logger.info("📖 Help Menu:")
                logger.info("  Q: Quit")
                logger.info("  X: Start Recording Data")
                logger.info("  Z: Stop Recording Data")
                logger.info("  Y: Save Recorded Data")
                logger.info("  G: Clear Recorded Data")
                logger.info("  E: Cycle through Pre-set Wrist Poses (if enabled)")
                logger.info("  F: Toggle Pre-set Wrist Poses")
                logger.info("  R: Reset Controller and Object Pose")
                logger.info("  T: Reset and Switch YCB Object")
                logger.info("  Arrow Keys: Adjust Origin Position")
                logger.info("  Space: Move Origin Up, C: Move Origin Down")
                logger.info("  I: Re-initialize Origin Position")
                logger.info("  V: Take Photo")
                current_key = None
            

            if joint_pos is None:
                J = False # logger.warning(f"{hand_type} hand is not detected.")
            else:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                # 输入人手关节参考值（位置/向量），输出机械臂各关节应达到的角度（qpos）
                hand_qpos = retargeting.retarget(ref_value) # retarget_from_video.SeqRetargeting.Retarget->optimizer.retarget

                # --- ARM START---
                # 引入机械臂
                current_arm_qpos = init_qpos[:arm_DoF]
                # # --- 获取目标腕部位姿（方向四元数需要修正） ---

                # camera2world
                wrist_rot = wrist_rot @ CAMERA2WORLD.T
                # world2camera
                wrist_rot = wrist_rot.T

                # 修正初始位置
                wrist_rot_right = wrist_rot @ HAND_POSE_FIX
                # 修正curobo初始姿态
                wrist_rot_right = wrist_rot @ CUROBO_POSE_FIX
                
                # 将旋转矩阵转换为四元数
                r = R.from_matrix(wrist_rot_right) 

                # # 裁剪旋转，避免机械臂末端过于倾斜导致IK失败
                # r = clamp_rotation(r) if clamp_rotation(r) is not None else last_rot
                # last_rot = r

                wrist_quat = r.as_quat()  # 格式: [x, y, z, w]

                # 测试
                if USING_PRESET_POSES:
                    # wrist_quat = this_quat
                    wrist_quat = R.from_matrix(R.from_quat(this_quat).as_matrix() @ CUROBO_POSE_FIX).as_quat()

                if not is_init:
                    init_position = wrist_pos @ CAMERA2WORLD.T
                    is_init = True

                # 目标腕部姿态 [x, y, z, qw, qx, qy, qz]
                wrist_pos_m = wrist_pos @ CAMERA2WORLD.T + origin_pos - init_position
                wrist_pos_m = np.clip(wrist_pos_m, wrist_pos_limits[0], wrist_pos_limits[1]) # 限制目标位置在机械臂可达范围内
                if not arm_pose_lock:
                    goal_pose = [wrist_pos_m[0], wrist_pos_m[1], wrist_pos_m[2], wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2]]
                # print(wrist_quat)
                # --- 判断是否需要重新规划 ---
                need_new_traj = False
                if last_goal_pose is None:
                    need_new_traj = True
                else:
                    # 检查位置变化
                    # need_new_traj = is_pose_changed(controller.pose.translation, controller.pose.rotation, 
                    #                                 wrist_pos_m, wrist_rot_right, 
                    #                                 pos_threshold=pose_threshold, rot_threshold=rot_threshold)
                    # pos_change = np.linalg.norm(np.array(goal_pose[:3]) - np.array(last_goal_pose[:3]))
                    pos_change = np.linalg.norm(np.array(goal_pose[:3]) - np.array(controller.pose.translation))
                    need_new_traj = pos_change > pose_threshold

                if need_new_traj:
                    # 获取当前机械臂关节角度作为起始
                    current_arm_qpos = controller.qpos[:arm_DoF]
                    # 先不加入手部qpos
                    current_qpos = np.hstack((current_arm_qpos, np.zeros(hand_DoF)))
                    # 转换到 Pinocchio 关节顺序
                    # IK计算新的机械臂qpos
                    new_qpos = controller.update(wrist_pos_m, wrist_rot_right, check_collision=True)
                    arm_qpos = new_qpos[pin_to_retargeting][:arm_DoF]
                    last_goal_pose = goal_pose


                # 合并机械臂-灵巧手关节位置
                qpos = np.hstack((arm_qpos,hand_qpos))

                # ---------------- 更新真实机器人目标关节角 ----------------
                pin_qpos = qpos[retargeting_to_pin]
                if enable_real_robot and real_robot is not None:
                    arm_target = pin_qpos[:7]
                    arm_target[0] = min(max(arm_target[0],-np.pi/3),np.pi/3) # 防止大幅度旋转
                    # QB Hand 只有6个自由度
                    # hand joint:'j1', 'j2', 'j3', 'j10', 'j11', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9'
                    # 需要的6个角度是大拇指旋转、拇指弯曲、食指弯曲、中指弯曲、无名指弯曲、小指弯曲
                    # 对应 'j1', 'j2', 'j4', 'j6', 'j8', 'j10'，在 Pinocchio 模型中索引分别是 0,1,5,7,9,3
                    free_indices = [0,1,5,7,9,3]
                    hand_target = pin_qpos[-11:][free_indices]
                    with real_arm_qpos_lock:
                        latest_arm_qpos[:] = arm_target
                    with real_hand_qpos_lock:
                        latest_hand_qpos[:] = hand_target

            # 记录当前帧时间（绝对时间戳）
            timestamp = time.time()
            # print(timestamp)
            # 根据检测结果组装数据
            if joint_pos is not None:
                # 检测成功
                # 手部检测结果
                # 将 keypoint_2d 转换为像素坐标
                h, w = rgb.shape[:2]
                current_keypoint_2d = landmarks_to_pixel_array(keypoint_2d, (h, w)) if detector_type == "Mediapipe" else keypoint_2d[:,:2]# 手部关键点像素坐标
                current_joint_pos = joint_pos  # 手部关节的3D空间位置
                current_cam_wrist_rot = wrist_rot  # 腕部旋转矩阵(相机坐标系)
                current_cam_wrist_pos = wrist_pos  # 腕部位置（相机坐标系）
                # 重定向结果
                current_ref_value = ref_value # 重定向参考值（位置或向量）
                current_target_qpos = qpos # 机械臂目标关节角度(retargeting关节顺序)
                current_target_wrist_rot = wrist_rot_right
                current_target_wrist_pos = np.expand_dims(wrist_pos_m, axis=0)
            else:
                # 检测失败，手部检测结果和重定向结果为 None，仿真数据取当前机械臂状态
                current_keypoint_2d = None
                current_joint_pos = None
                current_cam_wrist_rot = None
                current_cam_wrist_pos = None
                current_ref_value = None
                current_target_qpos = None
                current_target_wrist_rot = None
                current_target_wrist_pos = None

            if enable_real_robot and real_robot is not None:
                # 记录真实机器人当前状态
                pass
                # with real_arm_qpos_lock:
                #     current_real_arm_qpos = real_robot.get_arm_qpos() if real_robot.use_arm else None
                # with real_hand_qpos_lock:
                #     current_real_hand_qpos = real_robot.get_hand_qpos() if real_robot.use_hand else None
            else:
                current_real_arm_qpos = None
                current_real_hand_qpos = None

            
            if record_data:
                # 保存图像
                if record_video:
                    frame_id = int(timestamp * 1000)
                    rgb_path = f"data/teleop_data/{object_id}/{start_session}/rgb_{frame_id}.jpg"
                    depth_path = f"data/teleop_data/{object_id}/{start_session}/depth_{frame_id}.png"
                     # 保存 RGB（压缩为 JPEG）
                    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                    # 保存深度图（16位 PNG，需要归一化到毫米量级或保持原始 float）
                    # 假设 depth 是 float32 单通道，单位米，转为毫米 0~65535 范围（通常深度范围 0~10m）
                    depth_mm = (depth * 1000).astype(np.uint16)
                    cv2.imwrite(str(depth_path), depth_mm)
                else:
                    rgb_path = None
                    depth_path = None
                data_list.append({
                'timestamp': timestamp,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'keypoint_img': current_keypoint_2d,
                'joint_pos': current_joint_pos,
                'cam_wrist_rot': current_cam_wrist_rot,
                'cam_wrist_pos': current_cam_wrist_pos,
                'ref_value': current_ref_value,
                'target_qpos': current_target_qpos,
                'target_wrist_rot': current_target_wrist_rot,
                'target_wrist_pos': current_target_wrist_pos,
                'real_arm_qpos': current_real_arm_qpos,
                'real_hand_qpos': current_real_hand_qpos
                })

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # ===== 强制关闭所有窗口 =====
        cv2.destroyAllWindows()
        logger.info("🗃️  OpenCV windows closed.")


def produce_frame(queue: multiprocessing.Queue, virtual_video_file: str = ""):
    # 在子进程中创建RealsenseApp实例并连接相机
    camera = RealsenseApp(file=virtual_video_file)
    camera.connect_to_device()
    while not exit_flag.value:
        rgb, depth = camera.fetch_rgb_and_depth()
        time.sleep(1 / 30.0)
        if rgb is None:
            break
        # 防止队列满了卡死
        if not queue.full():
            queue.put((rgb, depth))

        # 以30/2 FPS的速度获取图像，防止过快导致CPU占用过高
        time.sleep(1/30.0)
    
    # 退出时释放相机
    logger.info("✅ 相机进程已退出")


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    virtual_video_file: str = "",
    camera_mat: np.ndarray = default_camera_mat,
    detector_type: str = "Mediapipe",
    headless: bool = False,
    multi_cam: bool = False,
    record_video: bool = False
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = RetargetingConfig._DEFAULT_URDF_DIR

    queue = multiprocessing.Queue(maxsize=10)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, virtual_video_file))

    # 对应消费者进程修改
    consumer_process = multiprocessing.Process(
    target=start_retargeting,
    args=(queue, str(robot_dir), str(config_path), robot_name, retargeting_type, hand_type, camera_mat, detector_type, headless, multi_cam, record_video)
)

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")

# RobotName: allegro; shadow; svh; leap; ability; inspire; panda
# RetargetingType: vector; position; dexpilot
# HandType: left; right
# Detector_type: MoCap; Mediapipe
if __name__ == "__main__":
    main(
        robot_name=RobotName.mimic_qb,
        retargeting_type=RetargetingType.dexpilot,
        hand_type=HandType.left,
        camera_path=None,
        virtual_video_file=video_file,
        camera_mat=default_camera_mat,
        detector_type=detector_name,
        headless=False,
        multi_cam=False,
        record_video = False
    )
