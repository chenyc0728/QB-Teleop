import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import pickle
import os

import cv2
import numpy as np
import sapien
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from scipy.spatial.transform import Rotation as R

from coord_converter import *
CAMERA2WORLD = MEDIAPIPE2SAPIEN3
from motion_control import PinRobot, PinRobotController, is_pose_changed, clamp_rotation
from sapien_demos.create_actors import create_box, create_sphere, create_capsule, create_table, create_mesh, load_YCB_object
from sapien_demos.camera import compute_camera_pose

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
                       headless: bool = False, multi_cam: bool = False, record_video: bool = False):
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

    # Sapien 仿真场景搭建
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup 场景与材质
    scene = sapien.Scene()
    # render_mat = sapien.render.RenderMaterial()
    # render_mat.base_color = [0.06, 0.08, 0.12, 1]
    # render_mat.metallic = 0.0
    # render_mat.roughness = 0.9
    # render_mat.specular = 0.8
    # scene.add_ground(0, render_material=render_mat, render_half_size=[1000, 1000])

    # 地面材质：加深加暗，突出白色机器手
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.02, 0.03, 0.05, 1]  # 近黑色地面
    render_mat.metallic = 0.0
    render_mat.roughness = 0.95
    render_mat.specular = 0.1
    scene.add_ground(0, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting 光照
    # scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    # scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    # scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    # scene.set_environment_map(
    #     create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    # )
    # scene.add_area_light_for_ray_tracing(
    #     sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    # )

    scene.set_environment_map(r"assets/gray_envmap.hdr")  # 添加环境贴图，提供全局光照和反射信息

    # scene.set_ambient_light([0.2, 0.2, 0.2])
    # scene.add_directional_light([1, -0.5, -1.5], [0.8, 0.8, 0.8])
    # scene.add_directional_light([-0.5, 0.5, -1], [0.3, 0.3, 0.3])

    # 环境光：轻微提升至 0.1，保持背景干净又不至于死黑
    scene.set_ambient_light([0.10, 0.10, 0.10])

    # 主光源（右前上方）：核心照明，阴影清晰，暖色木纹+白色机器人大气质感
    scene.add_directional_light(
        direction=[0.6, -1.2, -1.0],
        color=[1.6, 1.45, 1.3],  # 暖色白光，适配木纹桌
        shadow=True  # 仅主光开阴影，性能+效果最佳
    )

    # 补光+轮廓光二合一（左后上方）：消除死黑+勾勒白色机械臂轮廓，一物两用
    scene.add_directional_light(
        direction=[-0.6, 0.8, -1.2],
        color=[0.8, 0.75, 0.7],  # 柔和补光，保留立体感
        shadow=False
    )

    # Camera 相机与可视化
    # 0: 默认主视角 (观测整体)
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    # cam.set_local_pose(sapien.Pose([1, 0, 0.3], [0, 0, 0, -1]))
    # cam.set_local_pose(sapien.Pose([0.25, 0.25, 1.3], [0.947, -0.05, 0.254, -0.188])) # 从+x轴朝下观测
    cam.set_local_pose(sapien.Pose([0.35, 0.35, 1], [0.924, 0, 0, -0.383])) 

    # 1: 前视角 (正对手指，判断左右上下)
    front_cam = scene.add_camera(
        name="Front!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    front_cam.set_local_pose(sapien.Pose([1.1, 0.25, 1], [0.195, 0, 0, -0.981]))

    # 2: 俯视图 (上帝视角，左右前后)
    top_cam = scene.add_camera(
        name="Top!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    top_cam.set_local_pose(sapien.Pose([0.4, 0.1, 1.5], [0.831, 0, 0.555, 0]))

    # 3: 侧视角 (看 Y/Z 平面，判断上下前后)
    side_cam = scene.add_camera(
        name="Side!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    side_cam.set_local_pose(sapien.Pose([0.6, 0.5, 1], [0.707, 0, 0, -0.707]))

    cameras = {
        "main": cam,
        "front": front_cam,
        "top": top_cam,
        "side": side_cam
    }

    if not headless:
        viewer = Viewer()
        viewer.window.resize(1280,720)
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(top_cam.get_local_pose())

    """
        HAND LOADER
        需要得到重定向关节索引→Sapien 关节索引映射数组
    """
    # # Load robot and set it to a good pose to take picture 机械臂模型加载，大小位姿适配
    # ---- 加载xarm7-qb机器人 ----
    # 需要提供URDF文件路径
    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True
    # urdf_path = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qb\xarm7_qb_right_hand.urdf" #"D:\study\VScodes\curobo\src\curobo\content\assets\robot\ur_description\ur5e.urdf"  # 请替换为实际路径
    # urdf_path = r"D:\study\Grasp\URDF\xarm_qb\xarmqb.urdf"
    urdf_path = arm_path
    loader.scale = 1 # 0.4 # 模型大小

    loader.collision_material = scene.create_physical_material(
        static_friction=3.0,    # 静摩擦（越大越不滑）
        dynamic_friction=2.0,   # 动摩擦
        restitution=0.0         # 不弹
    )

    robot = loader.load(urdf_path)
    # 将机器人整体向左平移 x:前 y:左 z:上
    robot.set_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))  # 根据需要调整平移量

    # 末端挂载相机
    mounted_camera = scene.add_mounted_camera(
        name="mounted_camera",
        mount=robot.links[11].entity,  # 将相机挂载在机械臂末端链接上
        pose=sapien.Pose(p=[0, 0.001, -0.075],q=[0,0.707,0.707,0]),
        width=600,
        height=600,
        fovy=1,
        near=0.1,
        far=10,
    )
    cameras["mounted"] = mounted_camera  # 末端挂载相机

    # Different robot loader may have different orders for joints
    # 读取当前加载的机械臂活跃关节名称列表
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    # print(sapien_joint_names)
    # 从重定向配置中读取预定义的关节名称列表
    # 需要拼接 arm + hand
    retargeting_joint_names = list(arm_joint_names) + retargeting.joint_names
    # retargeting_joint_names = retargeting.joint_names
    # 重定向关节索引→Sapien 关节索引映射数组
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)
    print("Retargeting_to_sapien index obtained: ",retargeting_to_sapien)

    # 初始化机器人位置，使其不至于陷入地面，尽量靠近实际位姿
    # 这里按照Pinocchio模型的关节顺序初始化位置，后续会通过retargeting_to_sapien映射到正确的机械臂和手部关节位置
    init_qpos = np.zeros(robot.dof)
    # init_qpos[:7]=[-np.pi/4,-0.32,0.06,1.49,-0.3,0.78,-0.29] # 机械臂初始位置，单位为弧度
    init_qpos[:7]=[0,0.21,0,0.9,0,-0.75,0]
    robot.set_qpos(init_qpos[retargeting_to_sapien])
    
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
    # Sapien 关节索引→Pinocchio关节索引映射数组
    sapien_to_pin = np.array(
        [sapien_joint_names.index(name) for name in pin_joint_names]
    ).astype(int)

    # 初始化手臂关节角度
    arm_qpos = robot.get_qpos()[:arm_DoF]
    # arm_qpos = np.zeros(arm_DoF,)

    # 初始化Sapien中机械臂控制器
    active_joints = robot.get_active_joints()
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=1000, damping=100, force_limit=1000, mode="force")
        joint.set_drive_target(init_qpos[retargeting_to_sapien][joint_idx])

    # 创建仿真环境中的actor（如桌子、物体等），提供视觉参考和交互对象
    table_size = 1.15
    table_height = 0.8
    table_center = [0.36, 0, table_height]  # 根据需要调整桌子位置
    table_thickness = 0.1
    object_table_height = 0.1
    object_center = [0.5+0.18, 0-0.03, table_height+object_table_height+0.05]
    ycb_id = 15

    # object = create_box(
    #     scene,
    #     sapien.Pose(p=[table_center[0]-5*sphere_radius, table_center[1], table_height+0.05]),
    #     half_size=[box_half_size, box_half_size, box_half_size],
    #     color=[1.0, 0.0, 0.0],
    #     name="box",
    # )
    object_table = create_table(
        scene,
        sapien.Pose(p=[object_center[0], object_center[1], table_height+0.05]),
        size=0.3,
        height=object_table_height,
        thickness=table_thickness,
        color=(0.7, 0.6, 0.4),
        mass = 300.0,
        is_kinematic=True
    )
    object = load_YCB_object(
        scene,
        pose=sapien.Pose(p=[object_center[0], object_center[1], object_center[2]],q=[1,0,0,0]),
        size=0.5,
        category_id=ycb_id,
        static_friction=30.0,
        dynamic_friction=20.0,
    )
    # object = load_YCB_object(
    #     scene,
    #     pose=sapien.Pose(p=[object_center[0], object_center[1], object_center[2]],q=[1,0,0,0]),
    #     size=1.3,
    #     category_id=10,
    #     static_friction=30.0,
    #     dynamic_friction=20.0,
    # )
    object_init_pose = object.get_pose()
    table = create_table(
        scene,
        sapien.Pose(p=table_center),
        size=table_size,
        height=table_height,
        thickness=table_thickness,
        is_kinematic=True
    )
    robot.set_root_pose(sapien.Pose(p=[0.0, 0.0, table_height]))

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
        'sapien_joint_names': sapien_joint_names,
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
    i = 0

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
            cv2.imshow("realtime_retargeting_demo", cv2.resize(cv2.flip(bgr, 1),None,fx=0.7,fy=0.7)) # 左右翻转以匹配镜像输入
            
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
                robot.set_qpos(controller.qpos[retargeting_to_sapien])  # 重置后立即更新机械臂姿态

                # 重置后重新设置PD控制器的目标位置，确保机械臂回到初始位置
                for joint_idx, joint in enumerate(active_joints):
                    joint.set_drive_target(init_qpos[retargeting_to_sapien][joint_idx])

                if current_key == "t":
                    ycb_id = (ycb_id+1) % 21  # 假设有21个YCB对象
                    scene.remove_actor(object)
                    object = load_YCB_object(
                            scene,
                            pose=sapien.Pose(p=[object_center[0], object_center[1], object_center[2]],q=[1,0,0,0]),
                            category_id=ycb_id,
                            static_friction=30.0,
                            dynamic_friction=20.0,
                        )
                    logger.info(f"🔄 Switched to YCB Object : {YCB_CLASSES[ycb_id]}")
                object.set_pose(object_init_pose)
                object_table.set_pose(sapien.Pose(p=[object_center[0], object_center[1], table_height+0.05]))
                origin_pos = controller.pose.translation
                table.set_pose(sapien.Pose(p=table_center))
                is_init = False
                logger.info("🔧 Controller reset...")
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
            elif current_key == "1" and not headless:
                viewer.set_camera_pose(cam.get_local_pose())
                current_key = None
            elif current_key == "2" and not headless:
                viewer.set_camera_pose(front_cam.get_local_pose())
                current_key = None
            elif current_key == "3" and not headless:
                viewer.set_camera_pose(top_cam.get_local_pose())
                current_key = None
            elif current_key == "4" and not headless:
                viewer.set_camera_pose(side_cam.get_local_pose())
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
                    output_path = f"data/teleop_data/{YCB_CLASSES[ycb_id]}/Sequence_{start_session}.pkl"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        pickle.dump({'metadata': metadata, 'object_name': YCB_CLASSES[ycb_id], 'data': data_list}, f)
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
                shutil.rmtree(f"data/teleop_data/{YCB_CLASSES[ycb_id]}/{start_session}", ignore_errors=True)  # 删除整个目录及其内容
                logger.info("🗑️ Data cleared. Press 'X' to start recording new data.")
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
                logger.info("  1-4: Switch Camera Views (if not headless)")
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
                current_arm_qpos = robot.get_qpos()[:arm_DoF]
                # # --- 获取目标腕部位姿（方向四元数需要修正） ---

                # camera2world
                wrist_rot = wrist_rot @ CAMERA2WORLD.T
                # world2camera
                wrist_rot = wrist_rot.T

                # 修正初始位置
                wrist_rot_right = wrist_rot @ HAND_POSE_FIX
                # 修正curobo初始姿态
                wrist_rot_right = wrist_rot @ CUROBO_POSE_FIX
                # wrist_rot_right = ROTATE_Z @ wrist_rot_right
                if urdf_path == r"D:\study\Grasp\URDF\xarm_qb\xarmqb.urdf":
                    wrist_rot_right = wrist_rot_right @ ROTATE_Y
                
                # 将旋转矩阵转换为四元数
                r = R.from_matrix(wrist_rot_right) 

                # # 裁剪旋转，避免机械臂末端过于倾斜导致IK失败
                # r = clamp_rotation(r) if clamp_rotation(r) is not None else last_rot
                # last_rot = r

                wrist_quat = r.as_quat()  # 格式: [x, y, z, w]

                # 测试
                if USING_PRESET_POSES:
                    # wrist_quat = this_quat
                    if urdf_path != r"D:\study\Grasp\URDF\xarm_qb\xarmqb.urdf" and urdf_path != r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr\qbr.urdf":
                        wrist_quat = R.from_matrix(R.from_quat(this_quat).as_matrix() @ CUROBO_POSE_FIX).as_quat()
                    else:
                        wrist_quat = R.from_matrix(R.from_quat(this_quat).as_matrix() @ CUROBO_POSE_FIX @ ROTATE_Y).as_quat()

                # 在 retargeting 循环内，获取 wrist_rot 后立即打印
                # print("原始 wrist_rot:\n", wrist_rot_o)
                # print("变换后 wrist_rot:\n", CAMERA2WORLD @ wrist_rot.T)
                # print("四元数 (w,x,y,z):", wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2])

                if not is_init:
                    init_position = wrist_pos @ CAMERA2WORLD.T
                    is_init = True

                # 目标腕部姿态 [x, y, z, qw, qx, qy, qz]
                wrist_pos_m = wrist_pos @ CAMERA2WORLD.T + origin_pos - init_position
                wrist_pos_m = np.clip(wrist_pos_m, wrist_pos_limits[0], wrist_pos_limits[1]) # 限制目标位置在机械臂可达范围内
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
                    current_arm_qpos = robot.get_qpos()[:arm_DoF]
                    # 先不加入手部qpos
                    current_qpos = np.hstack((current_arm_qpos, np.zeros(hand_DoF)))
                    # 转换到 Pinocchio 关节顺序
                    # IK计算新的机械臂qpos
                    new_qpos = controller.update(wrist_pos_m, wrist_rot_right, check_collision=True)
                    arm_qpos = new_qpos[pin_to_retargeting][:arm_DoF]
                    last_goal_pose = goal_pose


                # 合并机械臂-灵巧手关节位置
                qpos = np.hstack((arm_qpos,hand_qpos))

                # 将计算出的关节角度按关节映射数组重排后，赋值给仿真机械臂更新姿态
                # robot.set_qpos(qpos[retargeting_to_sapien])

                # 更新PD控制器的目标位置
                for joint_idx, joint in enumerate(active_joints):
                    joint.set_drive_target(qpos[retargeting_to_sapien][joint_idx])

            for _ in range(2):
                scene.step()
                scene.update_render()
                if not headless:
                    viewer.render()
                    i += 1

                if multi_cam or take_photo or i % 3 == 0:
                    # 渲染所有相机图像
                    images = {}
                    for name, cams in cameras.items():
                        if name == "top":
                            continue  # 跳过主相机
                        cams.take_picture()
                        rgba = cams.get_picture("Color")  # [H, W, 4]
                        cam_bgr = cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        images[name] = cam_bgr
                    
                    # 拼接图像（2x2网格）
                    top_row = np.hstack([images["mounted"], images["main"]])
                    bottom_row = np.hstack([images["side"], images["front"]])
                    combined = np.vstack([top_row, bottom_row])
                    
                    # 添加视角标签
                    cv2.putText(combined, "Mounted View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(combined, "Main View", (620, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(combined, "Side View", (20, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(combined, "Front View", (620, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    
                    # 显示
                    combined = cv2.resize(combined,None,fx=0.7,fy=0.7)
                    cv2.imshow("Multi-View Grasping", combined)
                    cv2.waitKey(1) 

                    take_photo = False


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
                # 仿真数据（机械臂当前关节角度、末端位姿等）
                current_qpos = robot.get_qpos() # 机械臂当前关节角
                current_qf = robot.get_qf() # 机械臂当前关节力矩
                current_wrist_rot = robot.links[11].get_pose().q # 机械臂末端当前旋转(qw,qx,qy,qz)
                current_wrist_pos = robot.links[11].get_pose().p # 机械臂末端当前位移(x,y,z)
                current_object_rot = object.get_pose().q # 物体当前旋转(qw,qx,qy,qz)
                current_object_pos = object.get_pose().p # 物体当前位移(x,y,z)
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
                                
                current_qpos = robot.get_qpos() # 机械臂当前关节角
                current_qf = robot.get_qf() # 机械臂当前关节力矩
                current_wrist_rot = robot.links[11].get_pose().q # 机械臂末端当前旋转(qw,qx,qy,qz)
                current_wrist_pos = robot.links[11].get_pose().p # 机械臂末端当前位移(x,y,z)
                current_object_rot = object.get_pose().q # 物体当前旋转(qw,qx,qy,qz)
                current_object_pos = object.get_pose().p # 物体当前位移(x,y,z)
            
            if record_data:
                # 保存图像
                if record_video:
                    frame_id = int(timestamp * 1000)
                    rgb_path = f"data/teleop_data/{YCB_CLASSES[ycb_id]}/{start_session}/rgb_{frame_id}.jpg"
                    depth_path = f"data/teleop_data/{YCB_CLASSES[ycb_id]}/{start_session}/depth_{frame_id}.png"
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
                'qpos': current_qpos,
                # 'qf': current_qf,
                'wrist_rot': current_wrist_rot,
                'wrist_pos': current_wrist_pos,
                'object_rot': current_object_rot,
                'object_pos': current_object_pos
                })

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # ===== 强制关闭所有窗口 =====
        cv2.destroyAllWindows()
        if not headless:
            viewer.close()  # 关闭Sapien窗口
        logger.info("🗃️  Viewer & OpenCV windows closed.")
        # # 保存数据
        # import pickle
        # if data_list:
        #     output_path = f"data/retargeting_data_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        #     with open(output_path, 'wb') as f:
        #         pickle.dump({'metadata': metadata, 'data': data_list}, f)
        #     logger.info(f"Data saved to {output_path}")
        # ========== 核心修改：替换pickle为txt写入 ==========
        if data_list:
            # 定义txt输出路径
            output_path = f"data/retargeting_data/retargeting_data_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            # 打开txt文件并写入
            with open(output_path, 'w', encoding='utf-8') as f:
                # 1. 写入元数据
                f.write("="*50 + " 元数据 " + "="*50 + "\n")
                for k, v in metadata.items():
                    if isinstance(v, list):  # 关节名称列表特殊处理（换行）
                        f.write(f"{k}:\n")
                        for idx, name in enumerate(v):
                            f.write(f"  关节{idx+1}: {name}\n")
                    else:
                        f.write(f"{k}: {v}\n")
                f.write("\n" + "="*50 + " 逐帧数据 " + "="*50 + "\n")
                
                # 2. 写入逐帧数据
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
        # ========== 修改结束 ==========

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
        robot_name=RobotName.qb,
        retargeting_type=RetargetingType.dexpilot,
        hand_type=HandType.right,
        camera_path=None,
        virtual_video_file=video_file,
        camera_mat=default_camera_mat,
        detector_type=detector_name,
        headless=False,
        multi_cam=False,
        record_video = False
    )
