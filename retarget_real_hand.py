import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

# from .config import (
#     RobotName,
#     RetargetingType,
#     HandType,
#     get_default_config_path,
#     RetargetingConfig
# )
from config import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    RetargetingConfig
)

from detector.single_hand_detector import SingleHandDetector
from real_world_controller import XArm7QB
import threading

exit_flag = multiprocessing.Value('b', False)

def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    enable_real_robot = True
    # RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    # 加载指定的重定向配置文件
    logger.info(f"Start retargeting with config {config_path}")
    # 解析机械臂 URDF 路径、关节配置等，建立重定向优化序列
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    # 初始化手部检测器
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # Sapien 仿真场景搭建
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

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
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
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
    elif "qb" in robot_name:
        loader.scale = 1.7

    if "glb" not in robot_name and "qb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
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
    elif "qb" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))

        # from scipy.spatial.transform import Rotation
        # # 计算四元数
        # rot_mat = np.array([[0, 0, 1],
        #                     [1, 0, 0],
        #                     [0, 1, 0]])
        # quat = Rotation.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
        # # 设置基座位姿
        # robot.set_pose(sapien.Pose([0, 0, -0.2], quat))

    # Different robot loader may have different orders for joints
    # 读取当前加载的机械臂活跃关节名称列表
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    # 从重定向配置中读取预定义的关节名称列表
    retargeting_joint_names = retargeting.joint_names
    # 重定向关节索引→Sapien 关节索引映射数组
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # 真实机器人控制相关变量
    real_robot = None
    control_thread = None
    real_arm_qpos_lock = threading.Lock()
    real_hand_qpos_lock = threading.Lock()
    latest_arm_qpos = np.zeros(7)
    latest_hand_qpos = np.zeros(6)
    reset_flag = False
    
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
            with real_hand_qpos_lock:
                hand_qpos = latest_hand_qpos.copy()
            
            if real_robot is not None:
                try:
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
                hand_port="COM3",                 # 根据实际修改
                use_arm=False,
                use_hand=True,                    # 如果不需要手部可改为 False
            )
            real_robot.start()   # 启动机械臂内部控制线程（如果有）
            # 启动我们自己的控制线程
            control_thread = threading.Thread(target=real_robot_control_loop, daemon=True)
            control_thread.start()
            logger.info("Real robot controller started.")
        except Exception as e:
            logger.error(f"Failed to initialize real robot: {e}")
            enable_real_robot = False

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        # joint_pos：人手关节的 3D 空间位置数组
        # keypoint_2d：人手关节的 2D 像素坐标
        _, joint_pos, keypoint_2d, _, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            # logger.warning(f"{hand_type} hand is not detected.")
            pass
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
            qpos = retargeting.retarget(ref_value) # retarget_from_video.SeqRetargeting.Retarget->optimizer.retarget
            # 将计算出的关节角度按关节映射数组重排后，赋值给仿真机械臂更新姿态
            robot.set_qpos(qpos[retargeting_to_sapien])

            # ---------------- 更新真实机器人目标关节角 ----------------
            pin_qpos = qpos
            if enable_real_robot and real_robot is not None:
                # QB Hand 只有6个自由度
                # hand joint:'j1', 'j2', 'j3', 'j10', 'j11', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9'
                # 需要的6个角度是大拇指旋转、拇指弯曲、食指弯曲、中指弯曲、无名指弯曲、小指弯曲
                # 对应 'j1', 'j2', 'j4', 'j6', 'j8', 'j10'，在 Pinocchio 模型中索引分别是 0,1,5,7,9,3
                free_indices = [0,1,5,7,9,3]
                hand_target = pin_qpos[-11:][free_indices]
                with real_hand_qpos_lock:
                    latest_hand_qpos[:] = hand_target

        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = RetargetingConfig._DEFAULT_URDF_DIR

    queue = multiprocessing.Queue(maxsize=10)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, str(robot_dir), str(config_path)))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")

# RobotName: allegro; shadow; svh; leap; ability; inspire; panda
# RetargetingType: vector; position; dexpilot
# HandType: left; right
if __name__ == "__main__":
    main(
        robot_name=RobotName.mimic_qb,
        retargeting_type=RetargetingType.vector,
        hand_type=HandType.left,
        camera_path=None,
    )
