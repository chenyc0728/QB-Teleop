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

from scipy.spatial.transform import Rotation as R

from coord_converter import SAPIEN2MEDIAPIPE, PLOT2SAPIEN, ROTATE_Z, CUROBO_POSE_FIX, HAND_POSE_FIX, ROTATE_Y

from motion_control import PinRobot, PinRobotController, is_pose_changed

from config import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    RetargetingConfig
)

from detector.single_hand_detector import SingleHandDetector

# assembly urdf path
# arm_path = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qb\xarm7_qb_right_hand.urdf"
arm_path = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr\qbr.urdf"
mesh_path = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr"
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
                      robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType):
    USING_PRESET_POSES = False
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
    # render_mat = sapien.render.RenderMaterial()
    # render_mat.base_color = [0.06, 0.08, 0.12, 1]
    # render_mat.metallic = 0.0
    # render_mat.roughness = 0.9
    # render_mat.specular = 0.8
    # scene.add_ground(0, render_material=render_mat, render_half_size=[1000, 1000])

    # # Lighting 光照
    # scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    # scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    # scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    # scene.set_environment_map(
    #     create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    # )
    # scene.add_area_light_for_ray_tracing(
    #     sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    # )
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
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.8, -0.4, 0.3], [0, 0, 0, -1])) # 从+x轴观测
    # cam.set_local_pose(sapien.Pose([0, -1, 0.3], [0.707, 0, 0, 0.707])) # 从-y轴观测

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

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
    # urdf_path = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm_qb\xarmqb.urdf"
    loader.scale = 1 # 0.4 # 模型大小
    robot = loader.load(urdf_path)
    # 将机器人整体向左平移 x:前 y:左 z:上
    robot.set_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))  # 根据需要调整平移量

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
    init_qpos[:7]=[-np.pi/2,0.21,0,0.9,0,-0.75,0]
    robot.set_qpos(init_qpos[retargeting_to_sapien])
    
    # 建立机器人Pinocchio模型和控制器
    pin_robot = PinRobot(urdf_path, mesh_path)
    pin_joint_names = pin_robot.joint_names
    controller = PinRobotController(pin_robot, initial_qpos=init_qpos, hand_links=hand_links, ee_link="base_link", wrist_alpha=0.65)

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

    # 初始化轨迹策略
    last_goal_pose = None      # 上一次的目标位姿
    pose_threshold = 0.01      # 位姿变化阈值（位置变化0.05米或角度变化10度）
    rot_threshold = 10             # 角度变化阈值（单位：度）

    # 手部位置原点，避免过于靠近底座，设置为机械臂末端初始位置
    origin_pos = controller.pose.translation

    # 在 机器人加载完成后保存元数据
    metadata = {
        'robot_name': robot_name,          # 枚举值转为字符串
        'retargeting_type': retargeting_type.value,
        'hand_type': hand_type,
        'retargeting_joint_names': retargeting_joint_names,
        'sapien_joint_names': sapien_joint_names,
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
    
    # 在循环开始前记录起始时间
    start_time = time.time()
    record_data = False

    try:
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
            _, joint_pos, keypoint_2d, wrist_rot, wrist_pos = detector.detect(rgb)
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            cv2.imshow("realtime_retargeting_demo", bgr)
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
            elif key == ord("p") and USING_PRESET_POSES:
                id += 1
                this_quat = quat_list[id%10]
                logger.info(f"📝 Using Wrist Quat: {this_quat}")
            elif key == ord("o"):
                USING_PRESET_POSES = not USING_PRESET_POSES
                logger.info(f"📝 Using Pre-set Wrist Quat is set {USING_PRESET_POSES}")
                if USING_PRESET_POSES:
                    logger.info(f"📝 Using Wrist Quat: {this_quat}")
            elif key == ord("r"):
                controller.reset()
                robot.set_qpos(controller.qpos[retargeting_to_sapien])  # 重置后立即更新机械臂姿态
                logger.info("🔧 Controller reset...")
            elif key == ord("w"):
                origin_pos[0] -= 0.01
                logger.info(f"📍 Move origin forward: {origin_pos}")
            elif key == ord("s"):
                origin_pos[0] += 0.01
                logger.info(f"📍 Move origin backward: {origin_pos}")
            elif key == ord("a"):
                origin_pos[1] -= 0.01
                logger.info(f"📍 Move origin left: {origin_pos}")
            elif key == ord("d"):
                origin_pos[1] += 0.01
                logger.info(f"📍 Move origin right: {origin_pos}")
            elif key == ord("c"):
                origin_pos[2] -= 0.01
                logger.info(f"📍 Move origin down: {origin_pos}")
            elif key == ord("e"):
                origin_pos[2] += 0.01
                logger.info(f"📍 Move origin up: {origin_pos}")

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
                wrist_rot = wrist_rot @ SAPIEN2MEDIAPIPE
                # world2camera
                wrist_rot = wrist_rot.T

                # 修正初始位置
                wrist_rot_right = wrist_rot @ HAND_POSE_FIX
                # 修正curobo初始姿态
                wrist_rot_right = wrist_rot @ CUROBO_POSE_FIX
                # wrist_rot_right = ROTATE_Z @ wrist_rot_right
                if urdf_path == r"D:\study\Grasp\URDF\xarm_qb\xarmqb.urdf" or urdf_path == r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm_qb\xarmqb.urdf":
                    wrist_rot_right = wrist_rot_right @ ROTATE_Y
                
                # 将旋转矩阵转换为四元数
                r = R.from_matrix(wrist_rot_right) 
                wrist_quat = r.as_quat()  # 格式: [x, y, z, w]

                # 测试
                if USING_PRESET_POSES:
                    # wrist_quat = this_quat
                    if urdf_path != r"D:\study\Grasp\URDF\xarm_qb\xarmqb.urdf" and urdf_path != r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm_qb\xarmqb.urdf":
                        wrist_quat = R.from_matrix(R.from_quat(this_quat).as_matrix() @ CUROBO_POSE_FIX).as_quat()
                    else:
                        wrist_quat = R.from_matrix(R.from_quat(this_quat).as_matrix() @ CUROBO_POSE_FIX @ ROTATE_Y).as_quat()

                # 在 retargeting 循环内，获取 wrist_rot 后立即打印
                # print("原始 wrist_rot:\n", wrist_rot_o)
                # print("变换后 wrist_rot:\n", CAMERA2ROBOT @ wrist_rot.T)
                # print("四元数 (w,x,y,z):", wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2])

                # 目标腕部姿态 [x, y, z, qw, qx, qy, qz]
                wrist_pos_m = wrist_pos @ SAPIEN2MEDIAPIPE + origin_pos
                # wrist_pos_m = origin_pos
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
                robot.set_qpos(qpos[retargeting_to_sapien])

            for _ in range(2):
                viewer.render()


            # 记录当前帧时间（绝对时间戳）
            timestamp = time.time()
            # print(timestamp)
            # 根据检测结果组装数据
            if joint_pos is not None:
                # 检测成功
                current_joint_pos = joint_pos
                # 将 keypoint_2d 转换为像素坐标
                h, w = rgb.shape[:2]
                current_keypoint_2d = landmarks_to_pixel_array(keypoint_2d, (h, w))
                current_ref_value = ref_value
                current_qpos = qpos
                current_wrist_rot = wrist_rot
                current_wrist_pos = np.expand_dims(wrist_pos, axis=0)
            else:
                # 检测失败，joint_pos 和 keypoint_2d 为 None，ref_value 和 qpos 取当前机械臂状态或 None
                current_joint_pos = None
                current_keypoint_2d = None
                current_ref_value = None
                current_qpos = robot.get_qpos()   # 获取当前机器人关节角度
                current_wrist_rot = None
                current_wrist_pos = None
            
            if record_data:
                data_list.append({
                'timestamp': timestamp,
                'joint_pos': current_joint_pos,
                'keypoint_2d': current_keypoint_2d,
                'ref_value': current_ref_value,
                'qpos': current_qpos,
                'wrist_rot': current_wrist_rot,
                'wrist_pos': current_wrist_pos
            })

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # ===== 强制关闭所有窗口 =====
        cv2.destroyAllWindows()
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

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, camera_path))

    # 对应消费者进程修改
    consumer_process = multiprocessing.Process(
    target=start_retargeting,
    args=(queue, str(robot_dir), str(config_path), robot_name, retargeting_type, hand_type)
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
if __name__ == "__main__":
    main(
        robot_name=RobotName.qb,
        retargeting_type=RetargetingType.dexpilot,
        hand_type=HandType.right,
        camera_path=None,
    )
