from typing import Optional

import pinocchio as pin
import numpy as np
from numpy.linalg import norm, pinv
from scipy.spatial.transform import Rotation as R
from loguru import logger

from control.filters import LPFilter, LPRotationFilter

class PinRobot:
    def __init__(self, urdf_path: str, mesh_dir: str):
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_dir)
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.joint_names = list(self.model.names[1:self.model.njoints])  # 排除固定关节、世界基关节等非运动学关节
        self.link_names = [f.name for f in self.model.frames if f.type == pin.FrameType.BODY]
        self.damping = 1e-6
        self.dt = 1.0
        self.ik_successs = True # IK求解是否成功的标志，供外部调用时参考

    def add_collision_pairs_excluding_adjacent(self):
        """
        为 collision_model 添加碰撞对。
        跳过属于同一个连杆的几何体，并可选地跳过属于父子关节的几何体。
        """
        # 获取每个几何体对应的父关节 ID
        geom_joint = [self.collision_model.geometryObjects[i].parentJoint for i in range(len(self.collision_model.geometryObjects))]
        
        # 构建父子关系集合，用于快速查找
        parent_child_pairs = set()
        # 遍历所有关节，跳过根关节（索引0）
        for joint_id in range(1, self.model.njoints):
            parent = self.model.parents[joint_id]   # 获取父关节索引
            child = joint_id                   # 当前关节即为子关节
            parent_child_pairs.add((parent, child))
            parent_child_pairs.add((child, parent))  # 加入反向关系，便于无向查找
        
        for i in range(len(self.collision_model.geometryObjects)):
            for j in range(i+1, len(self.collision_model.geometryObjects)):
                joint_i = geom_joint[i]
                joint_j = geom_joint[j]
                # 跳过属于同一个连杆的几何体对（它们不会发生碰撞）
                if joint_i == joint_j:
                    continue
                # 可选：跳过属于父子关节的几何体对
                if (joint_i, joint_j) in parent_child_pairs:
                    continue
                self.collision_model.addCollisionPair(pin.CollisionPair(i, j))
        
        self.collision_data = self.collision_model.createData()
        logger.info(f"Manually added {len(self.collision_model.collisionPairs)} collision pairs.")

    def remove_collision_pairs_between_links(self, link1_name: str, link2_name: str) -> int:
        """
        删除 collision_model 中属于 link1 和 link2 的所有几何体之间的碰撞对。

        Args:
            link1_name: 第一个 link 的名称
            link2_name: 第二个 link 的名称

        Returns:
            删除的碰撞对数量
        """
        # 1. 获取两个 link 对应的关节 ID（每个 link 有一个父关节）
        if not self.model.existFrame(link1_name) or not self.model.existFrame(link2_name):
            raise ValueError(f"Link '{link1_name}' or '{link2_name}' does not exist in model.")
        
        joint1 = self.model.frames[self.model.getFrameId(link1_name)].parent
        joint2 = self.model.frames[self.model.getFrameId(link2_name)].parent

        # 2. 收集属于这两个 link 的所有几何体索引
        geom_ids_link1 = []
        geom_ids_link2 = []
        for geom_id, geom in enumerate(self.collision_model.geometryObjects):
            if geom.parentJoint == joint1:
                geom_ids_link1.append(geom_id)
            elif geom.parentJoint == joint2:
                geom_ids_link2.append(geom_id)

        if not geom_ids_link1 or not geom_ids_link2:
            logger.warning(f"No geometry objects found for link '{link1_name}' or '{link2_name}'.")
            return 0

        # 3. 安全删除碰撞对（反向遍历）
        removed_count = 0
        # 从后向前遍历，避免索引变化导致的问题
        for idx in range(len(self.collision_model.collisionPairs) - 1, -1, -1):
            cp = self.collision_model.collisionPairs[idx]
            # 检查该碰撞对是否连接了两个指定的 link
            if (cp.first in geom_ids_link1 and cp.second in geom_ids_link2) or \
            (cp.first in geom_ids_link2 and cp.second in geom_ids_link1):
                self.collision_model.removeCollisionPair(cp)
                removed_count += 1

        self.collision_data = self.collision_model.createData()
        # logger.info(f"Removed {removed_count} collision pair(s) between '{link1_name}' and '{link2_name}'.")
        return removed_count
    
    def check_collision(self, qpos: np.ndarray, forwardKinematics: bool =True, show_details: bool =False) -> bool:
        """
        检查给定关节位置下模型是否发生自碰撞。
        qpos: 机械臂当前关节位置（numpy 数组）
        forwardKinematics: 是否在检查碰撞前执行前向运动学（默认为 True）
        show_details: 是否显示碰撞详情（默认为 False）
        返回值：是否发生碰撞（布尔值）
        """
        if forwardKinematics:
            pin.forwardKinematics(self.model, self.data, qpos)

        # 更新几何体位置
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, qpos)

        # 执行碰撞检测
        stop_at_first_collision = True
        is_colliding = pin.computeCollisions(self.collision_model, self.collision_data, stop_at_first_collision)

        if is_colliding and show_details:
            for pair_index in range(len(self.collision_model.collisionPairs)):
                if self.collision_data.activeCollisionPairs[pair_index]:
                    # 获取该碰撞对的几何体索引
                    geom1, geom2 = self.collision_model.collisionPairs[pair_index].first, self.collision_model.collisionPairs[pair_index].second
                    # 检查是否发生碰撞
                    if pin.computeCollision(self.collision_model, self.collision_data, pair_index):
                        logger.info(f"Collision detected in pair {pair_index} between geometries {geom1} and {geom2}")
            logger.info(f"Configuration {qpos} is in self-collision!")
        return is_colliding
    
    def forward_kinematics(self, qpos: np.ndarray, link_name: str) -> pin.SE3:
        """
        计算给定关节位置下指定 link 的位姿。
        qpos: 机械臂当前关节位置（numpy 数组）
        link_name: 需要计算位姿的 link 名称
        返回值：指定 link 的位姿（pin.SE3 对象）
        """
        if not self.model.existFrame(link_name):
            raise ValueError(f"Link '{link_name}' does not exist in model.")
        
        pin.forwardKinematics(self.model, self.data, qpos)
        link_id = self.model.getFrameId(link_name)
        link_pose = pin.updateFramePlacement(self.model, self.data, link_id)
        return link_pose
    
    def inverse_kinematics(self, qpos: np.ndarray, oMdes: pin.SE3, link_name: str,
                            iter: int = 1000, collision_check: bool = False, show_details: bool = False) -> np.ndarray:
        """
        使用经典的 CLIK 算法求解机械臂末端执行器的逆运动学问题。
        qpos: 机械臂当前关节位置（numpy 数组）
        oMdes: 目标末端位姿
        link_name: 末端执行器所在的 Frame 名称
        iter: 迭代次数
        collision_check: 是否在求解过程中进行碰撞检查
        show_details: 是否显示碰撞详情
        返回值：求解得到的机械臂关节位置（numpy 数组）
        """
        self.ik_successs = True # 默认 IK 求解成功，除非迭代过程中发生数值问题或未能收敛
        wrist_id = self.model.getFrameId(link_name)
        successfully_converged = False
        for i in range(iter):
            pin.forwardKinematics(self.model, self.data, qpos)

            wrist_pose = pin.updateFramePlacement(self.model, self.data, wrist_id)
            iMd = wrist_pose.actInv(oMdes)

            err = pin.log(iMd).vector
            if norm(err) < 1e-5:
                if show_details: logger.info("successfully converged")
                successfully_converged = True
                break

            J = pin.computeFrameJacobian(self.model, self.data, qpos, wrist_id)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.damping * np.eye(6), err))
            if not collision_check:
                qpos = pin.integrate(self.model, qpos, v * self.dt)
                # 立即裁剪，确保每次迭代后关节位置都在合法范围内
                qpos = np.clip(qpos, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            else:
                alpha = 1.0  # 初始步长
                min_alpha = 0.1
                # 如果启用碰撞检查，则在每次迭代后检查新配置是否发生碰撞
                qpos_candidate = pin.integrate(self.model, qpos, v * self.dt)
                qpos_candidate = np.clip(qpos_candidate, self.model.lowerPositionLimit, self.model.upperPositionLimit)
                if not self.check_collision(qpos_candidate, forwardKinematics=False):
                    qpos = qpos_candidate  # 仅在不发生碰撞时更新配置
                else:
                    if show_details: logger.info(f"Collision detected during IK iteration {i}.")
                    while alpha >= min_alpha:
                        qpos_candidate = pin.integrate(self.model, qpos, v * self.dt * alpha)
                        qpos_candidate = np.clip(qpos_candidate, self.model.lowerPositionLimit, self.model.upperPositionLimit)
                        if not self.check_collision(qpos_candidate, forwardKinematics=False):
                            qpos = qpos_candidate
                            if show_details:
                                logger.info(f"Reduced step size to {alpha:.2f} to avoid collision.")
                            break
                        alpha *= 0.5  # 减小步长
                    if alpha < min_alpha:
                        if show_details:
                            logger.warning(f"Unable to find a collision-free configuration in iteration {i}.")
                            self.ik_successs = False
                        # 可以选择跳过更新，或者在这里实现更复杂的避障策略
                        qpos += np.random.normal(0, 0.01, qpos.shape)  # 添加随机扰动以尝试跳出局部碰撞状态
                        qpos = np.clip(qpos, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        qpos = np.clip(qpos, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        if not successfully_converged:
            self.ik_successs = False
            if show_details: logger.warning("Failed to converge to the desired pose within the maximum iterations.")
        return qpos

class PinRobotController:
    def __init__(self, robot: PinRobot, initial_qpos: Optional[np.ndarray],
                  wrist_alpha: float = 0.5, ee_link: str = "base_link", 
                  hand_links: list = None, retract_qpos: Optional[np.ndarray] = None):
        self.robot = robot
        self.qpos = initial_qpos.copy() if initial_qpos is not None else np.zeros(robot.model.nq)
        self.wrist_pos_fliter = LPFilter(wrist_alpha)
        self.wrist_rot_fliter = LPRotationFilter(wrist_alpha)
        self.ee_link = ee_link
        self.pose = self.robot.forward_kinematics(self.qpos, self.ee_link)  # 计算初始位姿
        self.failures = 0  # IK 求解失败次数统计
        self.failures_threshold = 50  # IK 求解失败次数阈值，超过后可以触发特殊处理（如重置、报警等）
        self.retract_qpos = retract_qpos.copy() if retract_qpos is not None else np.zeros(robot.model.nq)
        self.last_successful_qpos = self.qpos.copy()  # 记录上一次成功的 qpos，用于在 IK 失败时回退
        self.last_successful_pose = self.pose.copy()  # 记录上一次成功的位姿，用于在 IK 失败时回退
        self.init_flag = False  # 标志位，指示是否已完成第一次更新（用于滤波器的初始化）

        # 添加碰撞对，排除手部内部碰撞对
        self.robot.add_collision_pairs_excluding_adjacent()
        if hand_links is not None:
            # 删除手部内部所有 link 之间的碰撞对
            for i in range(len(hand_links)):
                for j in range(i + 1, len(hand_links)):
                    self.robot.remove_collision_pairs_between_links(hand_links[i], hand_links[j])

    def update(self, wrist_pos: np.array, wrist_rot: np.ndarray, check_collision: bool = True):
        """根据给定的末端位姿更新机械臂关节位置"""
        if not self.init_flag:
            # 初始化的时候不经过滤波器，滤波器自带初始化步骤，不需要额外修改
            self.init_flag = True
        # 滤波末端位姿，得到平滑的目标位姿
        wrist_pos = self.wrist_pos_fliter.next(wrist_pos)
        wrist_quat = R.from_matrix(wrist_rot).as_quat() # [qx,qy,qz,qw]
        wrist_quat = np.array([wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2]])  # 转换为 [qw, qx, qy, qz] 格式
        wrist_quat = self.wrist_rot_fliter.next(wrist_quat)
        wrist_quat = np.array([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]])  # 转回 [qx, qy, qz, qw] 格式
        wrist_rot = R.from_quat(wrist_quat).as_matrix()
        oMdes = pin.SE3(wrist_rot, wrist_pos)

        # 使用当前关节位置作为 IK 求解的初始 guess
        ik_qpos = self.qpos.copy()
        next_qpos = self.robot.inverse_kinematics(ik_qpos, oMdes, self.ee_link, collision_check=False)
        # 如果启用碰撞检查，则在 IK 求解过程中进行碰撞检测和避障
        # 事实上只需关注最终求解得到的 qpos 是否发生碰撞，如果发生碰撞则尝试在IK求解过程中规避碰撞
        if check_collision:
            if self.robot.check_collision(next_qpos, forwardKinematics=False):
                # logger.info("Collision detected in controller update. Attempting to resolve...")
                next_qpos = self.robot.inverse_kinematics(ik_qpos, oMdes, self.ee_link, collision_check=True)
        
        if self.robot.ik_successs:
            self.failures = 0  # IK 求解成功，重置失败次数统计

            # 只有IK求解成功时才更新 qpos 和 pose，否则保持不变并增加失败计数
            qpos = next_qpos

            # self.qpos = pin.interpolate(self.robot.model, self.qpos, qpos, 0.5)
            self.qpos = qpos.copy()
            self.pose = self.robot.forward_kinematics(self.qpos, self.ee_link)

            self.last_successful_qpos = next_qpos.copy()  # 更新最后成功的 qpos
            self.last_successful_pose = self.pose.copy()  # 更新最后成功的位姿
        else:
            self.failures += 1
            self.wrist_pos_fliter.cancel()  # 取消位置滤波器的上一次更新，回退到上一个位置
            self.wrist_rot_fliter.cancel()  # 取消旋转滤波器的上一次更新，回退到上一个旋转
            if self.failures % 10 == 0:  # 每10次失败记录一次警告日志
                logger.warning(f"IK solve failed. Failure count: {self.failures}/{self.failures_threshold}")
            if self.failures >= self.failures_threshold:
                logger.error("IK failure count exceeded threshold. Controller may be stuck. Consider resetting the robot or checking for issues.")
                # 可以在这里实现特殊处理逻辑，例如重置机械臂到初始位置、发送报警信号等
                # self.reset()  # 示例：重置机械臂到初始位置
                # logger.info("Robot has been reset due to repeated IK failures.")

        return self.qpos.copy()
    
    def set_qpos(self, qops: np.ndarray) -> np.ndarray:
        """仿真中机械臂不一定会达到预期位置，因此提供一个直接设置 qpos 的接口，供仿真环境调用"""
        self.qpos = qops.copy()
        self.pose = self.robot.forward_kinematics(self.qpos, self.ee_link)
        return self.qpos.copy()
    
    def reset(self, retract_qpos: Optional[np.ndarray] = None):
        """重置机械臂到初始位置"""
        self.failures = 0  # 重置 IK 求解失败次数统计
        self.qpos = retract_qpos.copy() if retract_qpos is not None else self.retract_qpos.copy()
        self.pose = self.robot.forward_kinematics(self.qpos, self.ee_link)
        self.wrist_pos_fliter.reset()
        self.wrist_rot_fliter.reset()
        self.init_flag = False  # 重置初始化标志，使下一次 update 时重新初始化滤波器
    
def clamp_rotation(R_curr: R, R_std: R = R.from_euler('zyx', [-90, 0, 180], degrees=True), 
                   limits_deg: tuple = (30, 60, 90),
                   order: str = 'zxy', abandon = True) -> R:
    """
    将当前旋转钳位到标准姿态附近的有效范围内。
    
    参数
    ----------
    R_curr : Rotation
        当前腕部旋转（世界坐标系）
    R_std : Rotation
        标准姿态旋转
    limits_deg : (float, float, float)
        各轴允许的最大角度（度），顺序与 order 一致
    order : str
        相对旋转的欧拉角分解顺序（如 'xyz', 'zyx' 等）
        z: 手部左右摆动（yaw）
        y: 手部旋转（yaw）
        x: 手部上下摆动（pitch）
    
    返回
    -------
    Rotation
        裁剪后的旋转矩阵（仍处于世界坐标系）
    """
    # 计算相对旋转（即从标准姿态到当前姿态的旋转）
    R_rel = R_std.inv() * R_curr
    
    # 将相对旋转分解为欧拉角（度）
    euler = R_rel.as_euler(order, degrees=True)
    
    if np.any(np.abs(euler) > limits_deg) and abandon:
        print("clip!")
        return None   # 或跳过该帧

    # 对每个角度进行钳位
    clamped_euler = np.clip(euler, -np.array(limits_deg), np.array(limits_deg))
    
    # 如果没有任何改变，直接返回原旋转
    if np.allclose(euler, clamped_euler):
        return R_curr
    
    # 重新合成受限的相对旋转
    R_rel_clamped = R.from_euler(order, clamped_euler, degrees=True)
    
    # 转换回世界坐标系
    R_clamped = R_std * R_rel_clamped
    return R_clamped

def is_pose_changed(
    wrist_pos_prev, wrist_rot_prev,
    wrist_pos_curr, wrist_rot_curr,
    pos_threshold=0.01,    # 位置阈值：米
    rot_threshold=5.0      # 旋转阈值：度
):
    """
    判断腕部姿态是否发生足够大的变化
    返回 True = 变化大 → 需要更新
    返回 False = 变化小 → 不需要更新
    """

    # ======================
    # 1. 判断位置变化（距离）
    # ======================
    pos_diff = np.linalg.norm(wrist_pos_curr - wrist_pos_prev)

    # ======================
    # 2. 判断旋转变化（角度）
    # ======================
    # 旋转差矩阵：R_prev → R_curr
    R_diff = wrist_rot_prev.T @ wrist_rot_curr

    # 计算旋转角（公式最稳定）
    cos_theta = (np.trace(R_diff) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
    rot_diff_rad = np.arccos(cos_theta)
    rot_diff_deg = np.rad2deg(rot_diff_rad)

    # ======================
    # 3. 最终判断
    # ======================
    pos_changed = pos_diff > pos_threshold
    rot_changed = rot_diff_deg > rot_threshold

    return pos_changed or rot_changed

if __name__ == "__main__":
    urdf = r"D:\study\Grasp\xarm7_qbr\qbr.urdf"
    # 关键：mesh_dir 指向 URDF 所在目录（即 meshes 的父目录），而不是 meshes 本身
    mesh_dir = r"D:\study\Grasp\xarm7_qbr"

    # # 使用凸包分解后的 URDF 加载模型
    # urdf = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr_decompose\qbr_decompose.urdf"
    # mesh_dir = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr_decompose"

    # # 加载 URDF
    # model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf, mesh_dir)

    hand_links = [
        "base_link",
        "link1", "link2", "link3", "link4", "link5",
        "link6", "link7", "link8", "link9", "link10", "link11"
    ]
    robot = PinRobot(urdf, mesh_dir)
    initial_qpos = np.zeros(robot.model.nq)
    # initial_qpos[:7]=[-np.pi/4,-0.32,0.06,1.49,-0.3,0.78,-0.29] # 机械臂初始位置，单位为弧度
    # initial_qpos[:7]=[0,0.21,0,2.04,0,0.06,0] # 机械臂初始位置，单位为弧度
    initial_qpos[:7]=[0,0.21,0,0.9,0,-0.75,0] # 机械臂初始位置，单位为弧度
    controller = PinRobotController(robot, initial_qpos=initial_qpos, wrist_alpha=0.5, ee_link="base_link", hand_links=hand_links)
    init_pose = controller.pose
    print("初始位姿：", init_pose)
    print(R.from_matrix(init_pose.rotation).as_euler('zyx', degrees=True))
    wrist_rot = R.from_euler('zyx', [-90, 0, 180], degrees=True).as_matrix()  # 末端执行器的旋转矩阵（示例为单位旋转）
    wrist_rot = R.from_euler('xzy', [180, 90, 0], degrees=True).as_matrix()  # 末端执行器的旋转矩阵（示例为单位旋转）
    # wrist_rot = np.array([[-0.04615052, -0.09079317, -0.99479984],
    #    [-0.99332603, -0.10120856,  0.05531923],
    #    [-0.10570487,  0.9907136 , -0.08551639]])
    print("wrist_rot:\n", wrist_rot)
    wrist_pos=init_pose.translation # 末端执行器的目标位置（示例为初始位置偏移）
    # wrist_pos = np.array([-0.00807995,  0.09134355,  0.01460218])+np.array([0.4, 0.25, 0.3])
    qpos = controller.update(wrist_pos=wrist_pos, wrist_rot=wrist_rot, check_collision=True)
    print("succsssfully converged") if controller.failures==0 else print("IK failed to converge at initialization!")
    # qpos = initial_qpos.copy()
    is_colliding = robot.check_collision(qpos)
    print(f"Is colliding: {is_colliding}")
    print("最终位姿：")
    print(controller.pose)
    # from sapien_example import visualize
    # visualize(r"D:\study\Grasp\xarm7_qbr\qbr.urdf", qpos, joint_names=robot.joint_names)

    import sapien
    scene = sapien.Scene()


    # 地面材质：加深加暗，突出白色机器手
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.02, 0.03, 0.05, 1]  # 近黑色地面
    render_mat.metallic = 0.0
    render_mat.roughness = 0.95
    render_mat.specular = 0.1
    scene.add_ground(0, render_material=render_mat, render_half_size=[1000, 1000])

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

    # 0: 默认主视角 (观测整体)
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    # cam.set_local_pose(sapien.Pose([0.25, 0.25, 1.3], [0.947, -0.05, 0.254, -0.188])) # 从-x轴观测，略微调整位置和角度以获得更好视角
    # cam.set_local_pose(sapien.Pose([0.35, 0.35, 1.2], [0.906, -0.075, 0.180, -0.375])) 
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

    viewer = scene.create_viewer()
    # viewer.set_camera_pose(sapien.Pose([0.4, 0.2, 1.6], [0.924, 0, 0.383, 0]))
    # viewer.set_camera_pose(sapien.Pose([0.25, 0.25, 1.4], [0.947, -0.05, 0.254, -0.188]))
    viewer.set_camera_pose(cam.get_local_pose())

    from pynput import keyboard
    import cv2
    current_key = None
    def on_press(key):
        global current_key
        # 普通字母、数字、符号键
        try:
            char = key.char
            current_key = char.lower()  # 统一小写，不分大小写
        except AttributeError:
            current_key = None


    # 启动监听
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True  # 设置为守护线程，主程序退出时自动结束监听
    listener.start()

    # Create Actors
    from sapien_demos.create_actors import *
    from sapien_demos.camera import compute_camera_pose
    # box = create_box(
    #     scene,
    #     sapien.Pose(p=[0, -0.65, 0.5 + 0.05]),
    #     half_size=[0.03, 0.03, 0.03],
    #     color=[1.0, 0.0, 0.0],
    #     name="box",
    # )
    # sphere = create_sphere(
    #     scene,
    #     sapien.Pose(p=[0.2, -0.65, 0.5 + 0.05]),
    #     radius=0.03,
    #     color=[0.0, 1.0, 0.0],
    #     name="sphere",
    # )
    # capsule = create_capsule(
    #     scene,
    #     sapien.Pose(p=[0.4, -0.65, 0.5 + 0.05]),
    #     radius=0.03,
    #     half_length=0.02,
    #     color=[0.0, 0.0, 1.0],
    #     name="capsule",
    # )
    # table = create_table(
    #     scene,
    #     sapien.Pose(p=[0.2, -0.65, 0.0]),
    #     size=0.6,
    #     height=0.5,
    #     thickness=0.08,
    # )
    # 创建仿真环境中的actor（如桌子、物体等），提供视觉参考和交互对象
    box_half_size = 0.03
    sphere_radius = 0.03
    capsule_radius = 0.03
    capsule_half_length = 0.02
    table_size = 1.15
    table_height = 0.8
    table_center = [0.36, 0, table_height]  # 根据需要调整桌子位置
    table_thickness = 0.1
    object_table_height = 0.1
    object_center = [0.5+0.18, 0, table_height+object_table_height+0.05]

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
        category_id=21,
        static_friction=30.0,
        dynamic_friction=20.0,
    )
    table = create_table(
        scene,
        sapien.Pose(p=table_center),
        size=table_size,
        height=table_height,
        thickness=table_thickness,
        is_kinematic=True
    )

    # Load URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    # SAPIEN 正确的全局摩擦设置方式（加载URDF之前用！）
    loader.collision_material = scene.create_physical_material(
        static_friction=3.0,    # 静摩擦（越大越不滑）
        dynamic_friction=2.0,   # 动摩擦
        restitution=0.0         # 不弹
    )


    arm = loader.load(urdf)

    Dof = arm.get_dof()
    arm.set_root_pose(sapien.Pose(p=[0.0, 0.0, table_height]))
    arm.set_qpos(initial_qpos)

    # 末端挂载相机
    mounted_camera = scene.add_mounted_camera(
        name="mounted_camera",
        mount=arm.links[11].entity,  # 将相机挂载在机械臂末端链接上
        pose=sapien.Pose(p=[0, 0.001, -0.075],q=[0,0.707,0.707,0]),
        width=600,
        height=600,
        fovy=1,
        near=0.1,
        far=10,
    )
    cameras["mounted"] = mounted_camera  # 末端挂载相机

    # 读取当前加载的机械臂活跃关节名称列表
    sapien_joint_names = [joint.get_name() for joint in arm.get_active_joints()]
    # print(sapien_joint_names)
    # 重定向关节索引→Sapien 关节索引映射数组
    retargeting_to_sapien = np.array(
        [robot.joint_names.index(name) for name in sapien_joint_names]
    ).astype(int) if robot.joint_names is not None else np.arange(Dof)
    arm.set_qpos(controller.qpos[retargeting_to_sapien])

    active_joints = arm.get_active_joints()
    use_internal_drive = True
    use_step = True
    scene.step()

    if use_internal_drive:
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(stiffness=1000, damping=100, force_limit=1000, mode="force")
            joint.set_drive_target(controller.qpos[retargeting_to_sapien][joint_idx])


    i = 0
    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            i += 1
            ## print(i)
            # if i % 4 == 0:
            #     wrist_pos = initial_pose.translation + np.array([0.1 * np.sin(i / 100), 0.1 * np.cos(i / 100), 0.05 * np.sin(i / 200)])
            #     wrist_rot = R.from_euler('zyx', [180, 180+5*np.sin(i/100), -15+30*np.sin(i/100)], degrees=True).as_matrix()
            #     qpos = controller.update(wrist_pos=wrist_pos, wrist_rot=wrist_rot, check_collision=True)
            #     # print(f"Step {i}, qpos: {controller.qpos}")
            if use_step:
                if use_internal_drive:
                    # for joint_idx, joint in enumerate(active_joints):
                        # joint.set_drive_target(controller.qpos[retargeting_to_sapien][joint_idx])
                    if current_key == "q":
                        logger.info("🤖 Update drive targets...")
                        for joint_idx, joint in enumerate(active_joints):
                            joint.set_drive_target(arm.get_qpos()[joint_idx])
                        current_key = None
                        pose = arm.get_pose()
                        scene.step()
                    # elif current_key == "z":
                    #     saved_camera_pose = cam.get_local_pose()
                    #     # 打印位置和四元数（可直接复制到 set_camera_pose 中使用）
                    #     pos = saved_camera_pose.p
                    #     quat = saved_camera_pose.q
                    #     print(f"Camera pose saved: Pose(p={list(pos)}, q={list(quat)})")
                    elif current_key == "x":
                        print(arm.links[11].get_pose())
                        current_key = None
                    elif current_key == "0":
                        viewer.set_camera_pose(cam.get_local_pose())
                        current_key = None
                    elif current_key == "1":
                        viewer.set_camera_pose(front_cam.get_local_pose())
                        current_key = None
                    elif current_key == "2":
                        viewer.set_camera_pose(top_cam.get_local_pose())
                        current_key = None
                    elif current_key == "3":
                        viewer.set_camera_pose(side_cam.get_local_pose())
                        current_key = None
                    elif current_key == "v":
                        current_key = None
                        # 渲染所有相机图像
                        images = {}
                        for name, cams in cameras.items():
                            if name == "main":
                                continue  # 跳过主相机
                            cams.take_picture()
                            rgba = cams.get_picture("Color")  # [H, W, 4]
                            cam_bgr = cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                            images[name] = cam_bgr
                        
                        # 拼接图像（2x2网格）
                        top_row = np.hstack([images["mounted"], images["top"]])
                        bottom_row = np.hstack([images["side"], images["front"]])
                        combined = np.vstack([top_row, bottom_row])
                        
                        # 添加视角标签
                        cv2.putText(combined, "Mounted View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(combined, "Top View", (620, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(combined, "Side View", (20, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(combined, "Front View", (620, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        
                        # 显示
                        cv2.imshow("Multi-View Grasping", combined)
                        cv2.waitKey(1) 
            else:
                arm.set_qpos(controller.qpos[retargeting_to_sapien])
        scene.update_render()
        viewer.render()

    