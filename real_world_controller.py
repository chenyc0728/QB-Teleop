from xarm.wrapper import XArmAPI
from qb_hand.dexterous_hand_controller import DexterousHand
import numpy as np
import time
import threading
import warnings

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._prev_err = None
        self._cum_err = 0

    def reset(self):
        self._prev_err = None
        self._cum_err = 0

    def control(self, err, dt):
        if self._prev_err is None:
            self._prev_err = err

        value = (
                self.kp * err
                + self.kd * (err - self._prev_err) / dt
                + self.ki * self._cum_err
        )

        self._prev_err = err
        self._cum_err += dt * err

        return value

class XArm7QB:
    def __init__(
                self,
                arm_ip_address = "192.168.1.204",
                control_dt = 0.02,
                internal_control_dt=1.0 / 250,
                initial_arm_qpos = [0,0.21,0,0.9,0,-0.75,0],
                hand_port="COM3",
                hand_baudrate=115200,
                hand_timeout=0.2,
                use_arm = True,
                use_hand = False,
                arm_mode = 4,
            ):
        self.control_dt = control_dt
        self.internal_control_dt = internal_control_dt
        self.initial_arm_qpos = initial_arm_qpos
        self.use_arm = use_arm
        self.use_hand = use_hand
        self.arm_mode = arm_mode
        self.hand_lower_limits = np.array([0, 0, 0, 0, 0, 0])  # rad
        self.hand_upper_limits = np.array([1.78, 0.29, 1.24, 1.24, 1.24, 1.24])  # rad
        if use_arm:
            self.arm = XArmAPI(arm_ip_address, is_radian=True)
            self.arm_velocity_limit = 1.0

            if not self.arm.connected:
                print("检测到机械臂未连接，现在开始连接...")
                code = self.arm.connect()
                if code != 0:
                    print(f"机械臂连接失败！错误码: {code}")
                    self.use_arm = False
                    exit()
            print("准备开始执行任务...")


            if self.arm_mode==4:
                default_kp = np.array([2, 2, 1, 1, 1, 1, 1]) * 5
                default_kd = default_kp / 20
                default_ki = np.zeros(7)
                self.max_arm_velocity = np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.5])
                self.arm_pid = PIDController(
                    kp=default_kp,
                    ki=default_ki,
                    kd=default_kd,
                )

            # Setup control thread
            # self._arm_thread = threading.Thread(target=self._internal_control_arm_qpos)
            # self._arm_lock = threading.Lock()
            # self._arm_pos_target = None
            # self._arm_should_stop = False
            # 获取当前机械臂位置作为初始目标
            current_qpos = self.get_arm_qpos()
            self._arm_pos_target = current_qpos if current_qpos is not None else np.zeros(7)
            self._arm_should_stop = False
            self._arm_thread = threading.Thread(target=self._internal_control_arm_qpos)
            self._arm_lock = threading.Lock()
        else:
            self.arm = None

        if use_hand:
            try:
                self.hand = DexterousHand(port=hand_port, baudrate=hand_baudrate, timeout=hand_timeout)
                print("机械手连接成功！准备开始执行任务...")
            except Exception as e:
                print(f"机械手连接失败！错误信息: {e}")
                self.use_hand = False
        else:
            self.hand = None
        
        self.reset()
        self.last_control_time = None

    def reset(self):
        if self.use_arm:
            self.arm.clean_error()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)

            self.arm.set_servo_angle(angle=self.initial_arm_qpos, speed=10, wait=True)
            self.arm.set_mode(self.arm_mode)
            self.arm.set_state(0)
            print("回到初始姿态...")

        if self.use_hand:
            self.hand.preset_open()

        time.sleep(0.5)

    def get_arm_qpos(self):
        """
        Get the current joint positions of the arm.
        :return: np.array of shape (7,)
        """
        if self.arm is None:
            return np.zeros(7)
        code, xarm_state = self.arm.get_joint_states(is_radian=True)
        return np.array(xarm_state[0])

    def get_ee_pose(self):
        """
        Get the current end-effector pose of the arm.
        :return: np.array of shape (6,) in the format of (x, y, z, roll, pitch, yaw)"""
        if self.arm is None:
            return np.zeros(6)
        code, xarm_eef_pose = self.arm.get_position(is_radian=True)
        return np.array(xarm_eef_pose)
    
    def clip_next_qpos(self, target_qpos, velocity_limit=0.8):
        """
        Clip the target joint positions to ensure that the resulting joint velocities do not exceed the specified velocity limit.
        :param target_qpos: np.array of shape (7,) representing the target joint positions
        :param velocity_limit: float, the maximum joint velocity limit
        :return: np.array of shape (7,) representing the clipped joint positions
        """
        if target_qpos is None:
            # 如果没有目标位置，返回当前位置
            return self.get_arm_qpos()
        
        current_qpos = self.get_arm_qpos()
        
        # 添加类型检查
        if current_qpos is None or target_qpos is None:
            return current_qpos if current_qpos is not None else np.zeros(7)
        # current_qpos = self.get_arm_qpos()
        error = target_qpos - current_qpos
        motion_scale = np.max(np.abs(error)) / (velocity_limit * self.control_dt)
        motion_scale = max(motion_scale,1) # 防止放大误差
        safe_control_qpos = current_qpos + error / motion_scale
        return safe_control_qpos

    def clip_arm_velocity(self, arm_qvel: np.ndarray):
        """
        Clip the arm joint velocities to ensure they do not exceed the specified velocity limit.
        :param arm_qvel: np.array of shape (7,) representing the desired joint velocities
        :return: np.array of shape (7,) representing the clipped joint velocities
        """
        velocity_overshot = np.abs(arm_qvel) / self.max_arm_velocity
        max_overshot = np.max(velocity_overshot)
        if max_overshot > 1 + 1e-4:
            if not hasattr(self, '_clip_print_counter'):
                self._clip_print_counter = 0
            self._clip_print_counter += 1
            safe_velocity = arm_qvel / max_overshot
            bottleneck_joint = np.argmax(velocity_overshot)
            if self._clip_print_counter % 100 == 0:
                print(f"Bottleneck joint for velocity clip: joint-{bottleneck_joint + 1} with overshoot {max_overshot}")
        else:
            safe_velocity = arm_qvel
        return safe_velocity

    def control_arm_qpos(self, arm_qpos: np.ndarray):
        """
        Set the target joint positions for the arm. The internal control thread will handle the actual movement towards these target positions.
        :param arm_qpos: np.array of shape (7,) representing the target joint positions
        """
        with self._arm_lock:
            self._arm_pos_target = arm_qpos

    def get_hand_qpos(self):
        """
        Get the current joint positions of the hand.
        :return: np.array of shape (6,) representing the current joint positions of the hand
        """
        if self.hand is None:
            return np.zeros(6)
        hand_angles = list(self.hand.query_angles()["angles"].values())
        hand_qpos = np.deg2rad(hand_angles)  # Convert from degrees to radians
        return np.array(hand_qpos)

    def control_hand_qpos(self, hand_qpos: np.ndarray):
        """
        Control the hand joint positions directly.
        :param hand_qpos: np.array of shape (6,) representing the target joint positions for the hand
        """
        if self.hand is not None:
            # 缩放到0-90度范围内，并转换为整数角度值
            # j1 j2 j4 j6 j8 j10
            # lower limits: [-1.79, 0, -1.24, -1.24, -1.24, -1.24], upper limits: [0, 0.29, 0, 0, 0, 0], rad
            # hand_angles = np.abs(np.rad2deg(hand_qpos[-6:]))  # Convert from radians to degrees
            hand_angles = (np.abs(hand_qpos[-6:])-self.hand_lower_limits)/self.hand_upper_limits * 90  # Scale to 0-90 degrees based on upper limits
            hand_angles = np.clip(hand_angles, 0, 90)  # Ensure angles are within 0-90 degrees
            angles = tuple(int(np.round(hand_angles[i])) for i in range(6)) # Convert to integers for the hand controller
            self.hand.move(angles)

    def _internal_control_arm_qpos(self):
        """Internal control loop for moving the arm towards the target joint positions set by control_arm_qpos."""
        while True:
            time.sleep(self.control_dt)
            if self._arm_should_stop:
                break

            if self.use_arm:
                with self._arm_lock:
                    arm_qpos = self._arm_pos_target

                code, state = self.arm.get_state()
                if code != 0:
                    print(f"*" * 100)
                    print(f"Arm error: {code}")
                    print(f"*" * 100)
                    self.use_arm = False

                if self.arm_mode == 0:
                    safe_control_qpos = self.clip_next_qpos(
                        arm_qpos, velocity_limit=self.arm_velocity_limit
                    )
                    self.arm.set_servo_angle(angle=safe_control_qpos)
                elif self.arm_mode == 1:
                    safe_control_qpos = self.clip_next_qpos(
                        arm_qpos, velocity_limit=self.arm_velocity_limit
                    )
                    self.arm.set_servo_angle_j(angles=safe_control_qpos)
                elif self.arm_mode == 4:
                    code, xarm_state = self.arm.get_joint_states(is_radian=True)
                    arm_current_qpos = xarm_state[0]
                    error = arm_qpos - arm_current_qpos
                    qvel = self.arm_pid.control(error, self.control_dt)
                    safe_qvel = self.clip_arm_velocity(qvel)
                    if np.max(np.abs(safe_qvel)) < 1e-4:   # 速度过小，发送零速度或跳过
                        safe_qvel = np.zeros(7)
                    code = self.arm.vc_set_joint_velocity(safe_qvel)
                    if code != 0:
                        print(f"vc_set_joint_velocity error: code={code}, qvel={safe_qvel}")

    def wait_until_next_control_signal(self):
        """Wait until it's time for the next control signal based on the specified control_dt."""
        if self.last_control_time is None:
            self.last_control_time = time.perf_counter()
        else:
            dt = time.perf_counter() - self.last_control_time
            if dt < self.control_dt:
                time.sleep(self.control_dt - dt)
            else:
                warnings.warn(
                    f"Control dt: {self.control_dt} can not be reached, actual dt: {dt}"
                )
            self.last_control_time = time.perf_counter()

    def stop(self):
        """Stop the robot's movement."""
        if self.use_hand:
            self.hand.emergency_stop(dofs=None)
        if self.use_arm:
            self.arm.vc_set_joint_velocity(np.zeros(7))
            self._arm_should_stop = True
            self._arm_thread.join()
            # self.arm.motion_enable(enable=False)

    def start(self):
        """Start the control threads for the arm and hand."""
        if self.use_arm:
            self._arm_thread.start()

    def get_qpos(self):
        """Get the current joint positions of both the arm and hand."""
        arm_qpos = self.get_arm_qpos() if self.use_arm else np.zeros(7)
        hand_qpos = self.get_hand_qpos() if self.use_hand else np.zeros(6)
        return np.concatenate([arm_qpos, hand_qpos])
    
if __name__ == "__main__":
    """
    测试机械臂在初始位置附近的循环运动。
    轨迹：每个关节相对于初始位置做正弦摆动。
    """
    # 初始关节位置（弧度），与原始 __init__ 中的 initial_arm_qpos 一致
    init_qpos = np.array([0, 0.21, 0, 0.9, 0, -0.75, 0])
    
    # 运动参数
    amplitude = 0.1          # 摆动幅度（弧度），约 5.7°，可根据需要调整
    period = 4.0             # 一个完整周期的时长（秒）
    control_dt = 0.02        # 控制周期，必须与类内部一致
    duration = 60.0          # 总运行时间（秒），若需要无限循环可设为 None
    
    # 初始化机器人（已修正单位）
    print("正在初始化机器人...")
    robot = XArm7QB(
        arm_ip_address="192.168.1.204",   # 根据实际IP修改
        hand_port="COM3",                 # 不需要手部时此参数无影响
        use_arm=True,
        use_hand=False,
        initial_arm_qpos=init_qpos.tolist()
    )
    
    # 启动内部控制线程
    robot.start()
    time.sleep(1.0)   # 等待连接稳定
    
    # 先让机械臂运动到初始位置（若当前不在初始位置）
    print(f"移动到初始位置: {np.round(init_qpos, 3)} rad")
    robot.control_arm_qpos(init_qpos)
    time.sleep(2.0)   # 等待到达，实际可检查到位状态，此处简单延时
    
    # 开始循环运动
    start_time = time.time()
    t = 0.0
    print(f"开始循环运动，周期={period}秒，幅度={amplitude} rad")
    try:
        while True:
            loop_start = time.time()
            
            # 计算当前时刻相对于起始时间的相位
            elapsed = time.time() - start_time
            if duration is not None and elapsed > duration:
                print("运动时间结束，准备停止")
                break
            
            # 生成目标位置：初始位置 + 正弦偏移
            phase = 2 * np.pi * elapsed / period
            # 各关节可设置不同偏移，这里所有关节使用相同相位和幅值
            offset = amplitude * np.sin(phase)
            target = init_qpos + offset   # 所有关节同步摆动
            
            # 发送目标位置给底层控制线程
            robot.control_arm_qpos(target)
            
            # 可选：打印当前状态（为避免刷屏，可降低频率）
            if int(elapsed * 10) % 20 == 0:  # 每2秒打印一次
                current = robot.get_arm_qpos()
                print(f"t={elapsed:.2f}s, 目标: {np.round(target, 3)}, 实际: {np.round(current, 3)}")
            
            # 保持控制频率
            elapsed_loop = time.time() - loop_start
            sleep_time = control_dt - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 控制循环超时警告
                print(f"警告: 控制循环超时 {elapsed_loop:.4f}s > {control_dt}s")
                time.sleep(0.001)  # 避免忙等待
    
    except KeyboardInterrupt:
        print("\n用户中断，正在停止...")
    finally:
        # 运动停止并回到初始位置（可选）
        print("回到初始位置...")
        robot.control_arm_qpos(init_qpos)
        time.sleep(2.0)
        # 停止机械臂控制线程
        robot.stop()
        print("测试结束")

# if __name__ == "__main__":
#     # Example usage
#         init_qpos = np.zeros(7+11)
#         # init_qpos[:7]=[-np.pi/4,-0.32,0.06,1.49,-0.3,0.78,-0.29] # 机械臂初始位置，单位为弧度
#         init_qpos[:7]=[0,0.21,0,0.9,0,-0.75,0]
#         real_robot = XArm7QB(
#             arm_ip_address="192.168.1.204",   # 根据实际修改
#             hand_port="COM3",                 # 根据实际修改
#             use_arm=False,
#             use_hand=True,                    # 如果不需要手部可改为 False
#             initial_arm_qpos=init_qpos[:7].tolist()
#         )
#         real_robot.start()   # 启动机械臂内部控制线程（如果有）
#         # real_robot.control_hand_qpos([-1.79, 0.29, -1.24, -1.24, -1.24, -1.24])  # 将机械臂和机械手移动到初始位置
#         real_robot.control_hand_qpos([0, 0, 0, 0, 0, 0])  # 将机械臂和机械手移动到初始位置
#         # real_robot.control_hand_qpos([-1.2, 0.29, -0.5, 0, -0.8, 0])  # 将机械臂和机械手移动到初始位置

# if __name__ == "__main__":
#     init_qpos = np.zeros(7+11)
#     init_qpos[:7] = [0, 0.21, 0, 0.9, 0, -0.75, 0]
    
#     print("正在初始化机器人...")
#     real_robot = XArm7QB(
#         arm_ip_address="192.168.1.204",
#         hand_port="COM3",
#         use_arm=True,
#         use_hand=False,
#         initial_arm_qpos=init_qpos[:7].tolist()
#     )
    
#     print("启动控制线程...")
#     real_robot.start()
    
#     # 等待系统稳定
#     time.sleep(2)
    
#     # 打印初始位置
#     initial_pos = real_robot.get_arm_qpos()
#     print(f"初始位置: {np.round(initial_pos, 3)}")
    
#     # 发送目标位置
#     target_pos = np.array(init_qpos[:7])
#     print(f"目标位置: {np.round(target_pos, 3)}")
#     real_robot.control_arm_qpos(target_pos)
    
#     # 监控位置变化
#     print("开始监控位置变化...")
#     try:
#         while True:
#             current_pos = real_robot.get_arm_qpos()
#             print(f"当前位置: {np.round(current_pos, 3)}")
            
#             # 检查是否到达目标
#             if np.allclose(current_pos, target_pos, atol=0.01):
#                 print("已达到目标位置！")
            
#             # 检查是否回到零位
#             if np.allclose(current_pos, np.zeros(7), atol=0.01):
#                 print("警告：机械臂回到零位！")
            
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("\n停止机器人...")
#         real_robot.stop()
#         print("机器人已停止")
    