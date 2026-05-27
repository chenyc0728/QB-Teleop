# for testing
from pathlib import Path
import numpy as np
from pathlib import Path
from time import time
import yaml

from ..config import HandType

import pytest
from ..config import (
    ROBOT_NAMES,
    get_default_config_path,
    RetargetingType,
    HandType,
    RobotName,
    RetargetingConfig,
)
from ..optimizer import VectorOptimizer, PositionOptimizer, Optimizer
from ..kinematics.robot_wrapper import RobotWrapper

from ..retarget_from_video import SeqRetargeting

VECTOR_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right.yml",
    "allegro_left": "teleop/allegro_hand_left.yml",
    "shadow_right": "teleop/shadow_hand_right.yml",
    "svh_right": "teleop/schunk_svh_hand_right.yml",
    "leap_right": "teleop/leap_hand_right.yml",
    "ability_right": "teleop/ability_hand_right.yml",
    "ability_left": "teleop/ability_hand_left.yml",
}
POSITION_CONFIG_DICT = {
    "allegro_right": "offline/allegro_hand_right.yml",
    "shadow_right": "offline/shadow_hand_right.yml",
    "svh_right": "offline/schunk_svh_hand_right.yml",
    "leap_right": "offline/leap_hand_right.yml",
    "ability_right": "offline/ability_hand_right.yml",
}
DEXPILOT_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right_dexpilot.yml",
    "allegro_left": "teleop/allegro_hand_left_dexpilot.yml",
    "shadow_right": "teleop/shadow_hand_right_dexpilot.yml",
    "svh_right": "teleop/schunk_svh_hand_right_dexpilot.yml",
    "leap_right": "teleop/leap_hand_right_dexpilot.yml",
}

ROBOT_NAMES = list(VECTOR_CONFIG_DICT.keys())



class TestOptimizer:
    np.set_printoptions(precision=4)
    config_dir = Path(__file__).parent.parent / "dex_retargeting" / "configs"
    robot_dir = Path(__file__).parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir.absolute()))
    DEXPILOT_ROBOT_NAMES = ROBOT_NAMES.copy()
    DEXPILOT_ROBOT_NAMES.remove(RobotName.ability)

    @staticmethod
    def sample_qpos(optimizer: Optimizer):
        joint_eps = 1e-5
        robot = optimizer.robot
        adaptor = optimizer.adaptor
        joint_limit = robot.joint_limits
        random_qpos = np.random.uniform(joint_limit[:, 0], joint_limit[:, 1])
        if adaptor is not None:
            random_qpos = adaptor.forward_qpos(random_qpos)

        init_qpos = np.clip(
            random_qpos + np.random.randn(robot.dof) * 0.5,
            joint_limit[:, 0] + joint_eps,
            joint_limit[:, 1] - joint_eps,
        )
        return random_qpos, init_qpos

    @staticmethod
    def compute_pin_qpos(
        optimizer: Optimizer, qpos: np.ndarray, fixed_qpos: np.ndarray
    ):
        adaptor = optimizer.adaptor
        full_qpos = np.zeros(optimizer.robot.model.nq)
        full_qpos[optimizer.idx_pin2target] = qpos
        full_qpos[optimizer.idx_pin2fixed] = fixed_qpos
        if adaptor is not None:
            full_qpos = adaptor.forward_qpos(full_qpos)
        return full_qpos

    @staticmethod
    def generate_vector_retargeting_data_gt(
        robot: RobotWrapper, optimizer: VectorOptimizer
    ):
        random_pin_qpos, init_qpos = TestOptimizer.sample_qpos(optimizer)
        robot.compute_forward_kinematics(random_pin_qpos)
        random_pos = np.array(
            [robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices]
        )
        origin_pos = random_pos[optimizer.origin_link_indices]
        task_pos = random_pos[optimizer.task_link_indices]
        random_target_vector = task_pos - origin_pos

        return random_pin_qpos, init_qpos, random_target_vector

    @staticmethod
    def generate_position_retargeting_data_gt(
        robot: RobotWrapper, optimizer: PositionOptimizer
    ):
        random_pin_qpos, init_qpos = TestOptimizer.sample_qpos(optimizer)
        robot.compute_forward_kinematics(random_pin_qpos)
        random_target_pos = np.array(
            [robot.get_link_pose(i)[:3, 3] for i in optimizer.target_link_indices]
        )

        return random_pin_qpos, init_qpos, random_target_pos

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType])
    def test_position_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(
            robot_name, RetargetingType.position, hand_type
        )

        # Note: The parameters below are adjusted solely for this test
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()
        assert isinstance(retargeting.optimizer, PositionOptimizer)

        robot: RobotWrapper = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random position
            random_qpos, init_qpos, random_target_pos = (
                self.generate_position_retargeting_data_gt(robot, optimizer)
            )
            fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

            # Set the initial qpos for retargeting
            retargeting.set_qpos(init_qpos)

            # Optimized position
            computed_qpos = retargeting.retarget(
                random_target_pos, fixed_qpos=fixed_qpos
            )[optimizer.idx_pin2target]

            # Check results
            robot.compute_forward_kinematics(
                self.compute_pin_qpos(optimizer, computed_qpos, fixed_qpos)
            )
            computed_target_pos = np.array(
                [robot.get_link_pose(i)[:3, 3] for i in optimizer.target_link_indices]
            )

            # Position difference
            error = np.mean(
                np.linalg.norm(computed_target_pos - random_target_pos, axis=-1)
            )
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization position error: {np.mean(errors['pos'])}")
        print(
            f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times"
        )
        assert np.mean(errors["pos"]) < 1e-2

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType])
    def test_vector_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(
            robot_name, RetargetingType.vector, hand_type
        )
        if config_path is None:
            return

        # Note: The parameters below are adjusted solely for this test
        # For retargeting from human to robot, their values should remain the default in the retargeting config
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["low_pass_alpha"] = 0
        override["scaling_factor"] = 1.0
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()
        assert retargeting.optimizer.retargeting_type == "VECTOR"

        robot: RobotWrapper = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random vector
            random_qpos, init_qpos, random_target_vector = (
                self.generate_vector_retargeting_data_gt(robot, optimizer)
            )
            fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

            # Using a different method compared to position retargeting to set initial qpos and perform optimization
            init_qpos = init_qpos[optimizer.idx_pin2target]

            # Optimized vector
            computed_qpos = optimizer.retarget(
                random_target_vector, fixed_qpos=fixed_qpos, last_qpos=init_qpos[:]
            )

            # Check results
            robot.compute_forward_kinematics(
                self.compute_pin_qpos(optimizer, computed_qpos, fixed_qpos)
            )
            computed_pos = np.array(
                [robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices]
            )
            computed_origin_pos = computed_pos[optimizer.origin_link_indices]
            computed_task_pos = computed_pos[optimizer.task_link_indices]
            computed_target_vector = computed_task_pos - computed_origin_pos

            # Vector difference
            error = np.mean(
                np.linalg.norm(computed_target_vector - random_target_vector, axis=-1)
            )
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization vector error: {np.mean(errors['pos'])}")
        print(
            f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times"
        )
        assert np.mean(errors["pos"]) < 1e-2

    @pytest.mark.parametrize("robot_name", DEXPILOT_ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType])
    def test_dexpilot_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(
            robot_name, RetargetingType.dexpilot, hand_type
        )
        if config_path is None:
            return

        # Note: The parameters below are adjusted solely for this test
        # For retargeting from human to robot, their values should remain the default in the retargeting config
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["low_pass_alpha"] = 0
        override["scaling_factor"] = 1.0
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()
        assert retargeting.optimizer.retargeting_type == "DEXPILOT"

        robot: RobotWrapper = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random vector
            random_qpos, init_qpos, random_target_vector = (
                self.generate_vector_retargeting_data_gt(robot, optimizer)
            )
            fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

            # Using a different method compared to position retargeting to set initial qpos and perform optimization
            init_qpos = init_qpos[optimizer.idx_pin2target]

            # Optimized vector
            computed_qpos = optimizer.retarget(
                random_target_vector, fixed_qpos=fixed_qpos, last_qpos=init_qpos[:]
            )

            robot.compute_forward_kinematics(
                self.compute_pin_qpos(optimizer, computed_qpos, fixed_qpos)
            )
            computed_pos = np.array(
                [robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices]
            )
            computed_origin_pos = computed_pos[optimizer.origin_link_indices]
            computed_task_pos = computed_pos[optimizer.task_link_indices]
            computed_target_vector = computed_task_pos - computed_origin_pos

            # Vector difference
            error = np.mean(
                np.linalg.norm(computed_target_vector - random_target_vector, axis=-1)
            )
            errors["pos"].append(error)

        tac = time()
        print(
            f"Mean optimization vector error for DexPilot retargeting: {np.mean(errors['pos'])}"
        )
        print(
            f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times"
        )
        assert np.mean(errors["pos"]) < 1e-2

class TestRetargetingConfig:
    config_dir = Path(__file__).parent.parent / "src/dex_retargeting" / "configs"
    robot_dir = Path(__file__).parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir.absolute()))

    config_paths = (
        list(VECTOR_CONFIG_DICT.values())
        + list(POSITION_CONFIG_DICT.values())
        + list(DEXPILOT_CONFIG_DICT.values())
    )

    @pytest.mark.parametrize("config_path", config_paths)
    def test_path_config_parsing(self, config_path):
        config_path = self.config_dir / config_path
        config = RetargetingConfig.load_from_file(config_path)
        retargeting = config.build()
        assert isinstance(retargeting, SeqRetargeting)

    def test_dict_config_parsing(self):
        cfg_str = """
        type: position
        urdf_path: ability_hand/ability_hand_right.urdf
        wrist_link_name: "base_link"

        target_joint_names: ['index_q1', 'middle_q1', 'pinky_q1', 'ring_q1', 'thumb_q1', 'thumb_q2']
        target_link_names: [ "thumb_tip",  "index_tip", "middle_tip", "ring_tip", "pinky_tip" ]

        target_link_human_indices: [ 4, 8, 12, 16, 20 ]

        low_pass_alpha: 1
        """
        cfg_dict = yaml.safe_load(cfg_str)
        config = RetargetingConfig.from_dict(cfg_dict)
        retargeting = config.build()
        assert isinstance(retargeting, SeqRetargeting)

    def test_multi_dict_config_parsing(self):
        cfg_str = """
        - type: vector
          urdf_path: allegro_hand/allegro_hand_right.urdf
          wrist_link_name: "wrist"

          target_joint_names: null
          target_origin_link_names: [ "wrist", "wrist", "wrist", "wrist" ]
          target_task_link_names: [ "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ]
          scaling_factor: 1.6

          # The joint indices of human hand joint which corresponds to each link in the target_link_names
          target_link_human_indices: [ [ 0, 0, 0, 0 ], [ 4, 8, 12, 16 ] ]

          low_pass_alpha: 0.2

        - type: DexPilot
          urdf_path: leap_hand/leap_hand_right.urdf
          wrist_link_name: "base"

          target_joint_names: null
          finger_tip_link_names: [ "thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head" ]
          scaling_factor: 1.6

          low_pass_alpha: 0.2
        """
        cfg_dict_list = yaml.safe_load(cfg_str)
        retargetings = []
        for cfg_dict in cfg_dict_list:
            config = RetargetingConfig.from_dict(cfg_dict)
            retargeting = config.build()
            retargetings.append(retargeting)
            assert isinstance(retargeting, SeqRetargeting)

    @pytest.mark.parametrize("config_path", POSITION_CONFIG_DICT.values())
    def test_add_dummy_joint(self, config_path):
        config_path = self.config_dir / config_path
        override = {"add_dummy_free_joint": False}
        config = RetargetingConfig.load_from_file(config_path, override)
        retargeting = config.build()
        robot = retargeting.optimizer.robot
        original_robot_dof = robot.dof
        original_active_dof = len(retargeting.optimizer.target_joint_names)

        override = {"add_dummy_free_joint": True}
        config = RetargetingConfig.load_from_file(config_path, override)
        retargeting = config.build()
        robot = retargeting.optimizer.robot

        assert robot.dof == original_robot_dof + 6
        assert retargeting.joint_limits.shape == (original_active_dof + 6, 2)
        dummy_joint_names = robot.dof_joint_names[:6]
        for i in range(6):
            assert "dummy" in dummy_joint_names[i]
