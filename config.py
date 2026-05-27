from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from typing import Union

import numpy as np
import yaml
import os
import enum

# from core.URDF_units import yourdfpy as urdf
# # import yourdfpy as urdf
# from .kinematics.kinematics_adaptor import MimicJointKinematicAdaptor
# from .optimizer import LPFilter
# from .kinematics.robot_wrapper import RobotWrapper

# from core.URDF_units.yourdfpy import DUMMY_JOINT_NAMES

# 修改路径，不作为模块
import URDF_units.yourdfpy as urdf
# import yourdfpy as urdf
from kinematics.kinematics_adaptor import MimicJointKinematicAdaptor
from optimizer import LPFilter
from kinematics.robot_wrapper import RobotWrapper
from URDF_units.yourdfpy import DUMMY_JOINT_NAMES



OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

# 表示要想shadow和mano对齐，则MANO的x轴对应shadow的z轴；MANO的y轴对应shadow的y轴；MANO的z轴对应shadow的-x轴；
OPERATOR2MANO_Afford = np.array(
    [
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ])

class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()

OPERATOR2MANO = {
    HandType.right: OPERATOR2MANO_RIGHT,
    HandType.left: OPERATOR2MANO_LEFT,
}

# UDRF和MANO路径修改
ROBOT_DIR = Path(r"D:\study\VScodes\Retargeting\assets\robots\hands").absolute()
MANO_DIR = Path(r"D:\study\VScodes\manopth\mano").absolute()


# ROBOT_DIR = Path(r"D:\Code\Assets\URDF\hand_urdf").absolute()
# MANO_DIR = Path(r"D:\Code\Assets\mano").absolute()

# XXX = enum.auto()
class RobotName(enum.Enum):
    allegro = enum.auto()
    shadow = enum.auto()
    svh = enum.auto()
    leap = enum.auto()
    ability = enum.auto()
    inspire = enum.auto()
    panda = enum.auto()
    qb = enum.auto() # 添加清瑞博源手
    mimic_qb = enum.auto() # 添加清瑞博源手的模拟关节版本

class RetargetingType(enum.Enum):
    vector = enum.auto()  # For teleoperation, no finger closing prior
    position = enum.auto()  # For offline data processing, especially hand-object interaction data
    dexpilot = enum.auto()  # For teleoperation, with finger closing prior

@dataclass
class RobotMeta:
    name: str
    is_gripper: bool = False

# RobotName.XXX: "XXX",
ROBOT_NAME_MAP = {
    RobotName.allegro: RobotMeta("allegro_hand"),
    RobotName.shadow: RobotMeta("shadow_hand"),
    RobotName.svh: RobotMeta("schunk_svh_hand"),
    RobotName.leap: RobotMeta("leap_hand"),
    RobotName.ability: RobotMeta("ability_hand"),
    RobotName.inspire: RobotMeta("inspire_hand"),
    RobotName.panda: RobotMeta("panda_gripper", is_gripper=True),
    RobotName.qb: RobotMeta("qb"), # 添加qb手
    RobotName.mimic_qb: RobotMeta("mimic_qb"), # 添加qb手的模拟关节版本
}

ROBOT_NAMES = list(ROBOT_NAME_MAP.keys())


def get_default_config_path(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType
) -> Optional[Path]:
    config_path = Path(__file__).parent / "yaml_configs"

    if retargeting_type is RetargetingType.position:
        config_path = config_path / "offline"
    else:
        config_path = config_path / "teleop"

    meta = ROBOT_NAME_MAP[robot_name]
    robot_str = meta.name

    suffix = "_dexpilot.yml" if retargeting_type == RetargetingType.dexpilot else ".yml"
    if meta.is_gripper:
        # Grippers don't usually differentiate hands in filenames
        config_name = f"{robot_str}{suffix}"
    else:
        config_name = f"{robot_str}_{hand_type.name}{suffix}"
    return config_path / config_name





@dataclass
class RetargetingConfig:
    type: str # "vector", "position", "dexpilot"
    urdf_path: str

    # Whether to add free joint to the root of the robot. Free joint enable the robot hand move freely in the space 
    add_dummy_free_joint: bool = False

    # Source refers to the retargeting input, which usually corresponds to the human hand 人体手关节索引（对应机器人手的 link）
    # The joint indices of human hand joint which corresponds to each link in the target_link_names
    target_link_human_indices: Optional[np.ndarray] = None

    # The link on the robot hand which corresponding to the wrist of human hand
    wrist_link_name: Optional[str] = None

    # Position retargeting link names
    target_link_names: Optional[List[str]] = None

    # Vector retargeting link names
    target_joint_names: Optional[List[str]] = None
    target_origin_link_names: Optional[List[str]] = None
    target_task_link_names: Optional[List[str]] = None

    # DexPilot retargeting link names
    finger_tip_link_names: Optional[List[str]] = None

    # Scaling factor for vector retargeting only
    # For example, Allegro is 1.6 times larger than normal human hand, then this scaling factor should be 1.6
    scaling_factor: float = 1.0

    # Optimization parameters
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # DexPilot optimizer parameters
    project_dist: float = 0.03
    escape_dist: float = 0.05

    # Joint limit tag
    has_joint_limits: bool = True

    # Mimic joint tag
    ignore_mimic_joint: bool = False

    # Low pass filter
    low_pass_alpha: float = 0.1

    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = ROBOT_DIR

    # 配置验证
    def __post_init__(self):
        # Retargeting type check
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")
        
        # Resolve URDF path immediately
        self._resolve_urdf_path()
        
        # Delegate validation to specific methods
        if self.type == "vector":
            self._validate_vector()
        elif self.type == "position":
            self._validate_position()
        elif self.type == "dexpilot":
            self._validate_dexpilot()
    def _resolve_urdf_path(self):
        path = Path(self.urdf_path)
        if not path.is_absolute():
            path = (self._DEFAULT_URDF_DIR / path).absolute()
        if not path.exists():
            raise FileNotFoundError(f"URDF path {path} does not exist")
        self.urdf_path = str(path)
    
    # Vector retargeting requires: target_origin_link_names + target_task_link_names
    def _validate_vector(self):
        if not self.target_origin_link_names or not self.target_task_link_names:
            raise ValueError("Vector retargeting requires: target_origin_link_names + target_task_link_names")
        if len(self.target_task_link_names) != len(self.target_origin_link_names):
            raise ValueError("Vector retargeting origin and task links dim mismatch")
        if self.target_link_human_indices is None:
            raise ValueError("Vector retargeting requires: target_link_human_indices")
        if self.target_link_human_indices.shape != (2, len(self.target_origin_link_names)):
            raise ValueError("Vector retargeting link names and link indices dim mismatch")

    # Position retargeting requires: target_link_names + target_link_human_indices
    def _validate_position(self):
        if self.target_link_names is None:
            raise ValueError("Position retargeting requires: target_link_names")
        if self.target_link_human_indices is None:
            raise ValueError("Position retargeting requires: target_link_human_indices")
            
        self.target_link_human_indices = self.target_link_human_indices.squeeze()
        if self.target_link_human_indices.shape != (len(self.target_link_names),):
            raise ValueError("Position retargeting link names and link indices dim mismatch")

    # DexPilot retargeting requires: finger_tip_link_names + wrist_link_name
    def _validate_dexpilot(self):
        if self.finger_tip_link_names is None or self.wrist_link_name is None:
            raise ValueError("DexPilot retargeting requires: finger_tip_link_names + wrist_link_name")
        if self.target_link_human_indices is not None:
                print(
                    "\033[33m",
                    "Target link human indices is provided in the DexPilot retargeting config, which is uncommon.\n"
                    "If you do not know exactly how it is used, please leave it to None for default.\n"
                    "\033[00m",
                )

    # 加载YAML文件 → 解析retargeting节点 → 转字典加载
    @classmethod
    def load_from_file(
        cls, config_path: Union[str, Path], override: Optional[Dict] = None
    ):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)
    # 字典转配置类（支持参数覆盖） → 处理numpy数组类型
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], override: Optional[Dict] = None):
        if "target_link_human_indices" in cfg:
            cfg["target_link_human_indices"] = np.array(cfg["target_link_human_indices"])
        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> "SeqRetargeting":
        # from .retarget_from_video import SeqRetargeting
        from retarget_from_video import SeqRetargeting
        import tempfile
        # Process the URDF with yourdfpy to better find file path
        # 1. 解析URDF文件（临时目录处理，兼容路径问题）
        robot_urdf = urdf.URDF.load(
            self.urdf_path,
            add_dummy_free_joints=self.add_dummy_free_joint,
            build_scene_graph=False,
        )
        urdf_name = self.urdf_path.split(os.path.sep)[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # Load pinocchio model
        # 2. 加载Pinocchio机器人模型（RobotWrapper）
        robot = RobotWrapper(temp_path)

        # Add 6D dummy joint to target joint names so that it will also be optimized
        if self.add_dummy_free_joint and self.target_joint_names is not None:
            self.target_joint_names = DUMMY_JOINT_NAMES + self.target_joint_names
        joint_names = (
            self.target_joint_names if self.target_joint_names is not None else robot.dof_joint_names
        )
        # Initialize Optimizer based on type
        # 3. 初始化对应类型的优化器（Vector/Position/DexPilot）
        optimizer = self._create_optimizer(robot, joint_names)

        # Setup Low Pass Filter
        # 4. 初始化低通滤波器（平滑动作）
        lp_filter = LPFilter(self.low_pass_alpha) if 0 <= self.low_pass_alpha <= 1 else None

        # Handle Mimic Joints
        # 5. 处理模拟关节（Mimic Joint）
        self._setup_mimic_joints(robot, robot_urdf, optimizer, joint_names)
        # 6. 构建最终的重定向器
        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
        )
        return retargeting
    def _create_optimizer(self, robot, joint_names):
        # from .optimizer import (
        #     VectorOptimizer,
        #     PositionOptimizer,
        #     DexPilotOptimizer,
        # )
        from optimizer import (
            VectorOptimizer,
            PositionOptimizer,
            DexPilotOptimizer,
        )
        
        """Helper to instantiate the correct optimizer."""
        if self.type == "position":
            return PositionOptimizer(
                robot, joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "vector":
            return VectorOptimizer(
                robot, joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "dexpilot":
            return DexPilotOptimizer(
                robot, joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                project_dist=self.project_dist,
                escape_dist=self.escape_dist,
            )
        else:
            raise RuntimeError(f"Unknown optimizer type: {self.type}")
        
    def _setup_mimic_joints(self, robot, robot_urdf, optimizer, joint_names):
        """Helper to configure mimic joints."""
        if self.ignore_mimic_joint:
            return

        has_mimic, sources, mimics, multipliers, offsets = parse_mimic_joint(robot_urdf)
        if has_mimic:
            adaptor = MimicJointKinematicAdaptor(
                robot,
                target_joint_names=joint_names,
                source_joint_names=sources,
                mimic_joint_names=mimics,
                multipliers=multipliers,
                offsets=offsets,
            )
            optimizer.set_kinematic_adaptor(adaptor)

def get_retargeting_config(config_path: Union[str, Path]) -> RetargetingConfig:
    config = RetargetingConfig.load_from_file(config_path)
    return config

# 解析URDF中的模拟关节（mimic joint） → 返回源关节/模拟关节/乘数/偏移量
def parse_mimic_joint(
    robot_urdf: urdf.URDF,
) -> Tuple[bool, List[str], List[str], List[float], List[float]]:
    mimic_joint_names = []
    source_joint_names = []
    multipliers = []
    offsets = []
    for name, joint in robot_urdf.joint_map.items():
        if joint.mimic is not None:
            mimic_joint_names.append(name)
            source_joint_names.append(joint.mimic.joint)
            multipliers.append(joint.mimic.multiplier)
            offsets.append(joint.mimic.offset)

    return (
        len(mimic_joint_names) > 0,
        source_joint_names,
        mimic_joint_names,
        multipliers,
        offsets,
    )
