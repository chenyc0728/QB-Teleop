from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import numpy
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

if not hasattr(numpy, 'int'):
    numpy.int = int
    numpy.float = float
    numpy.complex = complex
    numpy.object = object
    numpy.unicode = str
    numpy.str = str

from dataset import DexYCBVideoDataset
from config import RobotName, HandType
from config import RetargetingConfig
from viewer import RobotHandDatasetSAPIENViewer, HandDatasetSAPIENViewer


def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int):
    dataset = DexYCBVideoDataset(data_root)
    
    # 修复点：如果robots不是列表/元组，先包装成列表
    if not isinstance(robots, (list, tuple)):
        robots = [robots]

    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=False) # headless=True, use_ray_tracing=True
    else:
        viewer = RobotHandDatasetSAPIENViewer(list(robots), HandType.right, headless=False) # headless=True, use_ray_tracing=True

    # Data ID, feel free to change it to visualize different trajectory
    data_id = 0

    sampled_data = dataset[data_id]
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    viewer.load_object_hand(sampled_data)
    viewer.render_dexycb_data(sampled_data, fps)


def main(dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 10):
    data_root = Path(dexycb_dir).absolute()
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")
    viz_hand_object(robots, data_root, fps)

if __name__ == "__main__":
    main(
        dexycb_dir = "D:\study\SRTP_hand\subDexYCB",
        robots = RobotName.qb,
    )