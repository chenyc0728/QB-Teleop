from pathlib import Path

from typing import Union
import cv2
import tqdm
import pickle
import numpy as np
from pathlib import Path
from time import time

# from ..config import HandType

import sys
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config import HandType

def capture_webcam(video_path: str, video_capture_device: Union[str, int] = 0):
    """
    Capture video with the camera connected to your computer. Press `q` to end the recording.

    Args:
        video_path: The file path for the output video in .mp4 format.
        video_capture_device: the device id for your camera connected to the computer in OpenCV format.

    """
    cap = cv2.VideoCapture(video_capture_device)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )

    while True:
        ret, frame = cap.read()
        writer.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Recording finished")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def generate_human_data_from_video(video_path: str, output_path: str, hand_type: HandType = HandType.right):
    from ..detector.single_hand_detector import SingleHandDetector
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")
    else:
        data = []
        is_right=HandType.right == hand_type
        detector = SingleHandDetector(hand_type="Right", selfie=False) if is_right else SingleHandDetector(hand_type="Left", selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                _, joint_pos, _, _ = detector.detect(rgb)
                data.append(joint_pos)
                pbar.update(1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(data, f)

        cap.release()
        cv2.destroyAllWindows()

def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal

