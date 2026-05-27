import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
import cv2
from coord_converter import OPERATOR2MANO_RIGHT, OPERATOR2MANO_LEFT, SAPIEN2MEDIAPIPE

# OPERATOR2MANO_RIGHT = np.array(
#     [
#         [0, 0, -1],
#         [-1, 0, 0],
#         [0, 1, 0],
#     ]
# )

# OPERATOR2MANO_LEFT = np.array(
#     [
#         [0, 0, -1],
#         [1, 0, 0],
#         [0, -1, 0],
#     ]
# )

# SAPIEN2MEDIAPIPE = np.array(
#     [
#         [0, 1, 0],
#         [0, 0, -1],
#         [1, 0, 0]
#     ]
# )

# SAPIEN2MEDIAPIPE = np.array(
#     [
#         [0, -1, 0],
#         [0, 0, -1],
#         [-1, 0, 0]
#     ]
# )


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white", show_coord=False, keypoint_3d=None
    ):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

        if keypoint_2d is not None and show_coord:
            h, w = image.shape[:2]          # 获取图像尺寸
            for i, landmark in enumerate(keypoint_2d.landmark):
                x = int(landmark.x * w)     # 归一化 -> 像素坐标
                y = int(landmark.y * h)
                # 格式化坐标文本，保留两位小数
                if keypoint_3d is None:
                    # text = f"({i}({landmark.x:.2f},{landmark.y:.2f},{landmark.z:.2f}))"
                    text = f"({x:.2f},{y:.2f},{landmark.z:.2f})"
                else:
                    text = f"({i}({keypoint_3d[i,0]:.2f},{keypoint_3d[i,1]:.2f},{keypoint_3d[i,2]:.2f}))"
                # 在关键点右下方绘制白色文本（可根据背景调整颜色）
                cv2.putText(
                    image, text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )

        return image

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None, None # add

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None, None # add

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # Parse 3d keypoint from MediaPipe hand detector
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)

        # Obtain the original wrist 3d pos
        wrist_world = keypoint_3d_array[0].copy()  # 形状 (3,)
        # -> wrist coordinate system
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        # Rotation Matrix from Sapien World to MANO Hand
        Rotation = self.operator2mano.T @ mediapipe_wrist_rot.T # camera2hand @ SAPIEN2MEDIAPIPE
        # Rotation = mediapipe_wrist_rot

        return num_box, joint_pos, keypoint_2d, Rotation, wrist_world

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
