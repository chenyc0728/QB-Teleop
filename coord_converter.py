import numpy as np
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

# camera to world
SAPIEN2MEDIAPIPE = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ]
)

# SAPIEN2MEDIAPIPE = np.array(
#     [
#         [0, 1, 0],
#         [0, 0, -1],
#         [-1, 0, 0]
#     ]
# )

PLOT2SAPIEN = np.array(
    [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
)

HAND_POSE_FIX = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]
)
# CUROBO2SAPIEN = np.array([
#     [0, 1, 0],
#     [0, 0, -1],
#     [1, 0, 0]
# ])

CUROBO_POSE_FIX = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])

ROTATE_Y = np.array(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]
    )

# 实际为OPERATOR2MANO.T
SMPLX2MANO = np.array([
    [0, -1, 0],
    [0, 0, 1],
    [-1, 0 ,0]
])

ROTATE_Z = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]
)
# print(SAPIEN2MEDIAPIPE@SAPIEN2MEDIAPIPE.T)

# sapien相机从-x轴朝下观测时的旋转矩阵
# x backwaard, y leftward, z upward
MEDIAPIPE2SAPIEN1 = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0]
])

# x leftward, y forward, z upward
MEDIAPIPE2SAPIEN2 = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

# 假设相机在人手下方
MEDIAPIPE2SAPIEN3 = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
])
# print("我是最新版本！A 已定义")

YCB_CLASSES = {
1: "002_master_chef_can",
2: "003_cracker_box",
3: "004_sugar_box",
4: "005_tomato_soup_can",
5: "006_mustard_bottle",
6: "007_tuna_fish_can",
7: "008_pudding_box",
8: "009_gelatin_box",
9: "010_potted_meat_can",
10: "011_banana",
11: "019_pitcher_base",
12: "021_bleach_cleanser",
13: "024_bowl",
14: "025_mug",
15: "035_power_drill",
16: "036_wood_block",
17: "037_scissors",
18: "040_large_marker",
19: "051_large_clamp",
20: "052_extra_large_clamp",
21: "061_foam_brick"}

if __name__ == "__main__":
    print(HAND_POSE_FIX@ROTATE_Y)