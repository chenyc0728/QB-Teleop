"""Create actors (rigid bodies).

The actor (or rigid body) in Sapien is created through a sapien.ActorBuilder. An
actor is an SAPIEN entity that typically consists of a rigid body component (for
physical simulation) and a visual component (for rendering). Note that can have
multiple collision and visual shapes, and they do not need to correspond.

Concepts:
    - Create an actor by primitives (box, sphere, capsule)
    - Create an actor by mesh files
    - sapien.Pose

"""

import sapien as sapien
from sapien.utils import Viewer
import numpy as np
YCB_object_dir = r"D:/cyc/hand_data/models"

def create_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [4], rgba
        name: name of the actor.

    Returns:
        sapien.Entity
    """
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(
            half_size=half_size, material=sapien.physx.get_default_material()
        )
    )

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(
            half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)

    # name and pose may be changed after added to scene
    # entity.set_name(name)
    # entity.set_pose(pose)

    return entity


def create_box_v2(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [3] or [4], rgb or rgba
        name: name of the actor.

    Returns:
        sapien.Entity
    """
    half_size = np.array(half_size)
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)  # Add collision shape
    builder.add_box_visual(half_size=half_size, material=color)  # Add visual shape
    box: sapien.Entity = builder.build(name=name)
    box.set_pose(pose)
    return box


def create_sphere(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a sphere. See create_box."""
    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius)
    builder.add_sphere_visual(radius=radius, material=color)
    sphere = builder.build(name=name)
    sphere.set_pose(pose)
    return sphere


def create_capsule(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius,
    half_length,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a capsule (x-axis <-> half_length). See create_box."""
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=radius, half_length=half_length)
    builder.add_capsule_visual(radius=radius, half_length=half_length, material=color)
    capsule = builder.build(name=name)
    capsule.set_pose(pose)
    return capsule

import sapien

def create_table(
    scene: sapien.Scene,
    pose: sapien.Pose,
    size,
    height,
    thickness=0.1,
    color=(0.8, 0.6, 0.4),   # 保留颜色参数，用于木质桌面
    name="table",
    mass: float = 100.0,
    is_kinematic=False,
) -> sapien.Entity:
    builder = scene.create_actor_builder()

    # ---------- 创建视觉材质 ----------
    # 1. 木质桌面材质 (颜色使用传入的 color，或自定义)
    wood_mat = sapien.render.RenderMaterial()
    wood_mat.base_color = [color[0], color[1], color[2], 1.0]  # RGB 加 Alpha
    wood_mat.roughness = 0.65        # 粗糙度 0~1，0为镜面，1为完全粗糙（哑光）
    wood_mat.metallic = 0.05         # 金属度 0~1，木材是非金属，值很低
    wood_mat.specular = 0.5          # 高光强度，木材有一定高光但不刺眼


    # 2. 金属桌腿材质 (灰色，带金属光泽)
    leg_mat = wood_mat  # 从木质材质克隆，保持相似的颜色

    # 桌面（使用木质材质）
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])
    tabletop_half_size = [size / 2, size / 2, thickness / 2]
    builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
    builder.add_box_visual(
        pose=tabletop_pose, 
        half_size=tabletop_half_size, 
        material=wood_mat    # 使用材质对象，而不是颜色元组
    )

    # 桌腿（使用金属材质）
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (size - thickness) / 2
            y = j * (size - thickness) / 2
            table_leg_pose = sapien.Pose([x, y, -height / 2])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
            builder.add_box_collision(
                pose=table_leg_pose, half_size=table_leg_half_size
            )
            builder.add_box_visual(
                pose=table_leg_pose, half_size=table_leg_half_size, material=leg_mat
            )
    
    # 惯性和质量设置保持不变...
    cm_pose = sapien.Pose()
    inertia = np.array([mass * 0.1, mass * 0.1, mass * 0.1])
    builder.set_mass_and_inertia(mass, cm_pose, inertia)

    if is_kinematic:
        table = builder.build_kinematic(name=name)
    else:
        table = builder.build(name=name)
    table.set_pose(pose)
    return table
# def create_table(
#     scene: sapien.Scene,
#     pose: sapien.Pose,
#     size,
#     height,
#     thickness=0.1,
#     color=(0.8, 0.6, 0.4),
#     name="table",
#     mass: float = 100.0,
#     is_kinematic=False,
# ) -> sapien.Entity:
#     """Create a table (a collection of collision and visual shapes)."""
#     builder = scene.create_actor_builder()

#     # Tabletop
#     tabletop_pose = sapien.Pose(
#         [0.0, 0.0, -thickness / 2]
#     )  # Make the top surface's z equal to 0
#     tabletop_half_size = [size / 2, size / 2, thickness / 2]
#     builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
#     builder.add_box_visual(
#         pose=tabletop_pose, half_size=tabletop_half_size, material=color
#     )

#     # Table legs (x4)
#     for i in [-1, 1]:
#         for j in [-1, 1]:
#             x = i * (size - thickness) / 2
#             y = j * (size - thickness) / 2
#             table_leg_pose = sapien.Pose([x, y, -height / 2])
#             table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
#             builder.add_box_collision(
#                 pose=table_leg_pose, half_size=table_leg_half_size
#             )
#             builder.add_box_visual(
#                 pose=table_leg_pose, half_size=table_leg_half_size, material=color
#             )
#     # 质心：默认原点
#     cm_pose = sapien.Pose()
#     # 惯性矩阵：自动计算一个合理值（足够大，桌子不会乱晃）
#     inertia = np.array([mass * 0.1, mass * 0.1, mass * 0.1])
    
#     # 三个参数必须都传！
#     builder.set_mass_and_inertia(mass, cm_pose, inertia)

#     if is_kinematic:
#         table = builder.build_kinematic(name=name)
#     else:
#         table = builder.build(name=name)
#     table.set_pose(pose)
#     return table

def create_mesh(
    scene: sapien.Scene,
    pose: sapien.Pose,
    collision_mesh_file: str,
    visual_mesh_file: str,
    size = (1,1,1),  # 缩放尺寸：支持 单个浮点数(统一缩放) / (x,y,z)三元组(非均匀缩放)
    name: str = "mesh",
    static_friction: float = 1.5,   # 静摩擦（越大越难滑动）
    dynamic_friction: float = 1.0,  # 动摩擦
    restitution: float = 0.0,       # 弹性（0=不弹，抓取推荐0）
):
    """
    加载Mesh文件创建Actor，支持缩放控制实际尺寸
    :param size: 缩放系数，float=等比例缩放，tuple=(x_scale, y_scale, z_scale)
    """
    # 标准化缩放参数：直接转成列表/元组，SAPIEN原生支持
    if isinstance(size, (int, float)):
        # 统一缩放：XYZ 轴等比例
        scale = [size, size, size]
    else:
        # 非均匀缩放：XYZ 轴单独设置
        scale = tuple(size)  # 转成元组更稳定

    # 创建【物理材质】→ 控制摩擦力
    physical_material = scene.create_physical_material(
        static_friction,
        dynamic_friction,
        restitution
    )

    # 创建构建器，**碰撞体+视觉体用相同的scale**
    builder = scene.create_actor_builder()
    # 加载碰撞网格（带缩放）
    builder.add_convex_collision_from_file(
        filename=collision_mesh_file,
        scale=scale,  # 控制物理碰撞体尺寸
        material=physical_material  # 应用物理材质
    )
    # 加载视觉网格（带缩放）
    builder.add_visual_from_file(
        filename=visual_mesh_file,
        scale=scale  # 控制视觉显示尺寸
    )

    # 构建Actor并设置位姿
    mesh = builder.build(name=name)
    mesh.set_pose(pose)
    return mesh

def load_YCB_object(
    scene: sapien.Scene,
    pose: sapien.Pose,
    category_id: int,
    size = (1,1,1),
    static_friction: float = 1.5,
    dynamic_friction: float = 1.0,
    restitution: float = 0.0
):
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
    YCB_SIZE = {
    "002_master_chef_can": 0.85, # 圆柱形罐头，默认尺寸不变
    "003_cracker_box": 0.75, # 长方体饼干盒（立起）
    "004_sugar_box": 1.5, # 长方体糖盒
    "005_tomato_soup_can": 1.3, # 圆形罐子
    "006_mustard_bottle": 1.1, # 瓶子
    "007_tuna_fish_can": 1.1, # 扁罐
    "008_pudding_box": 0.95, # 长方体布丁盒（倒下）
    "009_gelatin_box": 1.05, # 长方体果冻盒（倒下）
    "010_potted_meat_can": 1.0, # 长方体肉罐头
    "011_banana": 1.3,
    "019_pitcher_base": 0.8, # 水壶
    "021_bleach_cleanser": 1.0, # 清洁剂瓶子
    "024_bowl": 1.0, # 碗
    "025_mug": 1.1, # 马克杯
    "035_power_drill": 1.2, # 电钻
    "036_wood_block": 0.75, # 木块
    "037_scissors": 1.6, # 剪刀（默认太小了）
    "040_large_marker": 1.2, # 大号记号笔
    "051_large_clamp": 1.0, # 大夹子
    "052_extra_large_clamp": 1.0, # 超大夹子
    "061_foam_brick": 1.4, # 泡沫砖
    }
    YCB_ORIENTATION = {
        "004_sugar_box": (0.707, 0, 0, 0.707), # 旋转90度，使得长边朝向x轴
        "006_mustard_bottle": (0, 0, 0, 1),
        "010_potted_meat_can": (0.707,0,0,0.707),
        "011_banana": (1, 0, 0, 0),
        "019_pitcher_base": (0.383, 0, 0, 0.924),
        "036_wood_block": (1, 0, 0, 0),
    }
    ycb_visual_file = f"{YCB_object_dir}/{YCB_CLASSES[category_id]}/textured.obj"
    ycb_collision_file = f"{YCB_object_dir}/{YCB_CLASSES[category_id]}/textured_simple.obj"
    try:
        size = YCB_SIZE[YCB_CLASSES[category_id]]
    except KeyError:
        size = 1.0  # 默认不缩放
    try:        
        orientation = YCB_ORIENTATION[YCB_CLASSES[category_id]]
        pose.q = orientation
    except KeyError:        
        pose.q = (1, 0, 0, 0)  # 默认不旋转
    return create_mesh(
        scene=scene,
        pose=pose,
        collision_mesh_file=ycb_collision_file,
        visual_mesh_file=ycb_visual_file,
        size=size,
        name=YCB_CLASSES[category_id],
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution
    )

def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # ---------------------------------------------------------------------------- #
    # Add actors
    # ---------------------------------------------------------------------------- #
    scene.add_ground(altitude=0)  # The ground is in fact a special actor.
    box = create_box(
        scene,
        sapien.Pose(p=[0, 0, 1.0 + 0.05]),
        half_size=[0.05, 0.05, 0.05],
        color=[1.0, 0.0, 0.0],
        name="box",
    )
    sphere = create_sphere(
        scene,
        sapien.Pose(p=[0, -0.2, 1.0 + 0.05]),
        radius=0.05,
        color=[0.0, 1.0, 0.0],
        name="sphere",
    )
    capsule = create_capsule(
        scene,
        sapien.Pose(p=[0, 0.2, 1.0 + 0.05]),
        radius=0.05,
        half_length=0.05,
        color=[0.0, 0.0, 1.0],
        name="capsule",
    )
    table = create_table(
        scene,
        sapien.Pose(p=[0, 0, 2.0]),
        size=1.0,
        height=1.0,
    )

    # # add a mesh
    # builder = scene.create_actor_builder()
    # builder.add_convex_collision_from_file(
    #     filename=r"D:\study\VScodes\Retargeting\assets\objects\banana\collision.obj"
    # )
    # builder.add_visual_from_file(filename=r"D:\study\VScodes\Retargeting\assets\objects\banana\visual.glb")
    # mesh = builder.build(name="mesh")
    # mesh.set_pose(sapien.Pose(p=[-0.2, 0, 1.0 + 0.05]))
    banana = load_YCB_object(
        scene,
        pose=sapien.Pose(p=[-0.2, 0, 1.0 + 0.05]),
        size=1.0,
        category_id=21
    )

    # ---------------------------------------------------------------------------- #

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()

    viewer.set_camera_xyz(x=-2, y=0, z=2.5)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
