# 参考 https://blog.csdn.net/weixin_39284111/article/details/147054228
import coacd
import os
import trimesh
from pathlib import Path
import xml.etree.ElementTree as ET
import copy
import open3d as o3d
from natsort import natsorted  
import numpy as np


def indent(elem, level=0):
    # 缩进（Python3.9+可用 ET.indent）
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def replace_mesh_collision_with_multi_convex(urdf_in: str,urdf_out: str,target_mesh_relpath: str, convex_dir: str, convex_glob: str):
    urdf_in_path  = Path(urdf_in).resolve()
    urdf_out_path = Path(urdf_out).resolve()
    urdf_dir      = urdf_in_path.parent

    tree = ET.parse(str(urdf_in_path))
    root = tree.getroot()

    convex_dir_path = Path(convex_dir).resolve()
    convex_files = natsorted(convex_dir_path.glob(convex_glob))
    if not convex_files:
        raise FileNotFoundError(f"在 {convex_dir_path} 下没有找到 {convex_glob}")

    # 遍历所有 link 的 collision，找到 geometry/mesh[filename=target_mesh_relpath]
    target_hits = 0
    for link in root.findall("link"):
        # 收集当下 link 的 collision 列表（防止迭代时增删问题）
        collisions = list(link.findall("collision"))
        for col in collisions:
            geom = col.find("geometry")
            mesh = (geom is not None) and geom.find("mesh")
            if mesh is None:
                continue

            filename = mesh.get("filename", "")
            # 严格匹配目标 mesh
            if filename != target_mesh_relpath:
                continue

            # 记录原 origin（保持相同局部位姿）
            origin = col.find("origin")

            # 删除旧 collision
            parent = link
            insert_idx = list(parent).index(col)
            parent.remove(col)

            # 逐个插入新 collision（每个都是 link 的直接子节点！）
            for fpath in convex_files:
                new_col = ET.Element("collision")

                if origin is not None:
                    new_col.append(copy.deepcopy(origin))
                else:
                    # 若原本没有 origin，则默认 <origin xyz="0 0 0"/>
                    new_ori = ET.SubElement(new_col, "origin")
                    new_ori.set("xyz", "0 0 0")

                new_geom = ET.SubElement(new_col, "geometry")
                new_mesh = ET.SubElement(new_geom, "mesh")

                # 写路径：reference 模式用相对 URDF 的相对路径；copy_rename 模式已在工程里
                rel = Path(os.path.relpath(fpath, urdf_dir)).as_posix()

                new_mesh.set("filename", rel)

                parent.insert(insert_idx, new_col)
                insert_idx += 1

            target_hits += 1

    if target_hits == 0:
        raise RuntimeError(f"未在 URDF 找到 filename='{target_mesh_relpath}' 的 <collision><mesh/>。")
    else:
        print()

    indent(root)
    urdf_out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(urdf_out_path), encoding="utf-8", xml_declaration=True)
    print(f"[OK] 已替换 {target_hits} 处碰撞体。输出：{urdf_out_path}")

def urdf_decompose(urdf_file, mesh_name_list, urdf_out = None, output_mesh_path = None, visualize = False):
    mesh = []
    def iter_collision_mesh_filenames(urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for link in root.findall('link'):
            for col in link.findall('collision'):
                geom = col.find('geometry')
                if geom is None:
                    continue
                mesh = geom.find('mesh')
                if mesh is None:
                    continue
                fn = mesh.get('filename')
                if not fn:
                    continue
                yield link.get('name', ''), fn  # (link 名, URDF 中写的相对/绝对路径)
    for link_name, fn in iter_collision_mesh_filenames(urdf_file):
        for mesh_name in mesh_name_list:
            if os.path.basename(fn) == mesh_name:
                # 可选：若 URDF 里可能写 package://，先去掉或自行解析
                if fn.startswith('package://'):
                    fn = fn.replace('package://', '', 1)
                p = Path(fn)
                if p.is_absolute():
                    mesh_path = p.resolve()
                else :
                    mesh_path = (Path(urdf_file).resolve().parent / p).resolve()

                mesh.append((link_name, fn, str(mesh_path)))

    stem, ext = os.path.splitext(os.path.basename(urdf_file))

    urdf_out = os.path.join(os.path.dirname(os.path.abspath(urdf_file)), f"{stem}_decompose{ext}") if urdf_out == None else urdf_out
    if not mesh:
        raise ValueError(f'未在 <collision> 中找到 filename 以 "{mesh_name_list}" 结尾的 mesh。')
    else:
        for link_name, fn, mesh_path in mesh:
            print(f'  link={link_name:15s}  filename="{fn}"')
            print(f'    -> 绝对路径: {mesh_path}')

            mesh_name = os.path.basename(mesh_path)
            mesh = trimesh.load(mesh_path, force="mesh")

            # 将加载的网格转换为 coacd 的 Mesh 对象
            mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)

            # max_convex_hull: 最大凸包数量。
            # threshold: 精度阈值。
            # max_iter: 最大迭代次数。
            parts = coacd.run_coacd(mesh_coacd)

            if visualize:
                # 原始网格
                original_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(mesh.vertices),
                    triangles=o3d.utility.Vector3iVector(mesh.faces)
                )
                original_mesh.paint_uniform_color([0.5, 0.5, 0.5])
                o3d.visualization.draw_geometries([original_mesh], window_name="Original Mesh")

                # 凸包分解后的网格
                convex_meshes = []
                for part in parts:
                    vertices = part[0]
                    faces = part[1]
                    convex_mesh = o3d.geometry.TriangleMesh(
                        vertices=o3d.utility.Vector3dVector(vertices),
                        triangles=o3d.utility.Vector3iVector(faces)
                    )
                    convex_mesh.paint_uniform_color(np.random.random(3))
                    convex_meshes.append(convex_mesh)

                o3d.visualization.draw_geometries(convex_meshes, window_name="Convex Decomposition")

            if output_mesh_path:
                out_dir = output_mesh_path
            else:
                out_dir = os.path.dirname(mesh_path)
            os.makedirs(out_dir, exist_ok=True)

            saved_paths = []
            for i, part in enumerate(parts):
                vertices, faces = part
                tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                path = os.path.join(out_dir, f"{os.path.splitext(mesh_name)[0]}_{i:03d}{Path(mesh_name).suffix.lower()}")
                tm.export(path)          # 也可用 .stl / .ply / .glb
                saved_paths.append(path)

            print(f"已保存 {len(saved_paths)} 个凸网格到：{out_dir}")

            # 处理urdf
            replace_mesh_collision_with_multi_convex(urdf_file, urdf_out, fn, out_dir, convex_glob=f"{os.path.splitext(mesh_name)[0]}_*{Path(mesh_name).suffix.lower()}")
            urdf_file = urdf_out

if __name__ == "__main__":
    # urdf_file = r'/home/zyz/zyz_programs/Simulation/sapien/assets/fridge/URDF_Data/urdf/URDF_Data.urdf'            # 原始urdf
    # mesh_name_list = ['Mesh_0.dae','Mesh_4.001.dae']                       # 需要分解的网格
    # # urdf_out = r'/home/zyz/zyz_programs/Simulation/sapien/assets/fridge/URDF_Data/urdf/URDF_Data_1.urdf'   # 输出urdf
    urdf_file = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr\qbr.urdf"
    mesh_name_list = [
        # 机械臂
        'link_base_collision.stl',
        'arm_link1_collision.stl',
        'arm_link2_collision.stl',
        'arm_link3_collision.stl',
        'arm_link4_collision.stl',
        'arm_link5_collision.stl',
        'arm_link6_collision.stl',
        'arm_link7_collision.stl',
        'flange_collision.stl',
        # 灵巧手
        'base_link_collision.stl',
        'link1_collision.stl',
        'link2_collision.stl',
        'link3_collision.stl',
        'link4_collision.stl',
        'link5_collision.stl',
        'link6_collision.stl',
        'link7_collision.stl',
        'link8_collision.stl',
        'link9_collision.stl',
        'link10_collision.stl',
        'link11_collision.stl',
    ]
    # mesh_name_list = ['link_base_visual.stl','arm_link1_visual.stl','arm_link2_visual.stl','arm_link3_visual.stl',
    #                   'arm_link4_visual.stl','arm_link5_visual.stl','arm_link6_visual.stl','arm_link7_visual.stl',
    #                   'flange_visual.stl','link_base_visual.stl','link1_visual.stl','link2_visual.stl',
    #                   'link3_visual.stl','link4_visual.stl','link5_visual.stl','link6_visual.stl',
    #                   'link7_visual.stl','link8_visual.stl','link_9_visual.stl','link10_visual.stl','link11_visual.stl']                       # 需要分解的网格
    urdf_out = r"D:\study\VScodes\Retargeting\assets\robots\assembly\xarm7_qbr_decompose\qbr_decompose.urdf"   # 输出urdf
    visualize = False 
    urdf_decompose(urdf_file, mesh_name_list, visualize = visualize)