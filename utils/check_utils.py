"""
for vis_.py scripts
"""

import numpy as np
import open3d as o3d
import trimesh

def setup_gripper():
    gripper_marker = create_gripper_marker()
    grasp_center_point = get_grasp_center_point()
    return gripper_marker, grasp_center_point

def create_o3d_point_cloud(point_cloud, color=[0, 0, 1]):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    point_cloud_o3d.paint_uniform_color(color)
    return point_cloud_o3d
def create_gripper_marker(color=[0, 0, 1], sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.
    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.
    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.0002,
        sections=sections,
        height=0,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.0002,
        sections=sections,
        height=0,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(radius=0.0002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]], height=0)
    cb2 = trimesh.creation.cylinder(
        radius=0.0002, sections=sections, segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]], height=0
    )
    gripper_mesh = trimesh_to_open3d(cfl)
    gripper_mesh += trimesh_to_open3d(cfr)
    gripper_mesh += trimesh_to_open3d(cb1)
    gripper_mesh += trimesh_to_open3d(cb2)
    # gripper_mesh.color = color
    gripper_mesh = gripper_mesh.paint_uniform_color(color)

    return gripper_mesh

def trimesh_to_open3d(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    return o3d_mesh

def get_grasp_center_point():
    right_gripper_top = np.asarray([[4.10000000e-02, -7.27595772e-12, 1.12169998e-01]])
    left_gripper_top = np.asarray([[-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]])
    right_gripper_base = np.asarray([[4.10000000e-02, -7.27595772e-12, 6.59999996e-02]])
    left_gripper_base = np.asarray([[-4.10000000e-02, -7.27595772e-12, 6.59999996e-02]])
    center_point = (right_gripper_top + left_gripper_top + right_gripper_base + left_gripper_base) / 4.0
    return center_point
