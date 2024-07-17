# -*- coding: utf-8 -*-
"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import numpy as np

from tqdm import tqdm

import trimesh
import trimesh.transformations as tra
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename, scale=1.0):
        """Constructor.
        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        ROOT_PATH = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
        self.filename = os.path.join(ROOT_PATH, filename)
        self.mesh = trimesh.load(self.filename, force='mesh')
        self.scale = scale  # Use the scale parameter provided during initialization

        # This handles the case when trimesh.load() returns a list of meshes
        if isinstance(self.mesh, list):
            print("Warning: Multiple meshes detected, concatenating into a single mesh.")
            self.mesh = trimesh.util.concatenate(self.mesh)

        # Apply scaling as provided by the user
        if self.scale != 1.0:
            self.mesh.apply_scale(self.scale)

        # Handling trimesh.Scene to trimesh.Trimesh conversion
        self.as_mesh()

        # Setting up the collision manager
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale):
        """Set scale of object mesh and apply it."""
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def as_mesh(self):
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces) for m in self.mesh.geometry.values()])


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""
    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder='..'):
        """Create a Franka Panda parallel-yaw gripper object.
        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.default_pregrasp_configuration = 0.04
        if q is None:
            q = self.default_pregrasp_configuration
        self.q = q
        # fn_base = root_folder + '/gripper_models/panda_gripper/hand.stl'
        # fn_finger = root_folder + '/gripper_models/panda_gripper/finger.stl'
        ROOT_PATH = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
        fn_base = os.path.join(ROOT_PATH, 'gripper_models/panda_gripper/hand.stl')
        fn_finger = os.path.join(ROOT_PATH, 'gripper_models/panda_gripper/finger.stl')

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r]) # 두 손가락 합치기
        self.hand = trimesh.util.concatenate([self.fingers, self.base]) # 손 전체 모델 생성

        self.ray_origins = []
        self.ray_directions = []

        # 손가락 당 접촉점에 대한 광선 원점과 방향 계산
        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            self.ray_origins.append(
                np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1])
            self.ray_origins.append(
                np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1])
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]])
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)
        # print(">>> ray origins: ")
        # print(self.ray_origins)
        # print(">>> ray directions: ")
        # print(self.ray_directions)
        self.standoff_range = np.array([max(self.finger_l.bounding_box.bounds[0, 2], self.base.bounding_box.bounds[1, 2]),self.finger_l.bounding_box.bounds[1, 2]])
        self.standoff_range[0] += 0.001

    def get_closing_rays(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.
        Arguments:
            transform {[numpy.array]} -- a 4x4 homogeneous matrix
        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T


def create_gripper(name, configuration=None, root_folder=''):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object(name = 'object', mesh = object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(np.min([manager.min_distance_single(gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))
    return [d == 0 for d in min_distance], min_distance


def grasp_quality_point_contacts(transforms, collisions, object_mesh, gripper_name='panda', silent=False):
    """Grasp quality function

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of float -- quality of grasps [0..1]
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree: # embree 가속 라이브러리가 있을 경우, 더 빠른 교차점 계산을 위해 사용
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    for p, colliding in tqdm(zip(transforms, collisions), total=len(transforms), disable=silent):
        if colliding: # 충돌이 있는 경우, 결과에 -1 추가
            res.append(-1)
        else: # 충돌이 없는 경우, 그리퍼가 닫히는 방향의 광선을 계산
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)

            if len(locations) == 0:
                res.append(0)
            else:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(ray_origins[index_ray] - locations, axis=1) < 2.0 * gripper.q

                if sum(valid_locations) == 0:
                    res.append(0)
                else:
                    contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    motion_normals = ray_directions[index_ray[valid_locations]]
                    dot_prods = (motion_normals * contact_normals).sum(axis=1)
                    res.append(np.cos(dot_prods).sum() / len(ray_origins))
    return res


def visualize_gripper_and_rays(gripper, rays_origins, rays_directions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot gripper mesh edges
    for mesh in [gripper.base, gripper.finger_l, gripper.finger_r]:
        edges = mesh.edges_unique
        edge_points = mesh.vertices[edges]
        for edge in edge_points:
            ax.plot3D(*zip(*edge), color='b')

    # Plot rays
    for origin, direction in zip(rays_origins, rays_directions):
        end_point = origin[:3] + direction[:3] * 0.1  # Adjust multiplier as needed
        ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]], color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def test_gripper():
    # Initialize the PandaGripper object
    panda_gripper = PandaGripper(root_folder="../")

    # Check if the base and fingers were loaded correctly
    assert panda_gripper.base is not None, "Base mesh not loaded."
    assert panda_gripper.finger_l is not None, "Left finger mesh not loaded."
    assert panda_gripper.finger_r is not None, "Right finger mesh not loaded."

    # Test get_closing_rays method
    # Create a simple transform (identity matrix, meaning no transformation)
    identity_transform = np.eye(4)
    rays_origins, rays_directions = panda_gripper.get_closing_rays(identity_transform)

    # Check if rays are generated
    assert len(rays_origins) > 0, "No ray origins generated."
    assert len(rays_directions) > 0, "No ray directions generated."
    assert len(rays_origins) == len(rays_directions), "Mismatch in number of ray origins and directions."

    # Optionally, apply a more complex transformation and test again
    # For example, rotate the gripper 45 degrees around the z-axis
    rotation_degrees = 45
    rotation_radians = np.deg2rad(rotation_degrees)
    rotation_matrix = R.from_euler('z', rotation_radians).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transformed_rays_origins, transformed_rays_directions = panda_gripper.get_closing_rays(transform)

    # Perform any additional checks you deem necessary
    print("Basic tests passed.")

    # Add visualization here
    rays_origins, rays_directions = panda_gripper.get_closing_rays(identity_transform)
    visualize_gripper_and_rays(panda_gripper, rays_origins, rays_directions)

    print("Visualization complete.")

if __name__ == "__main__":
    # test_gripper()

    import os
    import pickle
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(ROOT_PATH, 'dataset/generated_dataset/constrained_6Shelves_aa7c53c8744d9a24d810b14a81e12eca_0.003885597554766574.pickle')

    # read_object_grasp_data function
    breakpoint()
    json_dict = pickle.load(open(data_path, "rb"))
    # TODO test load object
    a_object = Object(json_dict['mesh/file'])