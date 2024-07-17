import os
import pickle
from utils import check_utils
import open3d as o3d
import copy
import numpy as np

def get_pickle_data_path():
    data_path = os.path.join(ROOT_PATH, "dataset/generated_dataset/grasps")
    pickle_files = [file for file in os.listdir(data_path) if file.endswith(".pickle")]
    print("Load {} files".format(len(pickle_files)))
    return pickle_files

def read_one_pickle_data(file_path):
    data_path = os.path.join(ROOT_PATH, "dataset/generated_dataset/grasps")
    file_path = os.path.join(data_path, file_path)
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_mesh_from_file(mesh_file, scale=1):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
    mesh.paint_uniform_color([0.768, 0.768, 0.768])
    return mesh

def check_data(data):
    print(data.keys())
    for k in data.keys():
        print("Key: ", k)
        print("Type: ", type(data[k]))
        if type(data[k]) == dict:
            print("Subkeys: ", data[k].keys())
        # print("Value: ", data[k])
        print("")

def preprocess_data(data):
    """
    900 camera point views
    total grasp data 2000
    each cam view has 1024 pc points
    """
    mesh = data["mesh/file"] # one obj file
    object_scale = data["mesh/scale"] # np.array - scalar
    rendered_point_clouds = data["rendering/point_clouds"] # list - 100 x (1024, 3) array
    camera_poses = data["rendering/camera_poses"]  # list - 100 x (4, 4) array
    grasp_transformation_matrices = data["grasps/transformations"] # array - (2000, 4, 4)
    grasp_successes = data["grasps/successes"] # array - (2000,)

    query_points_with_grasps_for_current_point_cloud = data["query_points/points_with_grasps_on_each_rendered_point_cloud"]
    # list - 100 // list - n(<=50==k) : element -> point id (n,) array
    # I_cnt = 0
    # for i in range(len(query_points_with_grasps_for_current_point_cloud)):
    #     if query_points_with_grasps_for_current_point_cloud[i] != []:
    #         I_cnt += 1
    # print("Total number of clusters: ", I_cnt)
    print("===" * 30)
    print("Total number of clusters: ", len(query_points_with_grasps_for_current_point_cloud))
    print("---" * 30)
    print("[0] element of Is: LEN {}, TYPE {}".format(len(query_points_with_grasps_for_current_point_cloud[0]), type(query_points_with_grasps_for_current_point_cloud[0])))
    print("Mean number of points in each cluster: ", np.mean([len(x) for x in query_points_with_grasps_for_current_point_cloud]))
    print("---"*30)
    print("[0][0] element of Is: LEN {}, TYPE {}".format(query_points_with_grasps_for_current_point_cloud[0][0].shape, type(query_points_with_grasps_for_current_point_cloud[0][0])))
    print("Mean number of points in each cluster: ", np.mean([x[0].shape[0] for x in query_points_with_grasps_for_current_point_cloud]))
    # np.mean([x[0].shape[0] for x in query_points_with_grasps_for_current_point_cloud])
    # 476.3511111111111
    # np.mean([x[1].shape[0] for x in query_points_with_grasps_for_current_point_cloud])
    # 204.18666666666667
    # np.mean([x[2].shape[0] for x in query_points_with_grasps_for_current_point_cloud])
    # 354.5388888888889
    # np.mean([x[3].shape[0] for x in query_points_with_grasps_for_current_point_cloud])
    # 376.06888888888886

    grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud = data["query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"]
    # list - 100 // list - n(<=50) : element -> grasp id (1,) array
    # G_cnt = 0
    # for i in range(len(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud)):
    #     if grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud[i] != []:
    #         G_cnt += 1
    # print("Total number of grasps: ", G_cnt)
    print("==="*30)
    print("Total number of grasps: ", len(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud))
    print("---" * 30)
    print("[0] element of Gs: LEN {}, TYPE {}".format(len(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud[0]), type(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud[0])))
    print("Mean number of grasps in each cluster: ", np.mean([len(x) for x in grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud]))
    print("---" * 30)
    print("[0][0] element of Gs: LEN {}, TYPE {}".format(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud[0][0].shape, type(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud[0][0])))
    print("Mean number of grasps in each cluster: ", np.mean([x[0].shape[0] for x in grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud]))
    # np.mean([x[0].shape[0] for x in grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud])
    # 54.986666666666665
    # np.mean([x[1].shape[0] for x in grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud])
    # 30.57888888888889
    # np.mean([x[2].shape[0] for x in grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud])
    # 43.38444444444445
    # np.mean([x[3].shape[0] for x in grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud])
    # 44.007777777777775

    # Load 21 files
    # constrained_Bird_6f15389d26f166488976b8df93909268_0.0015938256576545377.pickle
    # ==========================================================================================
    # Total number of clusters:  900
    # ------------------------------------------------------------------------------------------
    # [0] element of Is: LEN 50, TYPE <class 'list'>
    # Mean number of points in each cluster:  46.9
    # ------------------------------------------------------------------------------------------
    # [0][0] element of Is: LEN (946,), TYPE <class 'numpy.ndarray'>
    # Mean number of points in each cluster:  476.3511111111111
    # ==========================================================================================
    # Total number of grasps:  900
    # ------------------------------------------------------------------------------------------
    # [0] element of Gs: LEN 50, TYPE <class 'list'>
    # Mean number of grasps in each cluster:  46.9
    # ------------------------------------------------------------------------------------------
    # [0][0] element of Gs: LEN (93,), TYPE <class 'numpy.ndarray'>
    # Mean number of grasps in each cluster:  54.986666666666665

    assert len(query_points_with_grasps_for_current_point_cloud) == len(grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud)

    return (mesh, object_scale, rendered_point_clouds,  camera_poses,
            grasp_transformation_matrices, grasp_successes,
            query_points_with_grasps_for_current_point_cloud, grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud)


def unpack_data(mesh, object_scale, pcs, cam_poses,
                grasp_tm, Is, Gs, vis_success_grasps=True, vis_fail_grasps=True):
    o3d_mesh = load_mesh_from_file(mesh, object_scale) # ONE object
    # total 100 views
    for view_id, (pc, cam_pose, query_points_cluster, grasp_cluster) in enumerate(zip(pcs, cam_poses, Is, Gs)): # each view
        print("------ Camera View: ", view_id)
        o3d_mesh_transformed = copy.deepcopy(o3d_mesh)
        o3d_mesh_transformed.transform(cam_pose)  # Apply transform
        visualize(o3d_mesh_transformed, query_points_cluster, pc, grasp_tm, grasp_cluster, cam_pose, vis_success_grasps, vis_fail_grasps)


def visualize(o3d_mesh_transformed, query_points_cluster, pc, grasp_tm, grasp_cluster, cam_pose, vis_success_grasps, vis_fail_grasps):
    # 1 of 100 clusters
    render_limit = 0
    print("------ ------ This view has {} grasps".format(len(query_points_cluster)))
    for grasp_id, (ii, gg) in enumerate(zip(query_points_cluster, grasp_cluster)):
        print("------ ------ ------ Grasp Index: ", grasp_id)
        # gripper_visualizer.vertices = original_gripper.vertices
        visualize_all_grasps_per_query_point(o3d_mesh_transformed,
                                             pc,
                                             ii,
                                             grasp_tm,
                                             gg,
                                             cam_pose,
                                             vis_success_grasps,
                                             vis_fail_grasps)
        render_limit += 1
        if render_limit == 5: # visualize only 5 clusters
            break

def visualize_all_grasps_per_query_point(mesh, object_point_cloud,
                                         query_point_with_grasps, grasp_tm,
                                         grasp_indices, camera_pose, vis_success_grasps, vis_fail_grasps):
    all_grippers = []
    print("===" * 30)
    print("Success grasps: ", len(grasp_indices))
    print("Fail grasps: ", grasp_tm.shape[0] - len(grasp_indices))

    fail_list = [x for x in range(grasp_tm.shape[0]) if x not in list(grasp_indices)]
    assert len(grasp_indices) + len(fail_list) == grasp_tm.shape[0]

    if vis_success_grasps:
        cnt_success = 0
        # success grasps
        for grasp_index in grasp_indices:
            # print(cnt_success)
            if cnt_success == 50:
                break
            gripper_visualizer = check_utils.create_gripper_marker(color=[1, 1, 0])
            gripper_per_index = copy.deepcopy(gripper_visualizer)
            gripper_per_index.transform(grasp_tm[grasp_index])
            gripper_per_index.transform(camera_pose)
            all_grippers.append(gripper_per_index)
            cnt_success += 1

        print("Visualize {} SUCCESS grasps of total {}".format(cnt_success, len(grasp_indices)))

    if vis_fail_grasps:
        cnt_fail = 0
        # fail grasps
        for fail_index in fail_list:
            # print(cnt_fail)
            if cnt_fail == 50:
                break
            # fail_index = grasp_tm.shape[0] - grasp_index
            gripper_visualizer = check_utils.create_gripper_marker(color=[0, 0, 1])
            gripper_per_index = copy.deepcopy(gripper_visualizer)
            gripper_per_index.transform(grasp_tm[fail_index])
            gripper_per_index.transform(camera_pose)
            all_grippers.append(gripper_per_index)
            cnt_fail += 1
        print("Visualize {} FAIL grasps of total {}".format(cnt_fail, len(fail_list)))

    print("===" * 30)

    # visualize query point with grasps
    o3d_point_cloud = check_utils.create_o3d_point_cloud(object_point_cloud, color=[0, 1, 0])
    if query_point_with_grasps.size > 1:
        for point_with_grasp in query_point_with_grasps:
            o3d_point_cloud.colors[point_with_grasp] = [1, 0, 0]
    else:
        o3d_point_cloud.colors[query_point_with_grasps] = [1, 0, 0]

    geometries_to_draw = [mesh, o3d_point_cloud] + all_grippers
    o3d.visualization.draw_geometries(geometries_to_draw)


if __name__ == "__main__":
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    print(ROOT_PATH)
    generated_dataset_path = get_pickle_data_path()
    print(generated_dataset_path[0])
    one_data = read_one_pickle_data(generated_dataset_path[0])

    mesh, object_scale, pcs, cam_poses, grasp_tm, grasp_success, Is, Gs = preprocess_data(one_data)

    VIS_SUCCESS = True
    VIS_FAIL = True
    unpack_data(mesh, object_scale, pcs, cam_poses, # mesh, object_scale, pcs, cam_poses,
                grasp_tm, Is, Gs, # grasp_tm, Is, Gs, vis_success_grasps=True, vis_fail_grasps=True
                vis_success_grasps=VIS_SUCCESS, vis_fail_grasps=VIS_FAIL)