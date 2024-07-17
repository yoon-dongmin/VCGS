import torch
import sys
sys.path.append("../")
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
# import random
import copy
import pickle
from utils.sample import Object

class GraspEvaluatorData(BaseDataset):
    def __init__(self, opt,
                 ratio_positive=0.3,
                 ratio_hardnegative=0.4,
                 collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
                 collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
                 collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
                 collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
                 collision_hard_neg_num_perturbations=10):

        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = {}
        self.ratio_positive = self.set_ratios(ratio_positive)
        self.ratio_hardnegative = self.set_ratios(ratio_hardnegative)

        # ---------------------- VCGS ----------------------
        self.collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self.collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self.collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self.collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self.collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations

        for i in range(3):
            assert (collision_hard_neg_min_rotation[i] <= collision_hard_neg_max_rotation[i])
            assert (collision_hard_neg_min_translation[i] <= collision_hard_neg_max_translation[i])

    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            data = self.get_uniform_evaluator_data(path)
        # ---------------------- VCGS ----------------------
        except NoPositiveGraspsException:
            # if self.opt.skip_error:
            #     return None
            # else:
            return self.__getitem__(np.random.randint(0, self.size))

        gt_control_points = utils.transform_control_points_numpy(data[1], self.opt.num_grasps_per_object, mode='rt')
        meta = {}
        meta['pc'] = data[0][:, :, :3]
        meta['grasp_rt'] = gt_control_points[:, :, :3]
        meta['labels'] = data[2]
        # meta['quality'] = data[3]
        # meta['pc_pose'] = data[4]
        # meta['cad_path'] = data[5]
        # meta['cad_scale'] = data[6]
        return meta

    def __len__(self):
        return self.size

    def read_grasp_file(self, path, return_all_grasps=False):
        """only for evaluator"""
        # file_name = path
        # if self.caching and file_name in self.cache:
        #     pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds = copy.deepcopy(self.cache[file_name])
        #     return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds
        pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds = self.read_object_grasp_data(path, return_all_grasps=return_all_grasps)
        # if self.caching:
        #     self.cache[file_name] = (pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds)
        #     return copy.deepcopy(self.cache[file_name])
        return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds

    def get_hard_neg_cadidates(self, clusters, graps):# , camera_poses):
        hard_neg_candidates = []
        # hard_neg_cam_pose = []
        # for cluster, grasp, cam_pose in clusters, graps, camera_poses:
        for cluster, grasp in zip(clusters, graps):
        # for clusters, grasps in zip([positive_clusters, negative_clusters], [pos_grasps, neg_grasps]):
            # pos_grasps.shape
            # (803, 4, 4)
            # neg_grasps.shape
            # (1197, 4, 4)
            # positive_clusters.shape
            # (19, 2)
            # negative_clusters.shape
            # (20, 2)
            # for cluster in clusters:
            # selected_grasp = grasps[cluster[0]][cluster[1]]
            selected_grasp = grasp # grasp[cluster]
            hard_neg_candidates += utils.perturb_grasp(selected_grasp,
                                                       self.collision_hard_neg_num_perturbations,
                                                       self.collision_hard_neg_min_translation,
                                                       self.collision_hard_neg_max_translation,
                                                       self.collision_hard_neg_min_rotation,
                                                       self.collision_hard_neg_max_rotation)
            # hard_neg_cam_pose.append(cam_pose)
            # print("Get Hard Neg Candidates: ", len(hard_neg_candidates))
        return hard_neg_candidates# , hard_neg_cam_pose

    def get_uniform_evaluator_data(self, path):
        # different read_grasp_file function
        pos_grasps, neg_grasps, obj_mesh, point_clouds, camera_poses_for_prerendered_point_clouds = self.read_grasp_file(path, return_all_grasps=True)
        # pos_grasps, neg_grasps, obj_mesh, point_clouds = self.read_grasp_file(path, return_all_grasps=True)

        output_pcs = []
        output_grasps = []
        output_labels = []
        # output_camera_poses = []
        num_positive = int(self.opt.batch_size * self.ratio_positive) # 19 = int(64 * 0.3)
        num_hard_negative = int(self.opt.batch_size * self.ratio_hardnegative) # 25 = int(64 * 0.4)
        num_flex_negative = self.opt.batch_size - num_positive - num_hard_negative # 20 = 64 - 19 - 25
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps)
        negative_clusters = self.sample_grasp_indexes(num_flex_negative, neg_grasps)
        # print("-"*100)
        # print("Number of positive grasps: ", num_positive)
        # print("Number of hard negative grasps: ", num_hard_negative)
        # print("Number of flex negative grasps: ", num_flex_negative)
        # Number of positive grasps:  19
        # Number of hard negative grasps:  25
        # Number of flex negative grasps:  20
        # print("-"*100)
        # print("Positive Clusters: ", positive_clusters) # (19,2) array
        # print("Negative Clusters: ", negative_clusters)
        # print("-"*100)

        # Fill in Positive Examples.
        # hard_neg_candidates_total = []
        # hard_neg_candidates_total.append(self.get_hard_neg_cadidates(positive_clusters, pos_grasps))#, camera_poses_positive))
        # hard_neg_candidates_total.append(self.get_hard_neg_cadidates(negative_clusters, neg_grasps))#, camera_poses_negative))
        hard_neg_candidates = []
        # hard_neg_candidates_cam_pose = []

        for clusters, grasps in zip([positive_clusters, negative_clusters], [pos_grasps, neg_grasps]):
            # pos_grasps.shape
            # (803, 4, 4)
            # neg_grasps.shape
            # (1197, 4, 4)
            # positive_clusters.shape
            # (19, 2)
            # negative_clusters.shape
            # (20, 2)
            for cluster in clusters:
                # selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_grasp = grasps[cluster]
                hard_neg_candidates += utils.perturb_grasp(selected_grasp,
                                                           self.collision_hard_neg_num_perturbations,
                                                           self.collision_hard_neg_min_translation,
                                                           self.collision_hard_neg_max_translation,
                                                           self.collision_hard_neg_min_rotation,
                                                           self.collision_hard_neg_max_rotation)

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        # if path not in self.collision_hard_neg_queue or \
        #         len(self.collision_hard_neg_queue[path]) < num_hard_negative:
        # if path not in self.collision_hard_neg_queue:
        #     self.collision_hard_neg_queue[path] = []
        # hard negatives are perturbations of correct grasps.
        collisions, heuristic_qualities = utils.evaluate_grasps(hard_neg_candidates, obj_mesh)
        # collisions, heuristic_qualities = utils.evaluate_grasps(hard_neg_candidates_total, obj_mesh)
        hard_neg_mask = collisions | (heuristic_qualities < 0.001)
        hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()

        np.random.shuffle(hard_neg_indexes)
        # for index in hard_neg_indexes:
        #     self.collision_hard_neg_queue[path].append(hard_neg_candidates[index])
        # random.shuffle(self.collision_hard_neg_queue[path])

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            selected_grasp = pos_grasps[positive_cluster]#[positive_cluster[0]][positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_labels.append(1)
            # output_camera_poses.append(camera_poses_positive[positive_cluster])

        # Adding hard neg
        # for i in range(num_hard_negative):
        #     breakpoint()
        #     grasp = self.collision_hard_neg_queue[path][i]
        #     output_grasps.append(grasp)
        #     output_labels.append(0)

        for i in range(num_hard_negative):
            grasp = hard_neg_candidates[i]
            # grasp = hard_neg_candidates_total[i]
            output_grasps.append(grasp)
            output_labels.append(0)
        # self.collision_hard_neg_queue[path] = self.collision_hard_neg_queue[path][num_hard_negative:]

        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError('negative clusters should have the same length as num_flex_negative {} != {}'.format(len(negative_clusters), num_flex_negative))

        for negative_cluster in negative_clusters:
            # TODO deleted quality terms
            selected_grasp = neg_grasps[negative_cluster] # [negative_cluster[0]][negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_labels.append(0)
            # output_camera_poses.append(camera_poses_negative[negative_cluster])

        point_cloud_indices = self.sample_point_clouds(self.opt.num_grasps_per_object, point_clouds)
        # output_grasps  362 list - (4, 4) array
        # camera_poses_for_prerendered_point_clouds (900, 4, 4) array
        # camera_poses_for_prerendered_point_clouds[point_cloud_indices] (64, 4, 4) array
        # TODO check the shape of output_grasps. matrix multiplication is not possible.
        output_grasps = np.matmul(camera_poses_for_prerendered_point_clouds[point_cloud_indices], output_grasps)
        output_pcs = point_clouds[point_cloud_indices]  # np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)

        return output_pcs, output_grasps, output_labels

    def sample_point_clouds(self, num_samples, point_clouds):
        num_point_clouds = len(point_clouds)
        if num_point_clouds == 0:
            raise NoPositiveGraspsException
        replace_point_cloud_indices = num_samples > num_point_clouds
        point_cloud_indices = np.random.choice(range(num_point_clouds), size=num_samples, replace=replace_point_cloud_indices).astype(np.int32)
        return point_cloud_indices

    def read_object_grasp_data(self, data_path, return_all_grasps=True):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            grasps = data['grasps/transformations'] # 2000
            point_clouds = np.asarray(data['rendering/point_clouds'])
            camera_poses_for_prerendered_point_clouds = np.asarray(data['rendering/camera_poses']) # 900
            # all_query_points_per_point_cloud = np.asarray(data["query_points/points_with_grasps_on_each_rendered_point_cloud"])
            # grasp_indices_for_every_query_point_on_each_rendered_point_cloud = np.asarray(data["query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"])
            mesh_file = data["mesh/file"]
            mesh_scale = np.asarray(data["mesh/scale"])
            grasps_success = data["grasps/successes"] # flex_qualities

        object_model = Object(mesh_file)
        object_model.rescale(mesh_scale)
        successful_mask = (grasps_success == 1)
        positive_grasp_indexes = np.where(successful_mask)[0]
        negative_grasp_indexes = np.where(~successful_mask)[0]
        positive_grasps = grasps[positive_grasp_indexes, :, :]
        negative_grasps = grasps[negative_grasp_indexes, :, :]
        # print("Positive Grasps: ", positive_grasps.shape[0])
        # print("Negative Grasps: ", negative_grasps.shape[0])
        # TODO matching camera poses for each grasp
        # camera_poses_for_positive_grasps = camera_poses_for_prerendered_point_clouds[positive_grasp_indexes, :, :]
        # camera_poses_for_negative_grasps = camera_poses_for_prerendered_point_clouds[negative_grasp_indexes, :, :]

        def cluster_grasps(grasps, num_clusters = 32):
            """
            Cluster grasps by farthest point sampling.
            """
            cluster_indexes = np.asarray(utils.farthest_points(grasps, num_clusters, utils.distance_by_translation_grasp))
            output_grasps = []
            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                output_grasps.append(grasps[indexes, :, :])
            output_grasps = np.asarray(output_grasps)
            return output_grasps

        if not return_all_grasps:
            positive_grasps = cluster_grasps(positive_grasps)
            negative_grasps = cluster_grasps(negative_grasps)

        return positive_grasps, negative_grasps, object_model, point_clouds, camera_poses_for_prerendered_point_clouds
        # camera_poses_for_positive_grasps, camera_poses_for_negative_grasps

    def sample_grasp_indexes(self, n, grasps):
        """
        n = number of grasps to sample
        grasps = list of grasps (k, 4, 4)
        """
        # nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(grasps)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException
        # grasp_rows = np.random.choice(range(num_clusters), size=n, replace=replace).astype(np.int32)
        grasp_indexes = np.random.choice(range(num_clusters), size=n, replace=replace).astype(np.int32)
        return grasp_indexes

if __name__ == "__main__":
    TRAIN = True
    if TRAIN:
        from options.train_options import TrainOptions
        opt = TrainOptions().parse()
    else:
        # TODO test with TestOptions
        from options.test_options import TestOptions
        opt = TestOptions().parse()

    opt.arch = 'evaluator'
    # opt.caching = False
    opt.dataset_root_folder = '../dataset/grasps'

    from data import VCGS_DataLoader
    dataloader_ = VCGS_DataLoader(opt)
    training_dataset, validation_dataset, test_dataset = dataloader_.split_dataset(opt.dataset_split_ratio)
    training_dataset_loader = dataloader_.create_dataloader(training_dataset, shuffle_batches= opt.serial_batches)

    # Try to fetch a batch of data
    for i, data in enumerate(training_dataset_loader):
        if data is None:
            print("No data returned, possibly due to an error or data filtering.")
            continue

        # Assuming data contains the expected fields after fetching successfully
        print("Data Keys; ", data.keys())
        print(f"Batch {i + 1}")
        print(f"Point Clouds: {data['pc'].shape}")
        print(f"Grasp RT matrices: {data['grasp_rt'].shape}")
        print(f"Labels: {data['labels'].shape}")

        if i == 0:  # For demonstration, only process the first batch
            break