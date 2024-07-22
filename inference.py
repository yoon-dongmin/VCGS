import torch
import utils  # Assuming you have this utility module
#from models import create_model
from options.test_options import TestOptions
import numpy as np
import open3d as o3d


def visualize_blue_points(pointcloud):
    # 포인트 클라우드 데이터에서 점과 색상을 추출합니다.
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # 파란색 포인트를 필터링합니다.
    blue_mask = (colors[:, 2] > 0.5) & (colors[:, 0] < 0.3) & (colors[:, 1] < 0.3)
    blue_points = points[blue_mask]

    # 파란색 포인트 클라우드를 생성합니다.
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)
    blue_pcd.paint_uniform_color([0, 0, 1])  # 파란색으로 칠함

    # 원래 포인트 클라우드와 파란색 포인트 클라우드를 시각화합니다.
    o3d.visualization.draw_geometries([pointcloud, blue_pcd])


def farthest_point_sampling(point_cloud, num_samples):
    N, D = point_cloud.shape
    centroids = np.zeros((num_samples,))
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = point_cloud[farthest, :]
        dist = np.sum((point_cloud - centroid) ** 2, -1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return point_cloud[centroids.astype(np.int32)]

def get_blue_mask(colors):
    return (colors[:, 2] > 0.5) & (colors[:, 0] < 0.3) & (colors[:, 1] < 0.3)

def visualize_point_clouds(point_clouds, window_name="Point Clouds Visualization"):
    o3d.visualization.draw_geometries(point_clouds, window_name=window_name)

# Usage
if __name__ == "__main__":
    # TODO create dataset to put in dataset_test argument
    # opt = TestOptions().parse()
    # opt.serial_batches = True  # no shuffle
    # opt.name = input() # sampler_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2

    # model = create_model(opt)

    # .ply 파일 로드
    main_ply_file = "dataset/20240709_174752_pcd_0.ply"  # 주 포인트 클라우드 파일 경로

    # 포인트 클라우드 로드
    pointcloud = o3d.io.read_point_cloud(main_ply_file)

    # 포인트와 색상 추출
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # 전체 포인트 클라우드 중에서 Farthest Point Sampling을 통해 1024개의 포인트를 선택
    sampled_points = farthest_point_sampling(points, 1024)

    # 샘플링된 포인트 클라우드 생성
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.paint_uniform_color([1, 0, 0])  # 빨간색으로 칠함

    # 샘플링된 포인트에 대응하는 색상 추출
    sampled_indices = [np.where((points == sp).all(axis=1))[0][0] for sp in sampled_points]
    sampled_colors = colors[sampled_indices]

    # 샘플링된 포인트 중에서 파란색 포인트 필터링
    sampled_blue_mask = get_blue_mask(sampled_colors)
    sampled_blue_points = sampled_points[sampled_blue_mask]

    # 깊이 기준으로 정렬하여 윗부분 포인트를 우선적으로 선택
    sorted_sampled_blue_points = sampled_blue_points[np.argsort(-sampled_blue_points[:, 2])]

    # 윗부분 포인트에서 50개의 포인트를 선택
    top_50_blue_points = sorted_sampled_blue_points[:50]

    # 1024개의 포인트에 대해 파란색 영역은 1, 나머지는 0인 값을 갖는 벡터 생성
    blue_indices = np.isin(sampled_points, top_50_blue_points).all(axis=1).astype(int)

    # 상위 50개 파란색 포인트 클라우드 생성 및 색상 지정
    top_50_blue_pcd = o3d.geometry.PointCloud()
    top_50_blue_pcd.points = o3d.utility.Vector3dVector(top_50_blue_points)
    top_50_blue_pcd.paint_uniform_color([0, 0, 1])  # 파란색으로 칠함

    # 각 포인트 클라우드의 개수 출력
    print(f"Total points: {len(points)}")
    print(f"Sampled points: {len(sampled_points)}")
    print(f"Sampled blue points: {len(sampled_blue_points)}")
    print(f"Top 50 blue points: {len(top_50_blue_points)}")

    # 1024개의 포인트의 (x, y, z) 좌표 값을 가지는 벡터와 파란색 영역 여부를 나타내는 벡터 출력
    print("Sampled points (x, y, z):")
    print(sampled_points.shape)
    print("Blue indices (0 or 1):")
    print(blue_indices.shape)

    # 원래 포인트 클라우드와 샘플링된 포인트 클라우드, 상위 50개 파란색 포인트 시각화
    # visualize_point_clouds([top_50_blue_pcd])


