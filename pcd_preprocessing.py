import open3d as o3d
import os

def preprocess_pcd(pcd, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0):
    """
    PCD 전처리 (1. Voxel Downsampling, 2. Outlier Removal)
    Args:
        pcd (o3d.geometry.PointCloud): 원본 포인트 클라우드
        voxel_size (float): 다운샘플링 크기
        nb_neighbors (int): 아웃라이어 제거용 이웃 포인트 수
        std_ratio (float): 아웃라이어 제거 기준 (표준편차 비율)
    Returns:
        o3d.geometry.PointCloud: 전처리된 PCD
    """
    # 1. 다운샘플링
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2. 아웃라이어 제거 (Statistical Outlier Removal)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return pcd

# ========================== 실행 코드 ==========================

for i in range(4,27):
    # PCD 파일 로드
    pcd = o3d.io.read_point_cloud(f"./LiDAR/LiDAR/data/PCD_data/cloud_{i}.pcd")  # 파일 경로를 원하는 걸로 바꿔줘

    # 전처리 실행
    processed_pcd = preprocess_pcd(pcd, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0)

    # preprocessing pcd 저장
    os.makedirs("./LiDAR/LiDAR/data/PCD_data/processed", exist_ok=True)  # 저장할 디렉토리 생성
    o3d.io.write_point_cloud(f"./LiDAR/LiDAR/data/PCD_data/processed/processed_output_{i}.pcd", processed_pcd)

# 시각화
# o3d.visualization.draw_geometries([processed_pcd])
