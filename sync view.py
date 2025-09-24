import numpy as np
import cv2
import open3d as o3d

def load_camera_intrinsics_from_yaml(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return K, D

def load_pcd_points(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)  # (N, 3)
    return points

def project_lidar_to_image(lidar_points, R, T, K, image):
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # (N, 4)
    RT = np.hstack((R, T))  # (3, 4)
    P_cam = (RT @ lidar_hom.T).T  # (N, 3)

    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    valid = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)
    h, w = image.shape[:2]
    fov_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u, v, Z = u[fov_mask], v[fov_mask], Z[fov_mask]

    # FOV 마스크를 원래 lidar_points 크기로 확장
    full_fov_mask = np.zeros(lidar_points.shape[0], dtype=bool)
    full_fov_mask[np.where(valid)[0]] = fov_mask

    image_with_points = image.copy()
    black_image = np.zeros_like(image)
    for i in range(len(u)):
        color = (0, int(255 - Z[i] * 70), int(Z[i] * 70))  # depth 색상
        cv2.circle(image_with_points, (u[i], v[i]), 5, color, -1)
        cv2.circle(black_image, (u[i], v[i]), 5, color, -1)

    return image_with_points, black_image, full_fov_mask

# LiDAR 포인트 시각화 함수
def visualize_lidar_points_with_projection(lidar_points, fov_mask):
    import open3d as o3d

    # 원래 LiDAR 포인트
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(lidar_points)

    # 투영된 포인트 (FOV 내의 포인트만)
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(lidar_points[fov_mask])

    # 색상 설정 (원래 포인트는 회색, 투영된 포인트는 빨간색)
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
    projected_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색

    # 시각화
    o3d.visualization.draw_geometries([original_pcd, projected_pcd])

# 실행 부분
for i in range(4, 16):
    image = cv2.imread(f"./LiDAR/LiDAR/data/Image_data/calied_image/image_{i}.jpg")
    lidar_points = load_pcd_points(f"./LiDAR/LiDAR/data/PCD_data/processed/processed_output_{i}.pcd")
    K, D = load_camera_intrinsics_from_yaml("./LiDAR/LiDAR/file_calibration/camera_calibration.yaml")
    R = np.array([
        [0, -1, 0],
        [0,  0, -1],
        [1,  0, 0]
    ])
    T = np.array([[0.0], [0.72], [-0.05]])

    # 시각화
    result_img, black_image, full_fov_mask = project_lidar_to_image(lidar_points, R, T, K, image)
    cv2.imwrite(f"./LiDAR/LiDAR/output/lidar_image_overlap{i}.jpg", result_img)
    cv2.imwrite(f"./LiDAR/LiDAR/output/black_lidar_image_overlap{i}.jpg", black_image)

    # LiDAR 포인트 3D 시각화
    # visualize_lidar_points_with_projection(lidar_points, full_fov_mask)