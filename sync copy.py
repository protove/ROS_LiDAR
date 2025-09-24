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
    max_y_index = np.argmax(lidar_points[:, 1])         # Y 값이 가장 큰 인덱스
    max_y_point = lidar_points[max_y_index]   
    print(f"lidar_points Y값 범위: min {lidar_points[:,1].min()}, max {lidar_points[:,1].max()}")
    RT = np.hstack((R, T))  # (3, 4)
    P_cam = (RT @ lidar_hom.T).T  # (N, 3)
    print(f"P_cam: {P_cam}")
    max_y_index = np.argmax(P_cam[:, 1])         # Y 값이 가장 큰 인덱스
    max_y_point = P_cam[max_y_index]             # 해당 포인트 좌표
    print(f"해당 포인트 (X, Y, Z): {max_y_point}")
    print(f"P_cam Y값 범위: min {P_cam[:,1].min()}, max {P_cam[:,1].max()}")

    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    
    valid = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"cy: {cy}")
    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)
    print(f"fx: {fx}, fy: {fy}")
    print(f"u: {u}\nv: {v}")
    print(f"len_u: {len(u)}\nlen_v: {len(v)}")
    h, w = image.shape[:2]
    print(f"h: {h}, w: {w}")
    black_image = np.zeros_like(image)
    fov_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u, v, Z = u[fov_mask], v[fov_mask], Z[fov_mask]
    print(f"Z: {Z}")
    print(f"Max_Z: {max(Z)}")
    # print(f"masked_u: {u}\nmasked_v: {v}")
    print(f"len_masked_u: {len(u)}\nlen_masked_v: {len(v)}")
    print(f"max_masked_v: {max(v)}")

    image_with_points = image.copy()
    for i in range(len(u)):
        color = (0, int(255 - Z[i] * 70), int(Z[i] * 70))  # depth 색상
        cv2.circle(image_with_points, (u[i], v[i]), 5, color, -1)
        cv2.circle(black_image, (u[i], v[i]), 5, color, -1)

    return image_with_points, black_image

# ---------- 실행 부분 ----------
for i in range(4, 27):
    image = cv2.imread(f"./LiDAR/LiDAR/data/Image_data/calied_image/image_{i}.jpg")

    # LiDAR 포인트 로드 (.pcd → numpy)
    lidar_points = load_pcd_points(f"./LiDAR/LiDAR/data/PCD_data/processed/processed_output_{i}.pcd")
    print(f"Loaded LiDAR points shape: {lidar_points.shape}")
    print(f"Loaded LiDAR points: {lidar_points}")

    # 카메라 캘리브레이션 파라미터 로딩
    K, D = load_camera_intrinsics_from_yaml("./LiDAR/LiDAR/file_calibration/camera_calibration.yaml")

    # Extrinsic (예시: 단위 행렬 / 수동 지정 가능)
    R = np.R = np.array([
        [0, -1, 0],
        [0,  0, -1],
        [1,  0, 0]
    ])
    T = np.array([[0.0], [0.92], [-0.05]])

    # 시각화
    result_img, black_image = project_lidar_to_image(lidar_points, R, T, K, image)
    cv2.imwrite(f"./LiDAR/LiDAR/output/lidar_image_overlap{i}.jpg", result_img)
    cv2.imwrite(f"./LiDAR/LiDAR/output/black_lidar_image_overlap{i}.jpg", black_image)
# cv2.imshow("Projected LiDAR Points", result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
