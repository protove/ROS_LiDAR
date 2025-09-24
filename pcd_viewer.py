import open3d as o3d
import numpy as np

# PCD 파일을 읽어옵니다.
pcd = o3d.io.read_point_cloud("./data/PCD_data/processed/processed_output_4.pcd")
points = np.asarray(pcd.points)

print("전체 포인트 개수:", points.shape[0])
print("X range:", points[:, 0].min(), "~", points[:, 0].max())
print("Y range:", points[:, 1].min(), "~", points[:, 1].max())
print("Z range:", points[:, 2].min(), "~", points[:, 2].max())

# Point Cloud를 시각화합니다.
o3d.visualization.draw_geometries([pcd])