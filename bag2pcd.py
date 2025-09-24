import rosbag
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np

bag = rosbag.Bag('2025-03-13-15-57-36.bag.active')
point_cloud_topic = '/velodyne_points'

for topic, msg, t in bag.read_messages(topics=[point_cloud_topic]):
    # PointCloud2 메시지를 numpy로 변환
    pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc_array = np.array(list(pc))

    # Open3D 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)

    # PCD 파일로 저장
    o3d.io.write_point_cloud("output.pcd", pcd)

bag.close()
