#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import time
import os

# 전역 리스트 초기화 (콜백에서 누적)
point_list = []

def callback(msg):
    global point_list
    # ROS PointCloud2 메시지를 numpy array로 변환
    pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(pc))
    
    if len(points) > 0:
        point_list.append(points)

def collect_and_save_pointcloud(topic_name, duration=3, repeat=10, save_dir="/home/username/pcd_folder"):
    global point_list
    
    # 저장 폴더가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    rospy.init_node('pointcloud_accumulator', anonymous=True)
    rospy.Subscriber(topic_name, PointCloud2, callback)

    rospy.sleep(1.0)  # 토픽 연결 안정화 시간 (1초)

    for i in range(repeat):
        point_list = []  # 이전 포인트 클리어
        print(f"\n[{i+1}/{repeat}] 데이터 수집 시작... {duration}초 동안 수집 중...")

        start_time = time.time()
        
        while time.time() - start_time < duration and not rospy.is_shutdown():
            rospy.sleep(0.1)

        if len(point_list) == 0:
            print(f"[{i+1}/{repeat}] 수집된 데이터가 없습니다!")
            continue

        # 포인트 누적 후 병합
        merged_points = np.vstack(point_list)
        print(f"[{i+1}/{repeat}] 수집 완료! 총 포인트 수: {merged_points.shape[0]}개")

        # Open3D로 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)

        # 저장 경로 및 파일명
        save_path = os.path.join(save_dir, f"merged_3sec_{i}.pcd")

        # PCD 저장
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"[{i+1}/{repeat}] 저장 완료: {save_path}")

    print("\n모든 수집 및 저장이 완료되었습니다!")

if __name__ == '__main__':
    try:
        # 아래 경로와 토픽 이름을 상황에 맞게 수정!
        collect_and_save_pointcloud(
            topic_name="/yrl_pub/yrl_cloud",     # 사용할 ROS 토픽 이름
            duration=3,                          # 각 수집 시간(초)
            repeat=1,                           # 반복 횟수
            save_dir="../data" # 저장 폴더 (꼭 수정해서 사용!)
        )
    except rospy.ROSInterruptException:
        pass
