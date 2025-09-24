#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import time
import os
import cv2

# 전역 리스트 초기화
point_list = []

def callback(msg):
    global point_list
    pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(pc))
    if len(points) > 0:
        point_list.append(points)

def get_start_index(pcd_dir, image_dir):
    def extract_index(filenames, prefix, ext):
        indices = []
        for name in filenames:
            if name.startswith(prefix) and name.endswith(ext):
                try:
                    idx = int(name.replace(prefix, "").replace(ext, ""))
                    indices.append(idx)
                except:
                    continue
        return max(indices) + 1 if indices else 0

    pcd_files = os.listdir(pcd_dir)
    img_files = os.listdir(image_dir)

    pcd_start = extract_index(pcd_files, "cloud_", ".pcd")
    img_start = extract_index(img_files, "image_", ".jpg")

    return max(pcd_start, img_start)

def collect_and_save_pointcloud_and_image(topic_name, duration=3, repeat=1,
                                          pcd_dir="../data/PCD_data", image_dir="../data/Image_data"):
    global point_list

    # 저장 폴더 생성
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # 다음 저장할 인덱스 계산
    start_index = get_start_index(pcd_dir, image_dir)

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    # ROS 노드 시작 및 구독
    rospy.init_node('pointcloud_image_saver', anonymous=True)
    rospy.Subscriber(topic_name, PointCloud2, callback)
    rospy.sleep(1.0)  # 토픽 연결 안정화

    for i in range(repeat):
        idx = start_index + i
        point_list = []  # 초기화
        print(f"\n[{idx}] 포인트클라우드 수집 시작... ({duration}초 동안)")

        start_time = time.time()
        while time.time() - start_time < duration and not rospy.is_shutdown():
            rospy.sleep(0.1)

        if len(point_list) == 0:
            print(f"[{idx}] ❗ 포인트 클라우드가 수신되지 않았습니다.")
            continue

        # 포인트 병합
        merged_points = np.vstack(point_list)
        print(f"[{idx}] ✔ 수집 완료! 포인트 수: {merged_points.shape[0]}")

        # 포인트클라우드 저장
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd_path = os.path.join(pcd_dir, f"cloud_{idx}.pcd")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"[{idx}] 📦 포인트클라우드 저장 완료: {pcd_path}")

        # 웹캠 이미지 캡처
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(image_dir, f"image_{idx}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[{idx}] 📸 이미지 캡처 및 저장 완료: {img_path}")
        else:
            print(f"[{idx}] ❗ 웹캠 이미지 캡처 실패")

    # 종료 처리
    cap.release()
    print("\n✅ 모든 수집 및 저장이 완료되었습니다.")

if __name__ == '__main__':
    try:
        collect_and_save_pointcloud_and_image(
            topic_name="/yrl_pub/yrl_cloud",
            duration=3,
            repeat=1,
            pcd_dir="../data/PCD_data",
            image_dir="../data/Image_data"
        )
    except rospy.ROSInterruptException:
        pass
