import cv2
import numpy as np
import os
import glob
import sys
import time

def print_progress_bar(current, total, bar_length=40):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f"\rProgress: [{arrow}{spaces}] {int(percent * 100)}%")
    sys.stdout.flush()

def calibrate_camera_from_folder(folder_path, save_path='camera_calibration.yaml'):
    CHECKERBOARD = (7, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    images = glob.glob(os.path.join(folder_path, '*.jpg'))

    total_images = len(images)
    if total_images == 0:
        print(f"[ERROR] No images found in {folder_path}")
        return None

    print(f"[INFO] Starting calibration with {total_images} images...\n")

    for idx, fname in enumerate(images, 1):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH +
                        cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

        print_progress_bar(idx, total_images)
        time.sleep(0.01)

    print("\n[INFO] Corner detection complete.")

    if not objpoints:
        print("[ERROR] 체커보드 코너를 감지하지 못했습니다. 캘리브레이션 실패.")
        return None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("[INFO] Camera calibration successful.")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # YAML 파일로 저장
    fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", mtx)
    fs.write("dist_coeffs", dist)
    fs.release()

    print(f"[INFO] Calibration data saved to '{save_path}'")
    return {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

# 사용 예시
if __name__ == "__main__":
    folder_path = './LiDAR/LiDAR/calibration'       # 이미지 폴더 경로
    save_file = './LiDAR/LiDAR/file_calibration/camera_calibration.yaml'         # 저장할 YAML 파일명
    calibrate_camera_from_folder(folder_path, save_file)
