import cv2
import os
import glob
import numpy as np

def load_calibration_from_yaml(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def undistort_images_to_folder(image_folder, yaml_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 보정 파라미터 로드
    mtx, dist = load_calibration_from_yaml(yaml_path)
    
    # ✅ 연속 메모리 배열로 강제 변환 (OpenCV 오류 방지)
    mtx = np.ascontiguousarray(mtx)
    dist = np.ascontiguousarray(dist)

    # 입력 이미지 리스트
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # 필요시 확장자 추가 가능
    if not image_paths:
        print("[ERROR] 이미지가 없습니다.")
        return

    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 보정 행렬 계산
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 왜곡 제거
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # ROI 자르기 (선택사항 – 주석 처리 가능)
        x, y, w_roi, h_roi = roi
        if all(v > 0 for v in [x, y, w_roi, h_roi]):
            dst = dst[y:y + h_roi, x:x + w_roi]

        # 저장
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, dst)

        print(f"[INFO] Saved corrected image: {save_path}")

# 사용 예시
if __name__ == "__main__":
    input_folder = './LiDAR/LiDAR/data/Image_data'        # 원본 이미지 폴더
    yaml_file = './LiDAR/LiDAR/file_calibration/camera_calibration.yaml'           # 보정 파라미터 파일
    output_folder = './LiDAR/LiDAR/data/Image_data/calied_image'            # 저장 폴더

    undistort_images_to_folder(input_folder, yaml_file, output_folder)
