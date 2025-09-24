import cv2
import torch
from ultralytics import SAM
import numpy as np
import os
import random

# 모델 로드 (SAM2.1)
model = SAM("sam2.1_l.pt")

# 입력 이미지 로드 (BGR -> RGB 변환)
image_path = "./LiDAR/data/Image_data/20240716_140339.jpg"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 📌 **Automatic Mask Generation (프롬프트 없이 자동 분할)**
results = model(rgb_image, mode="segment")  # 자동 세그멘테이션 모드

# 결과가 리스트 형태이므로 첫 번째 요소를 가져옴
if isinstance(results, list) and len(results) > 0:
    results = results[0]  # 리스트의 첫 번째 요소를 가져옴

# 마스크 데이터 가져오기
masks = results.masks.data.cpu().numpy()  # (N, H, W) 형태

# 저장할 디렉토리 생성
os.makedirs("output", exist_ok=True)

# 🎨 랜덤 색상 마스크 오버레이 생성
mask_overlay = np.zeros_like(image, dtype=np.uint8)

for mask in masks:
    mask = (mask * 255).astype(np.uint8)  # 0,1 값을 0~255로 변환

    # 🎨 랜덤한 RGB 색상 생성
    color = [random.randint(50, 255) for _ in range(3)]
    
    # 채널 확장 후 컬러 마스크 적용
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for j in range(3):
        colored_mask[:, :, j] = mask * (color[j] / 255.0)

    # 원본 이미지와 합성
    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.5, 0)
    

# 원본 이미지와 마스크 오버레이 결합
overlay_result = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
# final_result = cv2.imread(r"C:\Users\sbdl1\OneDrive\Desktop\Code\depth_pro\depth_final\test18.jpg")
# final_result = cv2.addWeighted(final_result, 0.7, mask_overlay, 0.3, 0)
# 결과 저장
cv2.imwrite("./LiDAR/sam2_output/test1_output_mask.png", mask_overlay)
cv2.imwrite("./LiDAR/sam2_output/test1_output_overlay.png", overlay_result)
# cv2.imwrite("final/test23_final_output_overlay.png", final_result)

# 결과 출력
#cv2.imshow("Automatic Mask", overlay_result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
