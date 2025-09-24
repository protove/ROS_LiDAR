import cv2
import os

# 저장할 디렉토리 설정
save_directory = "./LiDAR/LiDAR/captured_images"
os.makedirs(save_directory, exist_ok=True)  # 디렉토리가 없으면 생성

# 웹캠 열기 (video0 기준, 다른 경우는 1, 2 등으로 시도)
cap = cv2.VideoCapture(0)

# 카메라 열기 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 이미지 저장 카운터 초기화
counter = 1

print("시작되었습니다. 's' 키를 눌러 사진을 저장하고, 'q' 키를 눌러 종료하세요.")

# 프레임 캡처 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 화면에 표시
    cv2.imshow("Webcam Stream", frame)

    # 키 입력 확인
    key = cv2.waitKey(1) & 0xFF
    
    # 's' 키를 누르면 이미지 저장
    if key == ord('s'):
        # 파일명 형식: image01.jpg, image02.jpg, ...
        filename = os.path.join(save_directory, f"image{counter:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"이미지가 저장되었습니다: {filename}")
        counter += 1
    
    # 'q' 키를 누르면 종료
    elif key == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
print(f"총 {counter-1}장의 이미지가 {save_directory}에 저장되었습니다.")
