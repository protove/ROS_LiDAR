import cv2

# 웹캠 열기 (video0 기준, 다른 경우는 1, 2 등으로 시도)
cap = cv2.VideoCapture(0)

# 카메라 열기 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 프레임 캡처 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 화면에 표시
    cv2.imshow("Webcam Stream", frame)

    # 종료: q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
