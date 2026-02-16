import cv2 as cv

device_camera_code = 0
path = r"C:\Users\giley\PycharmProjects\CV_tests\evening.avi"
cap = cv.VideoCapture(
    path
)

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    filename = f"C:\\Users\\giley\\PycharmProjects\\YOLO training\\dataset\\images\\trane\\img_evening{i:04d}.png"
    cv.imwrite(filename, frame)
    i = i + 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if i == 3471:
        break

cap.release()
cv.destroyAllWindows()
