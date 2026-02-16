import cv2 as cv

cap = cv.VideoCapture(0) #0 is code for webcamera

while True:
    ret, frame = cap.read() # read frame
    if frame is None:
        break

    cv.imwrite("frame.png", frame) # path where to save and the frame himself

    if cv.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved successfully.")