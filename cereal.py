import cv2, sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if ret==True:
        gray_eye = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_eyes = eyeCascade.detectMultiScale(gray_eye, 1.3, 5)

        for (x, y, w, h) in detected_eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (x, y), ((x + w,y + h)), (0, 0, 255), 1)
            cv2.line(frame, (x + w, y), ((x, y + h)), (0, 0, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
