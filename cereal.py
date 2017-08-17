from network_min import *
import random
import cv2

np.set_printoptions(suppress = True);

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)

roi = [[] for _ in range(4)]

def grab_pixels(x, y, w, h, image):
    roi = [[] for _ in range(4)]
    sample = image[y:y+h, x:x+h]

    count = index = 0
    for col in sample:
        for pixel in col:
            if (count >= (w * h) / 4):
                index += 1
                count = 0

            roi[index].append(pixel)
            count += 1;

def downsample(input, output_size):
    scale = len(input) / output_size
    temp_pixel = [0, 0, 0]
    temp_list = []

    count = 0
    for x in range(len(input)):
        if (count >= scale):
            temp_list.append(temp_pixel)
            count = 0

        for index in range(len(temp_pixel)):
            if (temp_pixel[index] < input[x][index]):
                temp_pixel[index] = input[x][index]
        count += 1

    random_number = 0
    for x in range(output_size - len(temp_list)):
        random_number = random.randint(0, len(temp_list))
        temp_list.insert(random_number, temp_list[random_number])

while True:
    ret, frame = video_capture.read()
    if (ret == True):
        gray_eye = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_eyes = eyeCascade.detectMultiScale(gray_eye, 1.3, 5)

        for (x, y, w, h) in detected_eyes:
            if cv2.waitKey(1) & 0xFF == ord('w'):
                grab_pixels(x, y, w, h, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (x, y), ((x + w,y + h)), (0, 0, 255), 1)
            cv2.line(frame, (x + w, y), ((x, y + h)), (0, 0, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
