import cv2 as cv
import numpy as np
from attendance_list import attendanceList
from train_faces import people


def attendanceCheck():
    for keys in attendanceList.keys():
        if people[label] == keys:
            attendanceList[keys] = True
        else:
            pass

    for key, value, in attendanceList.items():
        if value:
            print(f'{key} : Present')
        else:
            print(f'{key} : Absent')


haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(face_roi)

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.putText(frame, f'{people[label]}', (x, y - 6), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)

        cv.imshow('Face Recognition', frame)

    if cv.waitKey(20) & 0xFF == ord('x'):
        break

attendanceCheck()
capture.release()
cv.destroyAllWindows()
