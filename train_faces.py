import cv2 as cv
import os
import numpy as np

people = []
for i in os.listdir(r'C:\Programming Projects\Python Projects\_Projects\RECON\Faces'):
    people.append(i)

DIR = r'C:\Programming Projects\Python Projects\_Projects\RECON\Faces'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []


def training():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)

            for (x, y, w, h) in face_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)


training()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
