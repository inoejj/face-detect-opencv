import numpy as np
import cv2
import os
import pickle

#get cv2 xml in data
path_cv2 = os.path.dirname(cv2.__file__)
path_xml = os.path.join(path_cv2,'data','haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(path_xml)

#implement recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
#load trained recognizer
recognizer.read('trainer.yml')

labels = {}
#load label
with open('labels.pickle','rb') as f:
    loadded_labels = pickle.load(f)
    labels = {v:k for k,v in loadded_labels.items()} #to invert them

#load from webcam
cap = cv2.VideoCapture(0)

while(True):

    #capture frame by frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            cv2.putText(frame, str(int(conf)), (x, y+h), font,0.5, color, stroke, cv2.LINE_AA)

        #draw roi
        color = (255,0,0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)

    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


#when done, release cap
cap.release()
cv2.destroyAllWindows()