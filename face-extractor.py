import numpy as np
import cv2
import os

#get cv2 xml in data
path_cv2 = os.path.dirname(cv2.__file__)
path_xml = os.path.join(path_cv2,'data','haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(path_xml)

person_name = input('Please enter your name.')

imgpath = os.path.join('images',person_name)

if not os.path.exists(imgpath):
    os.mkdir(imgpath)

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    faces = face_cascade.detectMultiScale(img, 1.5, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


# Initialize Webcam
cap = cv2.VideoCapture(0)

#set count to the image exist in folder
count = len(os.listdir(imgpath))

print('Starting face extraction...')
# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))

        # Save file in specified directory with unique name
        file_name_path = imgpath +'/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100 or count >= 400:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
print("Process completed.")
