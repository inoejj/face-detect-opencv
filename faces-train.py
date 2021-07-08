import os
from PIL import Image
import numpy as np
import cv2
import pickle


basedir = os.path.dirname(os.path.abspath(__file__))
imagedir = os.path.join(basedir,'images')

#get cv2 xml in data
path_cv2 = os.path.dirname(cv2.__file__)
path_xml = os.path.join(path_cv2,'data','haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(path_xml)

#recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

curr_id = 0
label_id = {}
y_labels = []
x_train = []


#collect x_train
print('Collecting x train...\n')
for root,dirs,files in os.walk(imagedir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(' ', '_').lower()
            if label in label_id:
                pass
            else:
                label_id[label] = curr_id
                curr_id += 1

            id_ = label_id[label]

            pil_image = Image.open(path).convert('L') # bgr to gray
            size = (550,550)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image,'uint8')

            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open('labels.pickle','wb') as f:
    pickle.dump(label_id,f)

print('Applying data augmentation on datasets...\n')
#create more data by data augmentation
from skimage.transform import rotate
from skimage.util import random_noise
from tqdm import tqdm

final_train_data = []
final_target_train = []

for i in tqdm(range(len(x_train))):
    final_train_data.append(x_train[i])
    final_train_data.append(rotate(x_train[i], angle=45, mode= 'wrap'))
    final_train_data.append(np.fliplr(x_train[i]))
    final_train_data.append(np.flipud(x_train[i]))
    final_train_data.append(random_noise(x_train[i],var=0.2**2))
    for j in range(5):
        final_target_train.append(y_labels[i])


#train using opencv
print('Starting training...\n')
recognizer.train(final_train_data,np.array(final_target_train))
recognizer.save('trainer.yml')
print('Training completed.')
