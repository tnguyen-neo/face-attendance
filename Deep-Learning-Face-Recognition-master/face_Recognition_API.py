# Face Recognition

# Importing the libraries
import cv2
from keras.models import load_model
import requests
import numpy as np
import json

#model = load_model('facy_resize_color_model.h5')
model = load_model('facefeatures_model_03-03-2021.h5')

#load json
file = open('data.json', encoding="utf8")
data = json.load(file)
# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

names = []
email = []
for i in data['student']: 
    #print(i["studentRollID"]) 
    names.append(i["studentName"])
    email.append(i["studentEmail"])
#names = ['Tri', 'Thuan', 'Linh', 'Tinh', 'Nguyen']


urlCheckin = ''
urlCheckout =''
# Doing some Face Recognition with the webcam
videoCheckin = cv.VideoCapture(urlCheckin)

videoCheckout = cv.VideoCapture(urlCheckout)
while True:
    #get frame and the result (in boolean) if the frame is readable
    _, frame = videoCheckin.read()
    #flip frame shown on the screen in vertical dimension, to look like a mirror
    frame = cv2.flip(frame, 1)
    #detectMutiscale(image, scaleFactor, minNeighbors)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5) #note
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,224,224),2)

        if type(frame) is np.ndarray:
            #get frame
            input_im = frame[y:y+h, x:x+w]
            input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
            #normalize pixel values from dividing by 255 to between 0 and 1
            input_im = input_im / 255.
            input_im = input_im.reshape(1,224,224,3)
            #get the chance of predicting the image
            percent = model.predict(input_im, 1, verbose = 0)
            #print(percent)
            chance = np.max(percent, axis=1)
            #print(percent)
            if (chance > 0.75):
                res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
                #print(str(np.max(percent, axis=1)) + " " + str(res))
                cv2.putText(frame, names[res[0]], (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
                output = { "email":""+email[res[0]]+"", "status":True, "room":"201"}
                url = 'http://localhost:3001/api/pl/attendances/create'
                x = requests.post(url,json=output)
                #print("{'email':'" +email[res[0]] + "'",",'status':true,","'room':'201'}")
            else:
                cv2.putText(frame, "Unrecognized", (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Video', frame)
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

videoCheckin.release()



while True:
    #get frame and the result (in boolean) if the frame is readable
    _, frameout = videoCheckout.read()
    #flip frame shown on the screen in vertical dimension, to look like a mirror
    frameout = cv2.flip(frameout, 1)
    #detectMutiscale(image, scaleFactor, minNeighbors)
    facesout = face_cascade.detectMultiScale(frameout, 1.3, 5) #note
    
    # Crop all faces found
    for (x,y,w,h) in facesout:
        cv2.rectangle(frameout,(x,y),(x+w,y+h),(0,224,224),2)

        if type(frameout) is np.ndarray:
            #get frame
            input_im = frameout[y:y+h, x:x+w]
            input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
            #normalize pixel values from dividing by 255 to between 0 and 1
            input_im = input_im / 255.
            input_im = input_im.reshape(1,224,224,3)
            #get the chance of predicting the image
            percent = model.predict(input_im, 1, verbose = 0)
            #print(percent)
            chance = np.max(percent, axis=1)
            #print(percent)
            if (chance > 0.75):
                res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
                #print(str(np.max(percent, axis=1)) + " " + str(res))
                cv2.putText(frameout, names[res[0]], (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
                output = { "email":""+email[res[0]]+"", "status":False, "room":"201"}
                url = 'http://localhost:3001/api/pl/attendances/create'
                x = requests.post(url,json=output)
                #print("{'email':'" +email[res[0]] + "'",",'status':true,","'room':'201'}")
            else:
                cv2.putText(frameout, "Unrecognized", (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        else:
            cv2.putText(frameout,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Video', frameout)
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
videoCheckout.release()


cv2.destroyAllWindows()
