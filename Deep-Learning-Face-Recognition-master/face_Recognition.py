# Face Recognition

# Importing the libraries
import cv2, os
import numpy as np
from keras.models import load_model
from collections import Counter

model = load_model(r'model\24-03.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

names = ['Tri', 'Thuan', 'Linh', 'Tinh', 'Nguyen']

def returnStudentID(arr, studentID, alreadyAttendance):
    arr.append(studentID)
    if(Counter(arr).most_common()[0][1]>7):
        presentStudentID = Counter(arr).most_common()[0][0]
        arr.clear()
        if not presentStudentID in alreadyAttendance:
            alreadyAttendance.append(presentStudentID)
        return presentStudentID,arr,alreadyAttendance

studentList = []
alreadyAttendance = []
# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    #get frame and the result (in boolean) if the frame is readable
    _, frame = video_capture.read()
    #flip frame shown on the screen in vertical dimension, to look like a mirror
    frame = cv2.flip(frame, 1)
    #detectMutiscale(image, scaleFactor, minNeighbors)
    faces = face_cascade.detectMultiScale(
        frame, 
        scaleFactor= 1.3, 
        minNeighbors= 5, 
        minSize= (150, 150)) #note
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,224,224),2)

        if type(frame) is np.ndarray:
            #get frame
            input_im = frame[y:y+h, x:x+w]
            input_im = cv2.resize(input_im, (224, 224))
            #normalize pixel values from dividing by 255 to between 0 and 1
            input_im = input_im / 255.
            input_im = input_im.reshape(1,224,224,3)
            #get the chance of predicting the image
            percent = model.predict(input_im, 1, verbose = 0)
            np.set_printoptions(formatter={'float': '{:.3%}'.format})
            #print(percent)
            #show recognition if the chance is high enough
            chance = np.max(percent, axis=1)
            if (chance > 0.75):
                res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
                print(str(np.max(percent, axis=1)) + " " + names[res[0]])
                #print(returnStudentID(studentList, res[0], alreadyAttendance))
                #print(alreadyAttendance)
                cv2.putText(frame, names[res[0]] + str(np.max(percent, axis=1)), (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            else:
                cv2.putText(frame, "Unrecognized", (x+5,y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv2.imshow('Video', frame)
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()