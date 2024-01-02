import cv2, os

#Load haarcascades classifier
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def createFolderImage():
    print("Enter folder name:")
    image_folder = input()
    try:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
    except OSError:
        print("Error: Creating directory of data")
    return image_folder

def showFaceCropped(face_frame, count):
    window_resize = cv2.resize(face_frame, (480, 480))
    cv2.putText(window_resize, str(count), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Face Cropped", window_resize)


image_folder = createFolderImage()

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    #Detect face
    faces = face_classifier.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        face_resize = cv2.resize(face_crop, (224, 224))
        file_name = image_folder + "\\" + str(count) + ".jpg"
        cv2.imwrite(file_name, face_resize)
        print("Saving " + file_name)

        #Put count on images and display count
        showFaceCropped(face_crop, count)
        count += 1

    #27 is the Esc Key
    if cv2.waitKey(1) == 27 or count == 200:
        break

video_capture.release()
cv2.destroyAllWindows()
print("Collect complete")