import cv2
cam = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    msg, img = cam.read()
    faces = facedetect.detectMultiScale(img, 1.1, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    cv2.imshow("cam", img)
    key = cv2.waitKey(10)
    if key == 32:  # Press spacebar to exit the loop
        break

cam.release()
cv2.destroyAllWindows()
''' 
This is a program written in the process of learning of data visualization.
This program helps to identify the face of the user and draws a square around it.
This is continous loop as it runs as until user stops it.
The user has to stop the program The user is allowed to click Space button.
'''