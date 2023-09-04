import cv2 as cv

capture = cv.VideoCapture(0)
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml

haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    # Capture frame-by-frame
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
 

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    print(f'Number of faces found = {len(faces_rect)}')

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(0) & 0xFF == ord('s'):
        break

cv.waitKey(0)
capture.release()
cv.destroyAllWindows()