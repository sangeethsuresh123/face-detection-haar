import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
vid = cv.VideoCapture(0)

while (True):
    ret, frame = vid.read()
    # print(ret)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=35)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()

# print(f'Number of faces found = {len(faces_rect)}')
# print(faces_rect)
cv.waitKey(0)
