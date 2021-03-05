import cv2 as cv

img = cv.imread(r"C:\Users\liang\Pictures\Camera Roll\tfboys.jpg")
img_gray = cv.cvtColor(img, cv.IMREAD_GRAYSCALE)
face_casecade = cv.CascadeClassifier(r"C:\Users\liang\Anaconda3\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
faces = face_casecade.detectMultiScale(img_gray, scaleFactor = 1.03, minNeighbors = 5)

for x, y, w, h in faces:
    img = cv.rectangle(img, (x, y),(x+w,y+h), (0, 225, 0), 3)

# print(img)
cv.imshow('Image', img)

# resized = cv.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
# cv.imshow('Image', resized)
cv.waitKey(0)