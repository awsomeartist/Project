import cv2 as cv
import matplotlib.pyplot as plt 
from matplotlib.image import imread

#show image by using matplot
# img = imread(r"C:\Users\liang\Pictures\Camera Roll\kimono.jpg")
# plt.imshow(img)
# plt.show()

#show image by using opencv
# kimono = cv.imread(r"C:\Users\liang\Pictures\Camera Roll\kimono.jpg", cv.IMREAD_COLOR)
# kimono = cv.imread(r"C:\Users\liang\Pictures\Camera Roll\kimono.jpg", cv.IMREAD_GRAYSCALE)
# cv.imshow('kimono', kimono)
# cv.waitKey(0)


# play videos
capture = cv.VideoCapture(r'C:\Users\liang\Desktop\Desktops\Videos\masami\ma_section.mp4')
while True:
    isTrue,frame = capture.read()

    cv.imshow('Videos', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()