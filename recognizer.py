'''
            HAND WRITTEN DIGIT RECOGNITION
        Code contributed by LETHAKULA MANISH REDDY
'''


# Import the modules
import cv2
import joblib
from skimage.feature import hog
import numpy as np
from pil import ImageTk, Image, ImageDraw
import pil
from tkinter import *

width = 200
height = 200
center = height//2
white = (255, 255, 255)
green = (0,128,0)

def save():
    filename = "image.png"    
    image1.save(filename)

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

def clear():
    cv.delete('all')
    draw.rectangle([0,200, 200, 0],fill="white")
    
root = Tk()
root.title('HAND WRITTEN DIGIT RECOGNITION')
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()
image1 = pil.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
button=Button(text="save",command=save)
button.pack()
# Load the classifier
clf = joblib.load("classifier.pkl")
def predict():
    # Read the input image 
    im = cv2.imread("image.png")

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (51, 204, 51), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 51, 0), 3)

    cv2.imshow("output", im)
    cv2.waitKey()

button2=Button(text="predict",command=predict)
button2.pack()
button=Button(text="clear",command=clear)
button.pack()
root.mainloop()
