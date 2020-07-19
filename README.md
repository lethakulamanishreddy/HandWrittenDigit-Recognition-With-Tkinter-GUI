# HandWrittenDigit-Recognition-With-Tkinter-GUI

This repository contains Hand written Digit Recognition Prject with a GUI made using Tkinter.


## DEMO

https://youtu.be/pSpwV16Z4_Y

https://www.instagram.com/p/CCP5XQ3j7FS/?utm_source=ig_web_copy_link



## Requirements
--->joblib\
--->sklearn\
--->numpy\
--->skimage\
--->collections\
--->opencv\
--->PIL\
--->Tkinter

This consits of 2 python files\
1.classifiergenerator.py\
2.recognizer.py

## classifergenerator.py
This file contains python code for generating a classifier using sklearn's Linear SVC model.\
We will use mnist dataset to tarin the model.
## Step by step flow:
1.loading mnist dataset.\
2.Extracting hog features as numpy array.\
3.training the model.\
4.Saving the model. --->This will generate a classifier model (pickle file) classifier.pkl

## recognizer.py
This file contains python code which allows user to draw the digits using mouse pointer and recognize the digits.
## Step by step flow:
1.Loading the classifier model\
2.Creating GUI(3 buttons:save,predict,clear and canvas to draw)\
3.Saving the sketch as image and loading saved image\
4.Convertig imge into grayscale\
5.Applying gaussianblur\
6.Thresholding image\
7.Finding contours\
8.Drawing rectangle ariund the contours\
9.Recognizing the digit\
10.Putting text(number) on top of the rectangle

## Running

Open commandprompt and change direcrtory to project nad run'python recognizer.py'.\
this will open a GUI which consists of 3 buttons.\
1.save\
2.predict\
3.clear

Steps for realtime recognition\
step1 - draw a digit\
step2 - click save\
step3 - click predict--->this will show a popup image with the recognised value written on the sketch.\
step4 - clear--->this will clear the canvas

step1-./
.\
.\
.\
.

