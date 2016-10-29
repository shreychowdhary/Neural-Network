import cv2
import net
import pickle
import numpy as np

NN = net.Net(784,100,10)
f1 = open('W1.pkl','rb')
f2 = open('W2.pkl','rb')
NN.W1 = pickle.load(f1)
NN.W2 = pickle.load(f2)
f1.close()
f2.close()

im = cv2.imread("digit.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)


ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

cv2.imwrite("digitProcessed.jpg",im)
