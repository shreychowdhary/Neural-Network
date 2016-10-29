import cv2
import net
import pickle
import numpy as np


NN = net.Net(784,100,10)

f1 = open('W1.pkl','rb')
f2 = open('W2.pkl','rb')
fx = open('X.pkl','rb')
fyt = open('Yt.pkl','rb')
fy = open('Y.pkl','rb')
NN.W1 = pickle.load(f1)
NN.W2 = pickle.load(f2)
X = pickle.load(fx)
yt = pickle.load(fyt)
y = pickle.load(fy)
f1.close()
f2.close()
fx.close()
fyt.close()
fy.close()

res = np.argmax(NN.forward(X),axis = 1)
print float(np.sum(res==yt))/yt.shape[0]

im = cv2.imread("digit.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)


ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    leng = int((rect[3]-rect[1])* 1.6)
    pt1 = int((rect[1] + rect[3])// 2 - (leng // 2))
    pt2 = int((rect[0] + rect[2])// 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    cv2.imwrite("box.jpg",im)
    cv2.imwrite("processed.jpg",im_th)
    print rect[0]
    print rect[1]
    print rect[2]
    print rect[3]
    print leng
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    roi = cv2.dilate(roi, (3, 3))

    X = np.mat(roi).reshape(1,784)
    num = np.argmax(NN.forward(X),axis=1)
    print num
