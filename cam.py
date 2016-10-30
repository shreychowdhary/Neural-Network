import cv2
import net
import pickle
import numpy as np

camera_port = 0
ramp_frames = 30

camera = cv2.VideoCapture(camera_port)

def get_image():
    retval, im = camera.read()
    return im

for i in xrange(ramp_frames):
    temp = get_image()

NN = net.Net(784,100,10)

f1 = open('W1.pkl','rb')
f2 = open('W2.pkl','rb')

NN.W1 = pickle.load(f1)
NN.W2 = pickle.load(f2)

f1.close()
f2.close()

while True:
    im = get_image()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("blackandwhite.jpg",im_th)

    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    print len(rects)
    if len(rects) == 0:
        continue
    for rect in rects:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        cv2.imwrite("rect.jpg",im)
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        if(pt1 < 0 or pt2 < 0):
            print "error getting digit"
            continue
        # Resize the image

        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        X = np.mat(roi).reshape(1,784)
        print X
        num = np.argmax(NN.forward(X),axis=1)
        print num
    time.sleep(.5)
