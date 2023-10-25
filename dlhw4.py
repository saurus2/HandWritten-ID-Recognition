'''
tensorflow : 2.7
keras : 2.7
'''

import sys
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from subprocess import call
import keras
import tensorflow
print(keras.__version__)
print(tensorflow.__version__)

print("input number : webcam [0], video[1]")
option = int(input())
cp = 0
out = 0

if option == 0:
    cp = cv2.VideoCapture(0)
    w = round(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cp.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('webcam_output.mp4', fourcc, fps, (w, h))
elif option == 1:
    cp = cv2.VideoCapture('./video1.mp4')
    w = round(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cp.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('record_output.mp4', fourcc, fps, (w, h))

# cp = cv2.VideoCapture(0)
cp.set(3, 5*128)
cp.set(4, 5*128)
SIZE = 64
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    first_dim = 0
    second_dim = 1
else:
    input_shape = (img_rows, img_cols, 1)
    first_dim = 0
    second_dim = 3

def extract_digit(frame, rect, pad = 10):
    x, y, w, h = rect
    cropped_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad]

    cropped_digit = cropped_digit/255.0
    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        # cv2.imshow("cropped img", cropped_digit)
        cropped_digit = cv2.resize(cropped_digit, (SIZE, SIZE))
        # cv2.imshow("cropped img", frame[y-pad:y+h+pad, x-pad:x+w+pad])
    else:
        return
    return cropped_digit

print("loading model")
model = load_model("jihyeok2.h5")

for i in range(1000):
    ret, frame = cp.read(0)

    if not ret:
        cp.release()
        out.release()
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = SIZE)
    final_img = gray_img
    image_shown = frame

    contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]


    frame2 = frame.copy()

    w2 = round(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = round(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

    square_bg = np.ones((h2, w2), np.uint8) + 255
    square_bg.fill(255)

    for rect in rects:
        x, y, w, h = rect


        # cv2.imshow(';', frame2)


        try:
            # adding cropped images
            leng = int(h)

            cropped_img = frame2[y:y + h, x:x + w]

            square_bg2 = np.zeros((leng, leng, 3), np.uint8)

            w_begin = (leng // 2) - (w // 2)
            h_begin = (leng // 2) - (h // 2)
            w_end = 0
            if w % 2 == 0:
                w_end = (leng // 2) + (w // 2)
            else:
                w_end = (leng // 2) + (w // 2) + 1
            h_end = 0
            if h % 2 == 0:
                h_end = (leng // 2) + (h // 2)
            else:
                h_end = (leng // 2) + (h // 2) + 1

            square_bg2[h_begin:h_end, w_begin:w_end] = cropped_img.copy()

            image_shown[y + 200:y + h + 200, x:x + h] = square_bg2.copy()

            cropped_img = frame2[y:y + h, x:x + w]
            # image_shown[y + 200:y + h + 200, x:x + w] = cropped_img.copy()
            # image_shown[y + 200 :y + h + 200 , x:x + w] = cropped_img.copy()
            # cv2.imshow('l',cropped_img.copy())

        except:
            pass

        if i >= 0:
            mnist_frame = extract_digit(frame, rect, pad = 15)

            if mnist_frame is not None:
                mnist_frame = np.expand_dims(mnist_frame, first_dim)
                mnist_frame = np.expand_dims(mnist_frame, second_dim)

                result_arr = model.predict(mnist_frame)
                class_prediction = np.argmax(result_arr, axis=1)
                prediction = np.around(np.max(model.predict(mnist_frame, verbose = False)), 2)

                cv2.rectangle(image_shown, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=4)
                label = str(int(class_prediction))
                cv2.putText(image_shown, label, (rect[0], rect[1] - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)


                # adding cropped images
                leng = int(h)

                cropped_img = gray_img[y:y + h, x:x + w]

                square_bg2 = np.zeros((leng, leng), np.uint8)

                w_begin = (leng // 2) - (w // 2)
                h_begin = (leng // 2) - (h // 2)
                w_end = 0
                if w % 2 == 0:
                    w_end = (leng // 2) + (w // 2)
                else:
                    w_end = (leng // 2) + (w // 2) + 1
                h_end = 0
                if h % 2 == 0:
                    h_end = (leng // 2) + (h // 2)
                else:
                    h_end = (leng // 2) + (h // 2) + 1


                # cv2.imshow('sqbg', square_bg2)

                # square_bg[y: y + h, x: x + w] = cropped_img.copy()
                try:
                    square_bg2[h_begin:h_end, w_begin:w_end] = cropped_img.copy()
                    ty = y
                    tx = x
                    square_bg[ty:ty+h, tx:tx+h] = square_bg2.copy()
                except:
                    pass
                # cv2.imshow("cropped img", final_img[y - 15:y + h + 15, x - 15:x + w + 15])


    out.write(image_shown)

    cv2.imshow('frame', image_shown)
    cv2.imshow('preprocessed square image for each digit', square_bg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cp.release()
        out.release()
        break