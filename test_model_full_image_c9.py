import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

df = pd.read_csv(r"D:\HUSTer Senior\AI and Application\Project\coordition_2.csv")
path = r"C9_base_02.04.2021_12.55.40.jpg"
img = cv2.imread(path)
img_gray = cv2.imread(path, 0)

print(img_gray)
img1 = []
list_free = []
list_busy = []
for index in df['slot']:
    x = int(df['x'][index - 1])
    y = int(df['y'][index - 1])
    w = int(df['w'][index - 1] + x)
    h = int(df['h'][index - 1] + y)

    img1 = img_gray[y:h, x:w]
    image1 = cv2.resize(img1, dsize=(32, 32))
    image1 = image1 / 255.
    images = image1.reshape(1, 32, 32, 1)
    model = tf.keras.models.load_model(r"D:\HUSTer Senior\AI and Application\Project\model_tf.h5")
    predict = np.argmax(model.predict(images), axis=-1)
    # print(predict)
    if predict == 0:
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 3)
        cv2.putText(img, str(index), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        list_busy.append(index)
    else:
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 3)
        cv2.putText(img, str(index), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        list_free.append(index)

img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
cv2.imwrite('result.jpg', img)
print('vị trí trống:', list_free)
print('vị trí đã có ô tô:', list_busy)
cv2.imshow('img', img)
cv2.waitKey(0)