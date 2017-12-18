from keras.models import Sequential
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import os
from PIL import Image
import numpy as np
import shutil


def load_dataset(filedir):
    image_data_list = []
    label = []
    train_image_list = os.listdir(filedir + '/test_resize')
    for img in train_image_list:
        url = os.path.join(filedir + '/test_resize/' + img)
        image = load_img(url, target_size=(128, 128))
        image_data_list.append(img_to_array(image))
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data


model = load_model('my_model.h5')
x_train = load_dataset('./Desktop/PRBP')
answer = []
answer = model.predict_classes(x_train)
i = 0
shutil.rmtree('./keras_result')
os.mkdir('./keras_result')
for ig in os.listdir('./Desktop/PRBP/test_resize'):
    id_tag = ig.find(".")
    name = ig[0:id_tag]
    im = Image.open("./Desktop/PRBP/test_resize/" + ig)
    out = im.resize((128, 128))
    out.save("./keras_result/" + str(answer[i]) + name + ".jpg")
    i += 1
