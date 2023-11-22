#import the necessary package

#python cnn_basic.py

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os



print("loding the dataset.....")
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, help = "path"
" to directory containing dataset")
args = vars(ap.parse_args())



print("loding images.......")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []


for imagePath in imagePaths:

    image = Image.open(imagePath)
    image = np.array(image.resize((32, 32)))/255.0
    data.append(image)


    #label list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

#encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


(train_X, test_x, train_Y, test_y) = train_test_split(np.array(data),
np.array(labels), test_size = 0.25)

model = Sequential()
model.add(Conv2D(8, (3,3), padding = "same", input_shape = (32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides =(2,2)))
model.add(Conv2D(16, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(32, (3,3), padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation("softmax"))




print("training network.....")

opt = Adam(lr = 1e-3, decay = 1e-3/50)
model.compile(loss ="categorical_crossentropy", optimizer = opt,
metrics = ["accuracy"])
H = model.fit(train_X, train_Y, validation_data = (test_x, test_y),
epochs = 50, batch_size = 32)

print("[INFO] evaluting network....")
#print("[INFO] evaluating network...")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
