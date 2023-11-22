#import the necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#for necessary for image classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#this packages are necessary for every Classifier
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os





#input the images dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str,
help = "path to directory containing '3scenes' dataset")
ap.add_argument("-m", "--model", type = str,
help ="type of python machine"
"learning model to use")
args = vars(ap.parse_args())

def extract_color_stats(image):
    (R,G,B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
    np.std(G), np.std(B)]

    #return this features from color extrector
    return features


models = {
    "knn": KNeighborsClassifier(n_neighbors = 3),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver = "lbfgs", multi_class = "auto"),
    "svm":SVC(kernel = "linear")
    #KNeighborsClassifier(n_neighbors=5,
    #weights=’uniform’, algorithm=’auto’,
    #leaf_size=30, p=2, metric=’minkowski’,
    # metric_params=None, n_jobs=None, **kwargs)


}

print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []



for imagePath in imagePaths:

    image = Image.open(imagePath)
    features = extract_color_stats(image)
    #print("Features:", features)
    data.append(features)


    #labels List
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)


le = LabelEncoder()
labels = le.fit_transform(labels)
#print("Labels:", labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25)

print("[INFO] using '{}' model".format(args["model"]))

model = models[args["model"]]
model.fit(trainX, trainY)


print("[INFO] evaluating.....")
predictions = model.predict(testX)
print("Predictions:", predictions)
print(classification_report(testY, predictions,
target_names = le.classes_))
#print('features', features[0])
