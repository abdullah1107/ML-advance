#import the necessary package
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris



print("loding data.....")

dataset = load_iris()

#custom dataset import

#Now split the data training and testing
#(train_X, test_x, train_Y, test_y) = train_test_split(total_inputDataset,
#totaldatasetOutput, test_size = 0.25)
(train_X, test_x, train_Y, test_y) = train_test_split(dataset.data,
dataset.target, test_size = 0.25)

#encode the label as 1-hot vectors
lb_object = LabelBinarizer()
train_Y = lb_object.fit_transform(train_Y)
test_y = lb_object.transform(test_y)


#define the 4-3-3-3 model architecture using keras

model = Sequential()
model.add(Dense(3, input_shape = (4,), activation = "sigmoid"))
model.add(Dense(3, activation = "sigmoid"))
model.add(Dense(3, activation = 'softmax'))

#train the model using SGD

print("Training network.....")
opt = SGD(lr = 0.1, momentum = 0.9, decay = 0.1/100)
#decay = learning_rate/ total_epoc
#momentam = range
model.compile(loss = "categorical_crossentropy", optimizer = opt,
metrics = ['accuracy'])

H = model.fit(train_X, train_Y, validation_data=(test_x, test_y),
	epochs=100, batch_size=16)


print("[INFO] evaluating network...")
predictions = model.predict(test_x, batch_size=16)
print(classification_report(test_y.argmax(axis=1),
predictions.argmax(axis=1), target_names=dataset.target_names))
