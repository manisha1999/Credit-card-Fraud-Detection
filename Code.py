import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation
from keras.models import Sequential
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd


dataset = pd.read_csv(r"C:\Users\hp\Desktop\ml\creditcard.csv", header = 0)
model_features = dataset.iloc[:,1:30].columns

print(model_features)
dataset["Amount"] = (dataset["Amount"] - dataset["Amount"].min())/(dataset["Amount"].max() - dataset["Amount"].min())

dataset["Amount"].head(n=5)

dataset = dataset.sample(frac=1).reset_index(drop=True)
split = np.random.rand(len(dataset)) < 0.85
dataset_train = dataset[split]
dataset_test = dataset[~split]

x_train = dataset_train.as_matrix(columns = model_features)
y_train = dataset_train["Class"]
x_test = dataset_test.as_matrix(columns = model_features)
y_test = dataset_test["Class"]


y_train


print(dataset["Amount"].sum())
print(y_train.mean()*100)
print(y_test.mean()*100)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

y_test

y_train = np.asarray(y_train)             
y_test = np.asarray(y_test)             

class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train))

def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)



model = Sequential()
model.add(Dense(14, activation="relu", input_shape=(29,)))
model.add(Dense(7, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[precision, recall, "accuracy"])
                                               
model.summary()
model.fit(x=x_train, y=y_train, epochs = 10, class_weight = class_weights)
score = model.evaluate(x = x_test, y = y_test)
print ("Loss = " + str(score[0]))
print ("Precision metric = " + str(score[1]))
print ("Recall metric = " + str(score[2]))
print ("Accuracy metric = " + str(score[3]))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
y_pred = model.predict(x = x_test)
    
cnf_matrix = confusion_matrix(y_test, (y_pred>0.5).astype(int))
plot_confusion_matrix(cnf_matrix, classes = range(2))
plt.show()

y_test
test_frauds = pd.DataFrame({'Fraud':y_test[:,]})
test_frauds.head()

idx = test_frauds.index[test_frauds['Fraud'] == 1]
est_x_frauds = x_test[idx]
test_y_frauds = y_test[idx]

score_frauds = model.evaluate(x = test_x_frauds, y = test_y_frauds)
print ("Loss = " + str(score[0]))
print ("Precision metric = " + str(score[1]))
print ("Recall metric = " + str(score[2]))
print ("Accuracy metric = " + str(score[3]))


y_pred_frauds = model.predict(x = test_x_frauds)
cnf_matrix = confusion_matrix(test_y_frauds, (y_pred_frauds>0.5).astype(int))
plot_confusion_matrix(cnf_matrix, classes = range(2))
plt.show()

idx2 = test_frauds.index[test_frauds['Fraud'] == 0]
test_x_notfrauds = x_test[idx2]
test_y_notfrauds = y_test[idx2]
score_frauds = model.evaluate(x = test_x_notfrauds, y = test_y_notfrauds)
print ("Loss = " + str(score[0]))
print ("Precision metric = " + str(score[1]))
print ("Recall metric = " + str(score[2]))
print ("Accuracy metric = " + str(score[3]))

preds = model.predict(x = test_x_notfrauds)
cnf_matrix = confusion_matrix(test_y_notfrauds, (preds>0.5).astype(int))
plot_confusion_matrix(cnf_matrix, classes = range(2))
plt.show()





