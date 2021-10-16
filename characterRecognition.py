# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:45:31 2021

@author: sherin
"""
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#iris = datasets.load_iris()
digits = datasets.load_digits()
data=digits.data
labels=digits.target
images=digits.images
img=data[0].reshape((8,8))
# print(data[0].size)
# print(img)
# print(type(images))# numpy.ndarray
# print(images.shape)#(1797, 8, 8)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

clf = svm.SVC(gamma=0.001, C=100.)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.4, shuffle=False)

clf.fit(X_train,y_train)

predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test[10:], predicted[10:]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

# Report
print(metrics.classification_report(y_test, predicted))

#Confusion Matrix

cm = confusion_matrix(y_test, predicted, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


