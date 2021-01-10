import pandas
import numpy
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dirname = 'LAB_04_results'

if not os.path.exists(dirname):
    os.mkdir(dirname)


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


# TASK 1
X_train = numpy.loadtxt('X_train.txt', delimiter=' ')
X_test = numpy.loadtxt('X_test.txt', delimiter=' ')
Y_train = numpy.loadtxt('y_train.txt')
Y_test = numpy.loadtxt('y_test.txt')

# TASK 2
X = numpy.concatenate((X_train, X_test))
Y = numpy.concatenate((Y_train, Y_test))

pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)

original_train_dataset = train_test_split(X, Y, test_size=0.3)
original_train_x = original_train_dataset[0]
original_train_y = original_train_dataset[2]

pca_train_dataset = train_test_split(pca_X, Y, test_size=0.3)
pca_train_x = pca_train_dataset[0]
pca_test_x = pca_train_dataset[1]
pca_train_y = pca_train_dataset[2]
pca_test_y = pca_train_dataset[3]

pca_score = cross_validate(svm.SVC(), pca_train_x, pca_train_y, cv=5)
original_score = cross_validate(svm.SVC(), original_train_x, original_train_y, cv=5)

path = dirname + '/' + 'dim_reduction.csv'
file = open(path, 'w')
writer = csv.writer(file)

writer.writerow([' ', 'Before dim reduction', 'After dim reduction'])
writer.writerow(['Accuracy', original_score.get('test_score').mean(), pca_score.get('test_score').mean()])
writer.writerow(['Train time', original_score.get('fit_time').mean(), pca_score.get('fit_time').mean()])
writer.writerow(['Test time', original_score.get('score_time').mean(), pca_score.get('score_time').mean()])

file.close()


# TASK 3
classifiers = [
    ('svm', svm.SVC(gamma=0.1, kernel='rbf', probability=True)),
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('dt', DecisionTreeClassifier(max_depth=4)),
    ('rf', RandomForestClassifier())
]
weights = [0.3, 0.3, 0.2, 0.2]

voting_classifier = VotingClassifier(classifiers, weights=weights, voting='soft')
voting_classifier.fit(pca_train_x, pca_train_y)
voting_prediction = voting_classifier.predict(pca_test_x)
voting_score = voting_classifier.score(pca_test_x, pca_test_y)

path = dirname + '/' + 'ensembled_learning.csv'
file = open(path, 'w')
writer = csv.writer(file)

writer.writerow(['VotingClassifier scores'])
writer.writerow(['Accuracy score', accuracy_score(pca_test_y, voting_prediction)])
writer.writerow(['Recall score', recall_score(pca_test_y, voting_prediction, average='macro')])
writer.writerow(['F1 score', f1_score(pca_test_y, voting_prediction, average='macro')])
writer.writerow(["AUC score", multiclass_roc_auc_score(pca_test_y, voting_prediction, average='macro')])

file.close()

# TASK 4
h = 0.2
x_min, x_max = pca_X[:, 0].min() - .5, pca_X[:, 0].max() + .5
y_min, y_max = pca_X[:, 1].min() - .5, pca_X[:, 1].max() + .5
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))


# Plot dataset
figure = plt.figure(figsize=(27, 9))
cm = plt.cm.RdBu
cm_color_map = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, 2, 1)
ax.set_title("Dataset")
ax.scatter(pca_train_x[:, 0], pca_train_x[:, 1], c=pca_train_y, cmap=cm_color_map, edgecolors='k')
ax.scatter(pca_test_x[:, 0], pca_test_x[:, 1], c=pca_test_y, cmap=cm_color_map, alpha=0.6, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

# Plot voting classification
ax = plt.subplot(1, 2, 2)
ax.set_title("Voting classification")
Z = voting_classifier.predict(numpy.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
ax.scatter(pca_train_x[:, 0], pca_train_x[:, 1], c=pca_train_y, cmap=cm_color_map, edgecolors='k')
ax.scatter(pca_test_x[:, 0], pca_test_x[:, 1], c=pca_test_y, cmap=cm_color_map, edgecolors='k', alpha=0.6)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())


plt.tight_layout()
plt.show()
