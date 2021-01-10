import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score

# TASK 1
X_train = numpy.loadtxt('X_train.txt', delimiter=' ')
X_test = numpy.loadtxt('X_test.txt', delimiter=' ')
Y_train = numpy.loadtxt('y_train.txt')
Y_test = numpy.loadtxt('y_test.txt')

# TASK 2
classifiers = {'svm': svm.SVC(), 'knn': KNeighborsClassifier(), 'dt': DecisionTreeClassifier(), 'rf': RandomForestClassifier() }

predictions = {}
for key in classifiers:
    classifiers[key].fit(X_train, Y_train)
    predictions[key] = classifiers[key].predict(X_test)


# TASK 3
def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


for key in predictions:
    print('\n*** ' + key + ' ***')
    print('Accuracy score: {}'.format(accuracy_score(Y_test, predictions[key])))
    print('Recall score (micro): {}'.format(recall_score(Y_test, predictions[key], average='micro')))
    print('Recall score (macro): {}'.format(recall_score(Y_test, predictions[key], average='macro')))
    print('F1 score (micro): {}'.format(f1_score(Y_test, predictions[key], average='micro')))
    print('F1 score (macro): {}'.format(f1_score(Y_test, predictions[key], average='macro')))
    print("AUC score (micro): {}".format(multiclass_roc_auc_score(Y_test, predictions[key], average='micro')))
    print("AUC score (macro): {}".format(multiclass_roc_auc_score(Y_test, predictions[key], average='macro')))
    plt.figure(num='Confusion matrix: ' + key)
    seaborn.heatmap(confusion_matrix(Y_test, predictions[key]), annot=True, fmt='d')
    plt.show()


# TASK 4
class Score(object):
    def __init__(self, value):
        self.value = value
        self.avg = numpy.mean(value)
        self.std = numpy.std(value)


scores = {}
for key in classifiers:
    score = Score(cross_val_score(classifiers[key], X_train, Y_train, cv=5))
    scores[key] = score
    print('\n*** ' + key + ' ***')
    print('Cross validation score: {}'.format(score.value))
    print('Average: {}'.format(score.avg))
    print('Standard deviation: {}'.format(score.std))

print('\nThe best algorithms: ')
print('   maximum average: {}'.format(max(scores, key=lambda k: scores[k].avg)))
print('   minimum standard deviation: {}'.format(min(scores, key=lambda k: scores[k].std)))


# TASK 5

# Function sklearn.svm.SVC - parameters:
# C: float, default=1.0
# kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
# degree: int, default=3\
# gamma: {‘scale’, ‘auto’} or float, default=’scale’

c_values = [1000, 3000, 5000, 7000, 9000]
kernel_values = ['poly', 'rbf', 'sigmoid']
degree_values = [1, 2, 3, 4, 5]
gamma_values = ['scale', 'auto']
best_score = 0
best_score_info = ''
print('\nAccuracy scores:')
for c in c_values:
    for kernel in kernel_values:
        for degree in degree_values:
            for gamma in gamma_values:
                classifier = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
                classifier.fit(X_train, Y_train)
                prediction = classifier.predict(X_test)
                score = accuracy_score(Y_test, prediction)
                message = 'C: {}, kernel: {}, degree: {}, gamma: {}, score: {}'.format(c, kernel, degree, gamma, score)
                print(message)
                if score > best_score:
                    best_score = score
                    best_score_info = message
print('\n*** Best score -  ' + best_score_info + ' ***')


# Task 6
classifier = svm.SVC(C=1000, kernel='poly', degree=2, gamma='auto')
classifier.fit(X_train, Y_train)
predictions['svm_train'] = classifier.predict(X_train)
predictions['svm_test'] = classifier.predict(X_test)


def populate_subplot(ax, data):
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tomato', 'orange', 'lime', 'tan', 'navy']
    marker_list = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd', 'p', 'h', 'H']
    label_list = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT',
                  'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
    for i in range(1, 13):
        classes = data['class'] == i
        ax.scatter(data[0][classes], data[100][classes], c=color_list[i - 1], marker=marker_list[i - 1],
                   label=label_list[i - 1])
    ax.legend()


def print_real_expected_plot(x_file, y_file, predictions_key):
    fig, (real, expected) = plt.subplots(2, 1)

    # Train - real values
    data = pandas.read_csv(filepath_or_buffer=x_file, header=None, sep=' ')
    data['class'] = pandas.read_csv(filepath_or_buffer=y_file, header=None, sep=' ')
    populate_subplot(real, data)
    real.set_title(predictions_key + ' - real values')

    # Train - expected values
    data['class'] = predictions[predictions_key]
    populate_subplot(expected, data)
    expected.set_title(predictions_key + ' - expected values')

    plt.show()


print_real_expected_plot('X_train.txt', 'y_train.txt', 'svm_train')
print_real_expected_plot('X_test.txt', 'y_test.txt', 'svm_test')
