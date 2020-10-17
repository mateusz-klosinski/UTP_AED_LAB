from numpy import genfromtxt
from typing import Iterable

from sklearn import datasets, model_selection
import csv
import os
import pandas as pd

def zad1_loading_set () :
    all_data = datasets.load_wine()
    print(all_data.keys())
    print(all_data.get('DESCR'))
    print(all_data.get('data'))
    print(all_data)


def zad2_split_dataset() :
    all_data = datasets.load_wine()
    data = all_data.get('data')
    targets = all_data.get('target')
    split_dataset = model_selection.train_test_split(data, targets, test_size=0.4)
    dirname = 'out'
    filenames = ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv']

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    for index in range(4):
        filename = filenames[index]
        path = dirname + '/' + filename
        data = split_dataset[index]
        file = open(path, 'w')
        writer = csv.writer(file)
        writer = csv.writer(file)
        for element in data:
            if isinstance(element, Iterable):
                writer.writerow(element)
            else:
                writer.writerow([element])
        file.close()

    return split_dataset


def zad3_statistics():
    dirname = 'out'
    filenames = ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv']
    for filename in filenames:
        print("SUMMARY FOR: " + filename)
        array = genfromtxt(dirname + '/' + filename, delimiter=',')
        data = pd.DataFrame(array.flatten())
        print("MAX:")
        print(data.max())
        print("MIN:")
        print(data.min())
        print("COUNT")
        print(data.count())
        print("UNIQUE COUNT")
        print(data.value_counts().count())
        print("MEAN VALUE")
        print(data.dropna().mean())
        print("NULL COUNT")
        print(data.count() - data.dropna().count())
        print("MOST OFTEN OCCURS")
        print(data.value_counts().sort_values(ascending=False).head(1).keys())


zad1_loading_set()
print(zad2_split_dataset())
zad3_statistics()