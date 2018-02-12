import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import re
from sklearn import linear_model
from sklearn import preprocessing
# Analyse the data
OUT_PATH = 'CNN/_batch_regression'
FILEPATH = '/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts/CNN/simple_network.running.txt'

num_epochs = 211 + 1
results = open(FILEPATH)
current_batch = 1

epoch_results = dict()


for line in results:
    if current_batch > 15:
        current_batch = 1
    acc = re.findall('- (acc|val_acc): .*?([0-9.-]+)', line)
    if acc:
        if not (current_batch in epoch_results):
            epoch_results[current_batch] = list()
        epoch_results[current_batch].append(float(acc[0][1]))
        if len(acc) == 2:
            if not ('validation' in epoch_results):
                epoch_results['validation'] = list()
            epoch_results['validation'].append(float(acc[1][1]))
        current_batch += 1
results.close()

epochs = list(range(1, num_epochs))

le = preprocessing.LabelEncoder()
for key in epoch_results.keys():
    y_encoded = le.fit_transform(np.array(epoch_results[key]))
    plt.scatter(epochs, epoch_results[key], marker='o', c='r', linewidth=1.0)
    x = np.array(epochs, dtype='int64').reshape(-1, 1)

    long_run = np.array(range(1, 500)).reshape(-1, 1)

    reg = linear_model.LogisticRegression(C=0.8, solver='lbfgs', intercept_scaling=0.7, verbose=1)
    reg.fit(x, y_encoded)
    prediction = reg.predict(long_run)
    plt.plot(long_run, le.inverse_transform(prediction))
    plt.title('Batch: {0}'.format(key))
    plt.xlabel('epochs')
    plt.ylabel('batch_accuracy')
    plt.savefig(os.path.join(OUT_PATH, '{0}_{1}_epochs'.format(key, num_epochs)))
    plt.close()
