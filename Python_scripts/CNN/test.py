import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
# Analyse the data


import re
num_epochs = 158 + 1
results = open('Inception_ResNet_3D.running.txt')
current_batch = 1

epoch_results = dict()


for line in results:
    if current_batch > 24:
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


for key in epoch_results.keys():
    plt.scatter(epochs, epoch_results[key], marker='o', c='r', linewidth=1.0)
    x = np.array(epochs).reshape(-1, 1)
    y = np.array(epoch_results[key]).reshape(-1, 1)
    long_run = np.array(range(1, 3000)).reshape(-1, 1)

    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    plt.plot(long_run, reg.predict(long_run))
    plt.title('Batch: {0}'.format(key))
    plt.xlabel('epochs')
    plt.ylabel('batch_accuracy')
    plt.savefig(os.path.join('_batch_regression', '{0}_{1}_epochs'.format(key, num_epochs)))
    plt.close()
