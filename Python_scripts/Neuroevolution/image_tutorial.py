import __future__
import sys
sys.path.append('/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts')
import n_utils as ut

from PIL import Image
import numpy as np
from math import gcd

from scipy.io.wavfile import write


def load_AD():
    cch_file = '/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts/_cache/_all_data.pickle'
    data = ut.get_init_data(cch_file)

    ftrain = data['train']['train']['features']
    ltrain = data['train']['train']['labels']
    return ftrain, ltrain


if __name__ == '__main__':
    fld = '_sounds/'
    z, l = load_AD()
    # for i, (features, label) in enumerate(zip(z, l)):
    #     scaled = np.int16(features[43:63] / np.max(np.abs(features)) * 32767)
    #     filename = fld + str(i) + '.' + ut.get_AD_stage_name(int(label))
    #     write(filename, 116, scaled)

    for i, (features, label) in enumerate(zip(z, l)):
        filename = '_images/' + str(i) + '_' + \
            ut.get_AD_stage_name(int(label))
        features = features.reshape((1, -1))

        img = np.hstack((features, np.zeros((1, 5)))).reshape((11, 11))
        g = Image.fromarray(img).convert('RGB')
        g.save(filename + '.jpg')
