import neat
from optimize import TensorFeedForward, TensorGenome
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts')
import n_utils as ut

import visualize


def eval_genome(genomes, config):
    a = 2
    for genome_id, genome in genomes:
        net = TensorFeedForward(genome, config)
        try:
            acc = net.optimize(ftrain.T, ltrain.T, 2)
        except TypeError or ValueError:
            print('ha-ha exception, Looser!! Dump the instance')
            ut.dump_to_fld('_cache/failed_genome.pickle', genome)
        net.update_genome(genome)
        genome.fitness = acc


def plot_data(class_1, class_2, class_3):
    plt.scatter(class_1[:, 0], class_1[:, 1], c='r')
    plt.scatter(class_2[:, 0], class_2[:, 1], c='g')
    plt.scatter(class_3[:, 0], class_3[:, 1], c='b')
    plt.show()

    # plot_data(ftrain[ltrain == 0], ftrain[ltrain == 1], ftrain[ltrain == 2])


def debug():
    raw_features, raw_labels = make_blobs(200, 2, 3, random_state=45)
    ftrain, ftest, ltrain, ltest = train_test_split(
        raw_features, raw_labels, test_size=0.2, random_state=33)
    genome = ut.read_from_fld('_cache/failed_genome.pickle')
    net = TensorFeedForward(genome, config)
    print(net.optimize(ftrain.T, ltrain.T))


if __name__ == '__main__':
    cch_file = '/Users/XT/Documents/PhD/Granada/neuroimaging/Python_scripts/_cache/_all_data.pickle'
    data = ut.get_init_data(cch_file)

    ftrain = data['train']['train']['features']
    ltrain = data['train']['train']['labels']

    config_file = 'evolution_config'
    config = neat.Config(TensorGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    best = population.run(eval_genome)

    visualize.draw_net(config, best, True, fmt='jpg', filename='best_net')

    # debug()
