import utils
from features_learn import Features

import sys
sys.path.insert(0, '/Users/XT/Documents/GitHub/neat-python-master/')
sys.path.insert(0, '/Users/XT/Documents/GitHub/neat-python-master/examples/xor')
import neat
import visualize
import numpy as np


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = len(pca_data)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(pca_data, data['train']['train']['labels']):
            output = net.activate(xi)
            output = np.argmax(np.asarray(output))
            genome.fitness -= np.abs(output - xo)


cch_file = '_cache/_all_data.pickle'
data = utils.get_init_data(cch_file)

pca_data, _ = Features.apply_pca(data['train']['train']['features'], 11)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
winner = p.run(eval_genomes, 500)
print('\nBest genome:\n{!s}'.format(winner))
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(pca_data, data['train']['train']['labels']):
    output = winner_net.activate(xi)
    output = np.argmax(np.asarray(output))
    print("expected output {!r}, got {!r}".format(xo, output))

node_names = {-1: 'A', -2: 'B', -3: 'C', 0: 'A XOR B'}
visualize.draw_net(config, winner, True,
                   node_names=node_names, fmt='jpg', show_disabled=False)
