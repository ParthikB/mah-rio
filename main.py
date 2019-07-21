# Points to remember if game is changed:
# -- Configure 'config-feedforward' file accordingly.
# -- Don't forget to change the output layer node in the above file depending upon the number of agent actions.

import retro
import neat
import numpy as np
from handlers import *
import os
import pickle

env = retro.make('SuperMarioBros-Nes', 'Level1-1.state')

if os.path.isfile("best_fitness_log.npy"):
    print("Existing fitness log found. Loading...!")
    best_fitness_log = list(np.load("best_fitness_log.npy"))
else:
    print("No existing fitness log found. Creating new...!")
    best_fitness_log = []


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')


if __name__ == '__main__':  # If the module is run directly and not imported, then the following will run.

    pop = population_loader()
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Now we will check if the current Version's Checkpoint folder is available. If not, directory will be created.
    # Thus, checkpoints will be saved in that directory.
    if not os.path.isdir(f"C:/Users/ParthikB/PycharmProjects/retro/checkpoint/v{VERSION}"):
        os.mkdir(f"C:/Users/ParthikB/PycharmProjects/retro/checkpoint/v{VERSION}")

    os.chdir(f"C:/Users/ParthikB/PycharmProjects/retro/checkpoint/v{VERSION}")
    pop.add_reporter(neat.Checkpointer(10))

    winner = pop.run(eval_genomes)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
