# Points to remember if game is changed:
# configure 'config-feedforward' file
# don't forget to change the output layer node in the above file depending upon the number of agent actions.

import retro
import neat
import cv2
import numpy as np


env = retro.make('SuperMarioBros-Nes')

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        state = env.reset()
        action = env.action_space.sample()
        width, height, channels = env.observation_space.shape
        width = int(width/8)    # 28
        height = int(height/8)  # 30

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        current_fitness     = 0
        frame               = 0
        counter             = 0
        xpos                = 0
        xpos_max            = 0
        done                = False

        while not done:
            env.render()
            frame += 1

            state = cv2.resize(state, (width, height))
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = np.reshape(state, (height, width))

            state = state.flatten()

            nn_output = net.activate(state)
            # print(nn_output)

            state, reward, done, info = env.step(nn_output)

            xpos = info['xscrollLo']

            if xpos > xpos_max:
                current_fitness += 1
                xpos_max = xpos

            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, current_fitness)

            genome.fitness = current_fitness


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')


pop = neat.Population(config)

winner = pop.run(eval_genomes)
