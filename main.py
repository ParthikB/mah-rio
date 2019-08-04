import retro
import neat
import cv2
import os
import time
import numpy as np
import pickle
from statistics import mean
from handlers import *

env = retro.make("SuperMarioBros-Nes")
info(env)


def eval_genomes(genomes, config):
    times_level_finished = 0
    member = 0
    generation_fitness_log = []

    for genome_id, genome in genomes:
        state = env.reset()

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        temp_state, _, _, _ = env.step(env.action_space.sample())
        WIDTH  = int(temp_state.shape[0]/8)
        HEIGHT = int(temp_state.shape[1]/8)

        done = False
        current_fitness = 0
        generation_max_fitness = 0
        counter = 0
        member += 1

        while not done:
            env.render()

            state = cv2.resize(state, (WIDTH, HEIGHT))
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = np.reshape(state, (WIDTH, HEIGHT))

            state_arr = np.ndarray.flatten(state)
            action = net.activate(state_arr)

            state, reward, done, info = env.step(action)

            # print(info)
            level_end = info['levelHi']

            current_fitness += reward

            if level_end >= 1:
                current_fitness += 100000
                done = True
                times_level_finished += 1
                print("Vohoooo.....Level Completed...!!")

            if current_fitness > generation_max_fitness:
                generation_max_fitness = current_fitness
                counter = 0
            else:
                counter += 1

            if counter >= 250:
                done = True

            if times_level_finished >= 10:
                current_fitness += 1000000

            genome.fitness = current_fitness


        generation_fitness_log.append(generation_max_fitness)
        # print("Max fitness log :", generation_fitness_log)
        generation_mean_fitness = round(mean(generation_fitness_log), 2)
        generation_max_fitness  = round(max(generation_fitness_log), 2)
        if generation_max_fitness != 0:
            times_reached_max_fitness = generation_fitness_log.count(generation_max_fitness)
        else:
            times_reached_max_fitness = 0

        print(f"""----Member {member}---- Fitness Log updated || Current Fitness Score : {current_fitness}      || Times Max Fitness Reached : {times_reached_max_fitness}
                                      || Gen Max Fitness Score : {generation_max_fitness}      || Gen Mean Fitness Score    : {generation_mean_fitness}""" )


    generation_info.append([generation_mean_fitness, generation_max_fitness, times_reached_max_fitness])
    np.save(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}/fitness_log.npy", generation_info)
    print("Generation Info updated.")


def population_loader():
    choice = (input("Do you want to load a population checkpoint (y/n) : ")).lower()
    if choice == 'n':
        print("Creating new population...//")
        time.sleep(2)
        pop = neat.Population(config)  # Use this to start afresh.
    elif choice == 'y':
        version    = input(" -------Enter the model version     : ")
        checkpoint = int(input(" -------Enter the checkpoint number : "))
        try:
            os.chdir(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{version}")
            print(f"Loading Checkpoint {checkpoint} from model v{version} ...//")
            pop = neat.Checkpointer.restore_checkpoint(f"neat-checkpoint-{checkpoint}")
            print(f"Population at Checkpoint {checkpoint} loaded...//")
        except FileNotFoundError:
            print("Invalid Version/Checkpoint. Try again.")
            population_loader()
    else:
        print("Invalid input. Please answer in y/n.")
        population_loader()

    return pop

print(f"Version {VERSION} loaded.")
print()
if os.path.isfile(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}/fitness_log.npy"):
    print("Existing fitness log found. Loading...//")
    generation_info = list(np.load(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}/fitness_log.npy"))
    time.sleep(1)
    print("Loaded.")
else:
    print("No existing fitness log found. Creating new...//")
    generation_info = []
    time.sleep(1)
    print("Created.")


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

pop = population_loader()
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

if not os.path.isdir(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}"):
    os.mkdir(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}")

os.chdir(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}")
pop.add_reporter(neat.Checkpointer(10))

winner = pop.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
