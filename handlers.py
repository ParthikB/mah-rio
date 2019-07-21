# This file contains all the basic custom functions and PARAMETERS.
from main import *
import cv2
import os
import neat
import time
########### Parameters ##########
VERSION = 2.1
#################################
# os.mkdir(f"C:/Users/ParthikB/PycharmProjects/retro/checkpoints/v{VERSION}")


def eval_genomes(genomes, config):
    generation_fitness_log = []
    times_level_passed = 0
    for genome_id, genome in genomes:
        state = env.reset()
        action = env.action_space.sample()
        width, height, channels = env.observation_space.shape
        width = int(width/8)    # 28
        height = int(height/8)  # 30

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness       = 0
        current_fitness           = 0
        frame                     = 0
        counter                   = 0
        player_pos                = 0
        player_pos_max            = 0
        done                      = False

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
            # print(info['score'])
            # player_pos     = info['xscrollLo']
            level_end = info["levelHi"]

            # if player_pos > player_pos_max:
            #     current_fitness += 1
            #     player_pos_max = player_pos
            current_fitness += reward

            if level_end == 1:
                current_max_fitness = current_fitness = 100000
                times_level_passed += 1

            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True

            if times_level_passed == 10:
                current_max_fitness = current_fitness = 1000000

            genome.fitness = current_fitness

        generation_fitness_log.append(current_max_fitness)

    best_fitness = max(generation_fitness_log)
    best_fitness_log.append([best_fitness])
    np.save("best_fitness_log", best_fitness_log)


def info(env):
    print("---------- Environment Info ----------")
    print("Env type          : ", type(env))
    print("Action Space      : ", env.action_space)
    print("Observation Space : ", env.observation_space)
    print("--------------------------------------")


def population_loader():
    choice = (input("Do you want to load a population checkpoint (y/n) : ")).lower()
    if choice == 'n':
        print("Creating new population...//")
        time.sleep(2)
        pop = neat.Population(config)  # Use this to start afresh.
    elif choice == 'y':
        version    = float(input("-------Enter the model version     : "))
        checkpoint = int(input("-------Enter the checkpoint number : "))
        try:
            os.chdir(f"C:/Users/ParthikB/PycharmProjects/retro/checkpoint/v{version}")
            print(f"Loading Checkpoint {checkpoint} from model v{version}...//")
            pop = neat.Checkpointer.restore_checkpoint(f"neat-checkpoint-{checkpoint}")
            print(f"Population at Checkpoint {checkpoint} loaded...//")
        except FileNotFoundError:
            print("Invalid Version/Checkpoint. Try again.")
            population_loader()
    else:
        print("Invalid input. Please answer in y/n.")
        population_loader()

    return pop


def show(x):
    for data in x:
        img = data[0]
        action = data[1]
        cv2.imshow("img", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()



