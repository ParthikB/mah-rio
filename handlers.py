import numpy as np

VERSION = '4.0.2'

def info(env):
    print("------- Environment Parameters -------")
    print("Environment type  :", type(env))
    print("Observation Space :", env.observation_space)
    print("Action Space      :", env.action_space)
    print("--------------------------------------")


def data_loader(division=1):
    data = list(np.load(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}/fitness_log.npy"))

    mean = []
    max = []
    rep = []
    gen = []

    for x in data[0::division]:
        mean.append(x[0])
        max.append(x[1])
        rep.append(x[2])

    for i in range(len(max)):
        gen.append(i*division)


    return max, mean, gen, rep


if __name__ == '__main__':
    max, mean, gen, rep = data_loader(25)

