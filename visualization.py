import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from handlers import *

# data = list(np.load(f"C:/Users/ParthikB/PycharmProjects/mario/checkpoint/v{VERSION}/fitness_log.npy"))


def return_list(x):
    list = []
    for i in x:
        list.append(i[0])
    return list


def predict(X, y):
    X, y = pd.DataFrame(X), pd.DataFrame(y)
    model = LinearRegression().fit(X, y)
    future_generations = (np.array(list(range(gen[-1], gen[-1] + 100)))).reshape(-1, 1)
    prediction = model.predict(future_generations)
    # accuracy = sum((max_test/prediction)[0])/len(prediction) * 100
    return return_list(prediction), list(range(gen[-1], gen[-1] + 100))


def plot():
    # Plotting
    fig, ax1 = plt.subplots()

    ax_prediction = ax_fitness = ax1.twinx()
    ax1.set(ylim=(0, 12))
    plt.title(f"Version {VERSION}")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Times it reached Max Fitness")
    ax1.yaxis.label.set_color((0.25, 0.5, 0, 1))
    ax1.bar(gen, rep, color='#408000', alpha=0.5, width=1)
    # ax2.plot(generations, rep, color='k', alpha=0.25, marker='x')
    ax1.tick_params(axis='y')
    # ax2.fill_between(generations, 0, rep, facecolor='g', alpha=0.25)

    #################################### Learn how to make vedio line plot #####################################
    ax_fitness.grid()
    ax_fitness.set_ylabel("Fitness")
    ax_fitness.plot(gen, mean, '#3333ff', label='Mean')
    ax_fitness.plot(gen, max, '#FF8C00', label='Max')
    ax_fitness.tick_params(axis='y')
    ax_fitness.yaxis.label.set_color((0, 0, 0.4, 1))
    # plt.legend()

    ax_prediction.plot(future_generations, predicted_mean, '--b', label='Predicted mean', alpha=0.5)
    ax_prediction.plot(future_generations, predicted_max, '--r', label='Predicted max', alpha=0.3)
    plt.legend(loc=2)

    # plt.subplots_adjust(left=0.25)
    # plt.gcf().text(0.02, 0.5, "Heyaaaa", fontsize=14)

    fig.tight_layout()
    plt.show()


def hex_bin():
    plt.hexbin(gen, max, gridsize=(15, 15))
    plt.show()


# mean = []
# max = []
# rep = []
# generations = []
#
# for x in data:
#     mean.append(x[0])
#     max.append(x[1])
#     rep.append(x[2])
#
# for i in range(len(max)):
#     generations.append(i)

max, mean, gen, rep = data_loader()

predicted_max, future_generations = predict(gen, max)
predicted_mean, future_generations = predict(gen, mean)


# hex_bin()
plot()