# Just a dummy file which I use to perform various checks.
import numpy as np
from handlers import *
import os

os.chdir("C:/Users/ParthikB/PycharmProjects/retro/training_data")

data = np.load(DATA_NAME+'.npy', allow_pickle=True)

show(data)