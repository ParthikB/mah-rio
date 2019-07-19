# This file contains all the basic custom functions and PARAMETERS.

import cv2

########### Parameters ##########
GITHUB_VER = 0.1
#################################


def show(x):
    for data in x:
        img = data[0]
        action = data[1]
        cv2.imshow("img", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()


def info(env):
    print("---------- Environment Info ----------")
    print("Env type          : ", type(env))
    print("Action Space      : ", env.action_space)
    print("Observation Space : ", env.observation_space)
    print("--------------------------------------")
    
