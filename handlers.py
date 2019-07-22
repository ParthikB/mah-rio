
VERSION = 0.0

def info(env):
    print("------- Environment Parameters -------")
    print("Environment type  :", type(env))
    print("Observation Space :", env.observation_space)
    print("Action Space      :", env.action_space)
    print("--------------------------------------")


