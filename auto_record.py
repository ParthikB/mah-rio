import pyautogui as py
import time
# 37, 347
# 226, 140
# 1826, 164
print(py.position())


def countdown(min):
    for m in list(range(min))[::-1]:
        for s in list(range(60))[::-1]:
            print(f"Recording vedio in {m}:{s} minutes...")
            time.sleep(1)


for gen in range(50):
    print("new vedio")

    # py.hotkey('win', 'd')
    # time.sleep(0.5)

    py.click(23, 340)
    time.sleep(1)
    print("mario")

    py.hotkey('win', 'g')
    time.sleep(3)
    print("gamebar")

    py.click(226, 140)
    print("recording")
    time.sleep(3)

    py.click(1033, 337)
    print("minimize")

    # for how long the video shoulf be recorded...
    time.sleep(180)

    py.click(1826, 164)
    print("Recorded!")

    py.click(1033, 337)

    countdown(min=60)

