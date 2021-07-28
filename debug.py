import fire
import hypercrl

import sys


if __name__ == "__main__":
    render = True
    # render = False
    hypercrl.hnet("door_pose", 1223, "./save_dir", render=render)
