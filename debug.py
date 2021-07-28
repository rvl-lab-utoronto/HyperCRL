import fire
import hypercrl

import sys

if __name__ == "__main__":
    render = True
    # render = False
    hypercrl.hnet(env="door_pose", robot="IIWA", seed=1223, savepath="./save_dir", render=render)
