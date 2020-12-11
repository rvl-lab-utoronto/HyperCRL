import numpy as np
import os
import gym
import os.path as osp


from gym import utils
from gym.envs.mujoco import mujoco_env, ReacherEnv

import xml.etree.ElementTree as ET
import tempfile

class ReacherLengthEnv(ReacherEnv, utils.EzPickle):
    def __init__(
            self,
            body_parts=["link0", "link1"],
            length=[0.1, 0.1],
            *args,
            **kwargs):

        assert isinstance(self, mujoco_env.MujocoEnv)

        model_path = os.path.join(os.path.dirname(gym.envs.mujoco.__file__), 'assets', 'reacher.xml')
        # find the body_part we want
        tree = ET.parse(model_path)
        for body_part, leng in zip(body_parts, length):

            # grab the geoms
            geom = tree.find(".//geom[@name='%s']" % body_part)

            fromto = []
            for x in geom.attrib["fromto"].split(" "):
                fromto.append(float(x))
            fromto[3] = leng
            geom.attrib["fromto"] = " ".join([str(x) for x in fromto])

        # Modify the frame position of second link

        body_parts = ['body1', 'fingertip']
        lengths = [length[0], length[1]+0.01]

        for body_part, length in zip(body_parts, lengths):
            body = tree.find(".//body[@name='{}']".format(body_part))
            pos = []
            for x in body.attrib["pos"].split(" "):
                pos.append(float(x))
            pos[0] = length
            body.attrib['pos'] = " ".join([str(x) for x in pos])


        # create new xml
        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        # load the modified xml
        mujoco_env.MujocoEnv.__init__(self, model_path=file_path, frame_skip=2)
        utils.EzPickle.__init__(self)

if __name__ == "__main__":
    env = ReacherLengthEnv(length=[0.1, 0.15])
    env.reset()
    while True:
        env.render()
        x, reward, done, _ = env.step(env.action_space.sample())