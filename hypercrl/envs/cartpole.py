import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import xml.etree.ElementTree as ET
import tempfile

class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, model_path=None, pendulum_length=0.6):
        if model_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            model_path = '%s/assets/cartpole.xml' % dir_path

        self.pendulum_length = pendulum_length

        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = self.pendulum_length
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, self.pendulum_length]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def _get_ee_pos(self, x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - self.pendulum_length * np.sin(theta),
            -self.pendulum_length * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

class CartpoleBinEnv(CartpoleEnv):
    def __init__(self, offset = 0):
        self.offset = offset
        super(CartpoleBinEnv, self).__init__()

    def _get_obs(self):
        obs = super(CartpoleBinEnv, self)._get_obs()
        obs[0] += self.offset
        return obs

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = self.pendulum_length
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([self.offset, self.pendulum_length]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

class CartpoleLengthEnv(CartpoleEnv, utils.EzPickle):
    def __init__(
            self,
            body_parts=["cpole"],
            length=0.6,
            *args,
            **kwargs):

        assert isinstance(self, mujoco_env.MujocoEnv)

        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'cartpole.xml')
        # find the body_part we want
        tree = ET.parse(model_path)
        for body_part in body_parts:

            # grab the geoms
            geom = tree.find(".//geom[@name='%s']" % body_part)

            fromto = []
            for x in geom.attrib["fromto"].split(" "):
                fromto.append(float(x))
            fromto[-1] = -length
            geom.attrib["fromto"] = " ".join([str(x) for x in fromto])

        # create new xml
        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        # load the modified xml
        CartpoleEnv.__init__(self, model_path=file_path, pendulum_length=length)
        utils.EzPickle.__init__(self)

if __name__ == "__main__":
    env = CartpoleLengthEnv(length=1.2)
    env.reset()
    while True:
        env.render()
        x = env.step(env.action_space.sample())