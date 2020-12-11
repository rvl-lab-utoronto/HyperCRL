import torch
import numpy as np
from gym.envs.robotics.rotations import quat2euler, quat_mul

class GTCost():
    def __init__(self, clenv_name, state_dim, control_dim, reward_discount, gpuid):
        self.env_name = clenv_name
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.reward_discount = reward_discount
        if self.env_name == "metaworld10":
            self.metaworld_rew = MetaWorldRew(gpuid)
        
        self.cartpole_x = [0, -5, 5]
        self.pole_length = [0.6, 0.8, 0.4, 1.0, 1.2, 0.7, 0.5, 0.9, 1.1, 1.3]
        self.suite_push_goal = torch.tensor([[0.07, -0.08], [0.07, 0.08], [0.23, -0.08], [0.23, 0.08]], device=gpuid)
        self.suite_push_tolerance =  torch.tensor([0.08, 0.08], device=gpuid)
        self.suite_rot_goal = torch.tensor([[-0.08, -0.08], [-0.08, 0.08], [0., -0.08], [0., 0.08],
                [0., -0.08], [0., 0.08], [0.08, -0.08], [0.08, 0.08]], device=gpuid)
        self.suite_slide_goal = torch.tensor([[0.25, -0.03], [0.25, 0.03], [0.31, -0.03], [0.31, 0.03]], device=gpuid)
        self.des_q_w = np.array([[0.7071067811865476, 0, 0.7071067811865476, 0.]]) # [cos(pi/2), 0, sin(pi/2), 0]

    def __call__(self, x, u, t, task_id):
        x = x.view(-1, self.state_dim)
        u = u.view(-1, self.control_dim)
        if self.env_name.startswith("lqr"):
            cost = 0.1 * torch.sum((x * x), dim=-1) + torch.sum((u * u), dim=-1)
        elif self.env_name.startswith("pendulum"):
            theta = torch.atan2(x[:, 1], x[:, 0])
            cost = theta ** 2 + 0.1 * x[:, 2] ** 2 + 0.001 * u[:, 0] ** 2
        elif self.env_name.startswith("hopper"):
            alive = torch.isfinite(x).prod(dim=1) * (torch.abs(x[:, 2:]) < 100).prod(dim=1) \
                    * (x[:, 1] > 0.7) * (torch.abs(x[:, 2]) < 0.2)
            reward = alive * (1 + x[:, 0] - 1e-3 * (u**2).sum(dim=-1))
            cost = -reward
        elif self.env_name == "walker":
            alive = (x[:, 1] > 0.8) * (x[:, 1] < 2.0) \
                    *(x[:, 2] > -1.0) * (x[:, 2] < 1.0)
            reward = alive * (1 + x[:, 0] - 1e-3 * (u**2).sum(dim=-1))
            cost = -reward
        elif self.env_name.startswith("half_cheetah"):
            reward_ctrl = -0.1 * (u ** 2).sum(dim=1)
            reward_run = x[:, 0]
            cost = - reward_run - reward_ctrl
        elif self.env_name.startswith("inverted_pendulum"):
            reward = torch.isfinite(x).prod(dim=1) * (torch.abs(x[:, 1]) <= .2)
            cost = -reward
        elif self.env_name == "metaworld10":
            reward = self.metaworld_rew.reward(x, u, t, task_id)
            cost = -reward
        elif self.env_name == "cartpole_bin":
            l = 0.6
            a = torch.pow((x[:, 0] - self.cartpole_x[task_id]- l * torch.sin(x[:, 1])), 2) + torch.pow((-l * torch.cos(x[:, 1]) - l), 2)
            reward = torch.exp(-a/(l*l))
            reward -= 0.01 * torch.sum(torch.pow(u, 2), dim=-1)
            cost = -reward
        elif self.env_name == "cartpole":
            l = self.pole_length[task_id]
            a = torch.pow((x[:, 0] - l * torch.sin(x[:, 1])), 2) + torch.pow((-l * torch.cos(x[:, 1]) - l), 2)
            reward = torch.exp(-a/(l*l))
            reward -= 0.01 * torch.sum(torch.pow(u, 2), dim=-1)
            cost = -reward
        elif self.env_name == "reacher":
            cost_dist = torch.norm(x[:, -3:], p=2, dim=-1)
            cost_ctrl = (u**2).sum(dim=-1)
            cost = cost_dist + cost_ctrl
        elif self.env_name == "pusher":
            reward = 0
            c1 = x[:, 2:4]; c2 = x[:, 4:6]; c3 = x[:, 6:8]; c4 = x[:, 8:10]
            for i, corner in enumerate([c1, c2, c3, c4]):
                diff = torch.norm(corner - self.suite_push_goal[i], p=2, dim=-1)
                reward += 1 - torch.tanh(10.0 * diff)
            reward -= 0.25 * torch.norm(u, p=2, dim=-1)
            cost = -reward
        elif self.env_name == "pusher_rot":
            reward = 0
            for i in range(8):
                corner = x[:, 4+i*2 : 4+(i+1)*2]
                diff = torch.norm(corner - self.suite_rot_goal[i], p=2, dim=-1)
                reward += 1 - torch.tanh(10.0 * diff)
            reward -= 0.1 * torch.norm(u, p=2, dim=-1)
            cost = -reward
        elif self.env_name == "pusher_slide":
            reward = 0
            c1 = x[:, 10:12]; c2 = x[:, 12:14]; c3 = x[:, 14:16]; c4 = x[:, 16:18]
            for i, corner in enumerate([c1, c2, c3, c4]):
                diff = torch.norm(corner - self.suite_slide_goal[i], p=2, dim=-1)
                reward += 1 - torch.tanh(10.0 * diff)
            reward -= 0.1 * torch.norm(u, p=2, dim=-1)
            cost = -reward
        elif self.env_name == "door":
            dist = torch.norm(x[:, :3], p=2, dim=-1)
            reward_dist = -dist
            reward_log_dist = -torch.log(torch.pow(dist, 2) + 5e-3) - 5.0 
            reward_ori = 0
            reward_door = torch.abs(x[:, -1]) *50
            reward_doorknob = 0
            cost = - reward_door - reward_doorknob - reward_ori - reward_dist - reward_log_dist
        elif self.env_name == "door_pose":
            dist = torch.norm(x[:, :3], p=2, dim=-1)
            reward_dist = -dist
            reward_log_dist = -torch.log(torch.pow(dist, 2) + 5e-3) - 5.0
            euler_diff = quat2euler(quat_mul(np.tile(self.des_q_w, (x.size(0), 1)), x[:, 3:7].detach().cpu().numpy()))
            euler_diff = torch.tensor(euler_diff, dtype=x.dtype, device=x.device)
            reward_ori = -torch.norm(euler_diff[:, 0:2], p=2, dim=-1)
            reward_door = torch.abs(x[:, -2]) * 50
            reward_doorknob = torch.abs(x[:, -1]) * 20
            cost = -reward_door - reward_doorknob - reward_ori - reward_dist - reward_log_dist
        return cost
    
    def reward(self, x, u, t, task_id):
        cost = self.__call__(x, u, t, task_id)
        return -cost

class MetaWorldRew():
    def __init__(self, gpuid):
        # Goals
        reach_goal = torch.tensor([-0.1, 0.8, 0.2])
        push_goal = torch.tensor([0.1, 0.8, 0.02])
        pick_place_goal = torch.tensor([0.1, 0.8, 0.2])

        # Inital pose of finger
        init_fingerCOM = torch.tensor([-0.03322134,  0.50420014,  0.19491914])
        # Initial pose of object
        obj_init_pos = torch.tensor([0, 0.6, 0.02])
        # Initial Height of object on table
        objHeight = torch.tensor([0.01492813342356])

        # Target Height for grasping
        liftThresh = 0.04
        heightTarget = objHeight + liftThresh


        maxReachDist = torch.norm(init_fingerCOM - reach_goal, 2, dim=-1)
        maxPushDist = torch.norm(obj_init_pos[:2] - push_goal[:2], p=2)
        maxPlacingDist = torch.norm(torch.Tensor([obj_init_pos[0], obj_init_pos[1], heightTarget]) \
            - pick_place_goal) + heightTarget

        self.reach_goal = reach_goal.to(gpuid)
        self.push_goal = push_goal.to(gpuid)
        self.pick_place_goal = pick_place_goal.to(gpuid)
        self.init_fingerCOM = init_fingerCOM.to(gpuid)
        self.obj_init_pos = obj_init_pos.to(gpuid)
        self.objHeight = objHeight.to(gpuid)
        self.heightTarget = heightTarget.to(gpuid)
        self.maxReachDist = maxReachDist.to(gpuid)
        self.maxPushDist = maxPushDist.to(gpuid)
        self.maxPlacingDist = maxPlacingDist.to(gpuid)

        # Shaped reward constants
        self.c1 = 1000
        self.c2 = 0.01
        self.c3 = 0.001

    def reward(self, x, u, t, task_id):
        fingerCOM = torch.stack((x[:, 0], x[:, 1], x[:, 2]-0.045), dim=1)
        objPos = x[:, 3:6].clone()
        
        def reach_v1():
            reach_goal = self.reach_goal
            maxReachDist = self.maxReachDist
            c1, c2, c3 = self.c1, self.c2, self.c3

            reachDist = torch.norm(fingerCOM - reach_goal, p=2, dim=-1)
            reachRew = c1*(maxReachDist - reachDist) + c1*(torch.exp(-(reachDist**2)/c2) + torch.exp(-(reachDist**2)/c3))
            reachRew = torch.relu(reachRew)
            return reachRew

        def push_v1():
            push_goal = self.push_goal
            maxPushDist = self.maxPushDist
            c1, c2, c3 = self.c1, self.c2, self.c3

            reachDist = torch.norm(fingerCOM - objPos, p=2, dim=-1)
            pushDist = torch.norm(objPos[:, :2] - push_goal[:2], p=2, dim=-1)
            reachRew = -reachDist

            pushRew = 1000*(maxPushDist - pushDist) + c1*(torch.exp(-(pushDist**2)/c2) + torch.exp(-(pushDist**2)/c3))
            pushRew = (reachDist < 0.05) * torch.relu(pushRew)
                
            reward = reachRew + pushRew
            return reward

        def pick_place_v1():
            pick_place_goal = self.pick_place_goal
            maxPlacingDist = self.maxPlacingDist
            heightTarget = self.heightTarget
            init_fingerCOM = self.init_fingerCOM
            objHeight = self.objHeight
            c1, c2, c3 = self.c1, self.c2, self.c3

            reachDist = torch.norm(fingerCOM - objPos, p=2, dim=-1)
            placingDist = torch.norm(objPos - pick_place_goal, p=2, dim=-1)

            def reachReward():
                # Finger-obj Distance on xy plane
                reachDistxy = torch.norm(objPos[:, :-1] - fingerCOM[:, :-1], p=2, dim=-1)
                # finger-moved distance on z plane
                zRew = torch.abs(fingerCOM[:, -1] - init_fingerCOM[-1])
                cond = reachDistxy < 0.05
                # Close in on xy-distance first, then in z-direction
                reachRew = cond * (-reachDist) + (~cond) * (-reachDistxy - 2*zRew)
                #incentive to close fingers when reachDist is small
                cond = reachDist < 0.05
                reachRew = cond * (-reachDist + torch.relu(u[:, -1])/50) \
                        + (~cond) * reachRew
                return reachRew

            # check if object is picked above target height
            tolerance = 0.01 # about 25% error
            pickCompleted = objPos[:, 2] >= (heightTarget - tolerance)

            # check if object on the ground, far away from the goal, and from the gripper
            objDropped = (objPos[:, 2] < (objHeight + 0.005)) * \
                        (placingDist > 0.02) * (reachDist > 0.02) 

            def orig_pickReward():
                hScale = 100
                # pick reward encourges obj to be at target height
                cond = pickCompleted * (~objDropped)
                pickRew = cond * hScale * torch.min(heightTarget, objPos[:, 2]) 
                return pickRew

            def placeReward():
                cond = pickCompleted *  (reachDist < 0.1) * (~objDropped)
                placeRew = cond * torch.relu(1000*(maxPlacingDist - placingDist) \
                    + c1*(torch.exp(-(placingDist**2)/c2) + torch.exp(-(placingDist**2)/c3)))
                return placeRew

            reachRew = reachReward()
            pickRew = orig_pickReward()
            placeRew = placeReward()
            reward = reachRew + pickRew + placeRew
            return reward

        def door_open_v1():
            pass
        def drawer_open_v1():
            pass
        def drawer_close_v1():
            pass
        def button_press_topdown_v1():
            pass
        def peg_insert_side_v1():
            pass
        def window_open_v1():
            pass
        def window_close_v1():
            pass
        
        reward_fn = [reach_v1, push_v1, pick_place_v1, door_open_v1, drawer_open_v1,
                drawer_close_v1, button_press_topdown_v1, peg_insert_side_v1,
                window_open_v1, window_close_v1]
        
        reward = reward_fn[task_id]()
        return reward

def test_metaworld():
    from metaworld.benchmarks import MT10
    import numpy as np
    import random
    env = MT10.get_train_tasks()
    gpuid = 'cuda:0'
    rew_fn = MetaWorldRew(gpuid)

    env.seed(0)
    np.random.seed(0)
    random.seed(0)
    for task_id in range(2, 3):
        env.set_task(task_id)
        x_t = env.reset()

        x_torch = []
        u_torch = []
        rew_gt = []
        for i in range(150):
            u = env.action_space.sample()
            x_tt, rew, done, info = env.step(u)
            rew_gt.append(rew)
            x_torch.append(torch.FloatTensor(x_tt[:9]).view(1, -1))
            u_torch.append(torch.FloatTensor(u).view(1, -1))
            x_t = x_tt

        # x_t = env.reset()
        # for i in range(150):
        #     u = env.action_space.sample()
        #     x_tt, rew, done, info = env.step(u)
        #     rew_gt.append(rew)
        #     x_torch.append(torch.FloatTensor(x_tt[:9]).view(1, -1))
        #     u_torch.append(torch.FloatTensor(u).view(1, -1))
        #     x_t = x_tt

        x_torch = torch.cat(x_torch, dim=0).to(gpuid)
        u_torch = torch.cat(u_torch, dim=0).to(gpuid)

        rew = rew_fn.reward(x_torch, u_torch, 0, task_id)
        rew_gt = torch.tensor(rew_gt).to(gpuid)
        diff = torch.abs((rew_gt - rew)/rew_gt)
        print("Task", task_id)
        print("max diff", diff.max().item())
        print("mean diff", diff.mean().item())

def test_mujoco(env, name, task_id=0):
    gpuid = 'cuda:0'
    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]
    cost_fn = GTCost(name, x_dim, u_dim, 1.0, gpuid)

    x_t = env.reset()
    x_torch = []
    u_torch = []
    rew_gt = []
    not_dones = []
    dones = []
    for i in range(200):
        u = env.action_space.sample()
        x_tt, rew, done, info = env.step(u)
        rew_gt.append(rew)
        x_torch.append(torch.FloatTensor(x_tt).view(1, x_dim))
        u_torch.append(torch.FloatTensor(u).view(1, u_dim))
        x_t = x_tt
        if done:
            dones.append(i)
            x_t = env.reset()
        else:
            not_dones.append(i)

    x_torch = torch.cat(x_torch, dim=0).to(gpuid)
    u_torch = torch.cat(u_torch, dim=0).to(gpuid)
    rew = -cost_fn(x_torch, u_torch, 0, task_id)
    rew_gt = torch.tensor(rew_gt).to(gpuid)
    diff = torch.abs((rew_gt[not_dones] - rew[not_dones]) / rew_gt[not_dones])
    print(f"Env {name}")
    print(f"max diff, {diff.max().item():.4f}")
    print(f"mean diff, {diff.mean().item():.4f}")

    print("reward for done states", rew[dones].tolist())
    print()

def test_hopper():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    import hypercrl.envs.mujoco

    name = 'hopper'

    env = gym.make('MBRLHopper-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('HopperGravityHalf-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('HopperBigTorso-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('HopperSmallFoot-v0')
    test_mujoco(env, name)
    env.close()

def test_cheetah():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    import hypercrl.envs.mujoco
    name = 'half_cheetah'

    env = gym.make('HalfCheetahGravityHalf-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('HalfCheetahSmallTorso-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('HalfCheetahBigHead-v0')
    test_mujoco(env, name)
    env.close()

def test_walker():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    import hypercrl.envs.mujoco
    name = 'walker'

    env = gym.make('Walker2dGravityHalf-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('Walker2dBigTorso-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('Walker2dBigThigh-v0')
    test_mujoco(env, name)
    env.close()

def test_reacher():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    import hypercrl.envs.mujoco
    name = 'reacher'

    env = gym.make('Reacher-v2')
    test_mujoco(env, name)
    env.close()

    env = gym.make('ReacherLong1-v0')
    test_mujoco(env, name)
    env.close()


def test_inverted_pendulum():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    import hypercrl.envs.mujoco
    name = 'inverted_pendulum'

    env = gym.make('InvertedPendulum-v2')
    test_mujoco(env, name)
    env.close()

    env = gym.make('InvertedPendulumSmallPole-v0')
    test_mujoco(env, name)
    env.close()

    env = gym.make('InvertedPendulumBigPole-v0')
    test_mujoco(env, name)
    env.close()

def test_cartpole():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    import hypercrl.envs

    # name="cartpole_bin"

    # env = gym.make('MBRLCartpole-v0')
    # test_mujoco(env, name, 0)
    # env.close()

    # env = gym.make('CartpoleLeft1-v0')
    # test_mujoco(env, name, 1)
    # env.close()

    # env = gym.make('CartpoleRight1-v0')
    # test_mujoco(env, name, 2)
    # env.close()

    name = "cartpole"
    env = gym.make('MBRLCartpole-v0')
    test_mujoco(env, name, 0)
    env.close()

    env = gym.make('CartpoleLong1-v0')
    test_mujoco(env, name, 1)
    env.close()

    env = gym.make('CartpoleShort1-v0')
    test_mujoco(env, name, 2)
    env.close()

    env = gym.make('CartpoleLong2-v0')
    test_mujoco(env, name, 3)
    env.close() 

    env = gym.make('CartpoleLong3-v0')
    test_mujoco(env, name, 4)
    env.close() 

def test_pusher():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    from hypercrl.envs.rs import PandaCL

    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper

    env = suite.make(env_name="PandaCL", density=[10000, 10000], robots="Panda",
                controller_configs= load_controller_config(default_controller="OSC_POSITION"),
                has_renderer=False)
    env = GymWrapper(env)

    name = "pusher"

    test_mujoco(env, name, 0)

def test_door():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    from hypercrl.envs.rs import PandaDoor

    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper

    env = suite.make(env_name="PandaDoor", handle_type="pull", robots="Panda",
                controller_configs = load_controller_config(default_controller="OSC_POSE"),
                pose_control=True, has_renderer=False)
    env = GymWrapper(env)
    name = "door_pose"
    test_mujoco(env, name, 0)
    env.close()

    env = suite.make(env_name="PandaDoor", handle_type="round", robots="Panda",
                controller_configs = load_controller_config(default_controller="OSC_POSE"),
                pose_control=True, has_renderer=False)
    env = GymWrapper(env)
    name = "door_pose"
    test_mujoco(env, name, 0)
    env.close()

    env = suite.make(env_name="PandaDoor", handle_type="lever", robots="Panda",
                controller_configs = load_controller_config(default_controller="OSC_POSE"),
                pose_control=True, has_renderer=False)
    env = GymWrapper(env)
    name = "door_pose"
    test_mujoco(env, name, 0)
    env.close()

    env = suite.make(env_name="PandaDoor", handle_type="pull", robots="Panda",
                controller_configs= load_controller_config(default_controller="OSC_POSITION"),
                has_renderer=False)
    env = GymWrapper(env)

    name = "door"

    test_mujoco(env, name, 0)

def test_pusher_rot():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    from hypercrl.envs.rs import PandaRot

    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper

    env = suite.make(env_name="PandaRot", robots="Panda",
                controller_configs= load_controller_config(default_controller="OSC_POSITION"),
                has_renderer=False)
    env = GymWrapper(env)

    name = "pusher_rot"

    test_mujoco(env, name, 0)

def test_pusher_slide():
    import sys
    sys.path.insert(0,'/home/philiph/Documents/Continual-Learning')
    import gym
    from hypercrl.envs.rs import PandaSlide

    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper

    env = suite.make(env_name="PandaSlide", robots="Panda",
                controller_configs= load_controller_config(default_controller="OSC_POSITION"),
                has_renderer=False)
    env = GymWrapper(env)

    name = "pusher_slide"

    test_mujoco(env, name, 0)

if __name__ == "__main__":
    # test_hopper()
    # test_cheetah()
    # test_walker()
    # test_inverted_pendulum()
    # test_metaworld()
    #test_reacher()
    #test_cartpole()
    #test_pusher()
    #test_pusher_rot()
    test_pusher_slide()
    #test_door()