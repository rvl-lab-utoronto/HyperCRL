from collections import OrderedDict
import numpy as np
import os

from robosuite.utils.transform_utils import convert_quat

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

from robosuite.models.arenas import Arena, TableArena
from robosuite.models.tasks import Task, UniformRandomSampler, TableTopTask

from robosuite.models.grippers import PandaGripper
from robosuite.models.objects import MujocoXMLObject
from gym.envs.robotics.rotations import (quat2euler, subtract_euler, quat_mul,\
                                        quat2axisangle, quat_conjugate, quat2mat)

def print_quat(obs): 
    print("Position in from world to EE", obs['robot0_robot-state'][:3])
    quat = self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name)
    print("Quat [w, x, y, z]", quat)
    axis, angle = quat2axisangle(quat)
    print("Axis e", axis, "Angle", angle)
    print("euler", quat2euler(quat))

class DoorArena(Arena):
    """Workspace that contains a tabletop with two fixed pegs."""

    def __init__(self, handle):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'assets', 'door', f'{handle}.xml')
        super().__init__(path)
        self.floor = self.worldbody.find("./geom[@name='floor']")

        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))

class OpenDoorTask(Task):
    """
    Creates MJCF model of a open door class task.

    A tabletop task consists of opening the door.
    This class combines the robot, the door+handle
    arena
    """

    def __init__(self, mujoco_arena, mujoco_robot):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
        """
        super().__init__()

        self.merge_robot(mujoco_robot)
        self.merge_arena(mujoco_arena)
        #self.save_model("/home/philiph/model.xml", pretty=True)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.merge(mujoco_arena)

class PandaDoor(RobotEnv):
    """
    This class corresponds to the lifting task for a single robot arm.
    """

    def __init__(
        self,
        robots,
        handle_type="lever",
        mass_scale=1.0,
        handle_ypos = 0.0,
        joint_range = [-1.57, 1.57],
        pose_control=False,
        controller_configs=None,
        gripper_types="PandaGripper",
        gripper_visualizations=False,
        initialization_noise="default",  
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=2.25,
        reward_shaping=True,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=220,
        ignore_done=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        """
        Args:
            robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
                (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
                Note: Must be a single single-arm robot!

            controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
                custom controller. Else, uses the default controller for this specific task. Should either be single
                dict if same controller is to be used for all robots or else it should be a list of the same length as
                "robots" param

            gripper_types (str or list of str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
                with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
                overrides the default gripper. Should either be single str if same gripper type is to be used for all
                robots or else it should be a list of the same length as "robots" param

            gripper_visualizations (bool or list of bool): True if using gripper visualization.
                Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
                robots or else it should be a list of the same length as "robots" param

            initialization_noise (float or list of floats): The scale factor of uni-variate Gaussian random noise
                applied to each of a robot's given initial joint positions. Setting this value to "None" or 0.0 results
                in no noise being applied. Should either be single float if same noise value is to be used for all
                robots or else it should be a list of the same length as "robots" param

            use_camera_obs (bool): if True, every observation includes rendered image(s)

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_scale (float): Scales the normalized reward function by the amount specified

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering

            render_camera (str): Name of camera to render if `has_renderer` is True.

            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

            control_freq (float): how many control signals to receive in every second. This sets the amount of
                simulation time that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_names (str or list of str): name of camera to be rendered. Should either be single str if
                same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
                Note: At least one camera must be specified if @use_camera_obs is True.
                Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                    convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                    robot's camera list).

            camera_heights (int or list of int): height of camera frame. Should either be single int if
                same height is to be used for all cameras' frames or else it should be a list of the same length as
                "camera names" param.

            camera_widths (int or list of int): width of camera frame. Should either be single int if
                same width is to be used for all cameras' frames or else it should be a list of the same length as
                "camera names" param.

            camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
                bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
                "camera names" param.

        """
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        # settings for robot base
        self.base_x_pose = 0.70
        self.base_y_pose = 0.0
        self.base_phi_ang = 3.1415926

        # Handle type
        self.handle_type = handle_type
        self.pose_control = pose_control

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = True

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self._success = False
  
        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False
            )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

        self.modify_door_mass(mass_scale)
        self.modify_door_handle_pos(handle_ypos)
        if self.handle_type != "pull":
            self.modify_door_handle_joint_range(joint_range)
        self.sim.forward()
        self.sim.reset()

    def modify_door_mass(self, scale):
        idx = self.sim.model.body_names.index("door_link")
        self.sim.model.body_mass[idx] *= scale
        self.sim.model.body_inertia[idx] *= scale 
    
    def modify_door_handle_pos(self, ypos):
        idx = self.sim.model.body_names.index("knob_link")
        self.sim.model.body_pos[idx][1] = ypos

    def modify_door_handle_joint_range(self, jntrange):
        idx = self.sim.model.joint_names.index("hinge1")
        self.sim.model.jnt_range[idx][:] = jntrange

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Pushing: in {0, 1}, non-zero if arm has push the cube towards goal

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        reward = 0.

        # sparse completion reward
        # if self._check_success():
        #     reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            reward_dist = -np.linalg.norm(self._get_dist_vec())
            reward_log_dist = -np.log(np.square(np.linalg.norm(reward_dist))+5e-3) - 5.0 
            reward_ori = - np.linalg.norm(self._get_ori_diff_no_xaxis()) if self.pose_control else 0
            reward_door = abs(self._get_door_hinge_pos(sin=False)) *50

            if self.handle_type == "lever" or self.handle_type == "round":
                reward_doorknob = abs(self._get_door_knob_hinge_pos()) * 20 #*50
                reward = reward_door + reward_doorknob + reward_ori + reward_dist + reward_log_dist
            else:
                reward = reward_door + reward_ori + reward_dist + reward_log_dist

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        self.robots[0].robot_model.set_base_xpos((self.base_x_pose, self.base_y_pose, 0))
        self.robots[0].robot_model.set_base_ori([0, 0, self.base_phi_ang])

        # load model for table top workspace
        self.mujoco_arena = DoorArena(self.handle_type)

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = OpenDoorTask(
            self.mujoco_arena,
            [robot.robot_model for robot in self.robots]
        )

        self.model.place_visual()

    def _get_knob_pos(self):
        if self.handle_type == "pull":
            knob = (self.sim.data.geom_xpos[self.knob_id] + self.sim.data.geom_xpos[self.knob_id2]) / 2
        elif self.handle_type == "lever":
            knob = self.sim.data.geom_xpos[self.knob_id]
        elif self.handle_type == "round":
            knob = self.sim.data.geom_xpos[self.knob_id]
        return knob

    def _get_gripper_pos(self):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id]

    def _get_dist_vec(self):
        gripper_pos = self._get_gripper_pos()
        knob_pos = self._get_knob_pos()
        return gripper_pos - knob_pos

    def _get_door_handle_ori(self):
        return self.sim.data.get_body_xquat("knob_link")

    def _get_ori_diff_no_xaxis(self):
        world_q_handle = self._get_door_handle_ori()
        world_q_ee = self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name)

        handle_q_ee = quat_mul(quat_conjugate(world_q_handle), world_q_ee)
        handle_q_des = np.array([0.7071067811865476, 0, -0.7071067811865476, 0.]) # [w, x, y, z]
        #world_q_des = quat_mul(world_q_handle, np.array([0.7071, 0, -0.7071, 0.]))

        des_q_ee = quat_mul(quat_conjugate(handle_q_des), handle_q_ee)
        #des_q_ee2 = quat_mul(quat_conjugate(world_q_des), world_q_ee)
        euler_diff = quat2euler(des_q_ee)
        euler_diff[2] = 0
        return euler_diff

    def _get_door_hinge_pos(self, sin=False):
        pos = self.sim.data.get_joint_qpos("hinge0")
        if not sin:
            return pos
        else:
            return [np.cos(pos), np.sin(pos)]

    def _get_door_knob_hinge_pos(self):
        if self.handle_type == "pull":
            return 0
        return self.sim.data.get_joint_qpos("hinge1")

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        if self.handle_type == "pull":
            self.knob_id = self.sim.model.geom_name2id("door_knob_2")
            self.knob_id2 = self.sim.model.geom_name2id("door_knob_3")
        elif self.handle_type == "round":
            self.knob_id = self.sim.model.geom_name2id("door_knob_2")
        elif self.handle_type == "lever":
            self.knob_id = self.sim.model.geom_name2id("door_knob_4")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset the door
        self.sim.data.set_joint_qpos("hinge0", 0)
        self.sim.data.set_joint_qvel("hinge0", 0)
        if self.handle_type != "pull":
            self.sim.data.set_joint_qpos("hinge1", 0)
            self.sim.data.set_joint_qvel("hinge1", 0)

    def reset(self):
        """
        Reset to a certain position/orientation
        Note the reset_pos is specified as a relative position
            with respect to the door handle frame

        The reset_ori is the rotation/attitude of the desired EE
        with respect to the world frame

        The controller takes an axis-angle [e1, e2, e3], which specify
        a rotation of the EE_t_1 wrt EE_t_0 in the world frame
        """
        obs = super().reset()
        self._success = False

        # Set action space and also set starting pose
        if self.pose_control:
            action = np.zeros(7)
            # Move the gripper behind the joint
            reset_pos = np.array([0.1, 0., 0.]) # wrt to handle
            # reset_ori is the attitude of desired EE wrt world
            # such that it transforms a vector from desired EE frame to world
            reset_ori = np.array([-0.5, -0.5, 0.5, 0.5])
            
            def angle_diff():
                eef_ori = self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name)
                q_diff = quat_mul(quat_conjugate(eef_ori), reset_ori)
                axis, theta = quat2axisangle(q_diff)
                # Here, (axis, theta) specify the rotation of desired wrt body
                # We take the direction vector (axis) and transform it
                # to world frame (axis_world)
                axis_world = quat2mat(eef_ori) @ axis
                return axis_world * theta

            action[:3] = reset_pos - obs['robot0_robot-state'][:3]
            action[3:6] = angle_diff()
            for _ in range(20):
                obs, _, _, _ = self.step(10 * action)
                action[:3] = reset_pos - obs['robot0_robot-state'][:3]
                action[3:6] = angle_diff()

        else:
            reset_pos = np.array([0.1, 0., 0.05]) # Desire_wrt_handle
            action = reset_pos - obs['robot0_robot-state']
            for _ in range(20):
                obs, _, _, _ = self.step(action * 10)
                action = reset_pos - obs['robot0_robot-state']

        return self._get_observation()

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            if not self.pose_control:
                # remove the gripper open dim
                lo = lo[:-1]
                hi = hi[:-1]
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    def step(self, action):
        if not self.pose_control:
            action = np.array(list(action) + [1])
        obs, rew, done, _ = super().step(action)

        self._success |= (self._get_door_hinge_pos() >= 0.2)
        info = {"success": self._success}
        return obs, rew, done, info

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        # low-level object information 
        # Get robot prefix
        # pr = self.robots[0].robot_model.naming_prefix
        # knob = self._get_knob_pos()
        dist = self._get_dist_vec()
        door = self._get_door_hinge_pos(sin=False)
        objects = np.array([door])

        if not self.pose_control:
            dii = {"robot0_robot-state": dist, "object-state": objects}
            return dii

        # Pose Control
        # end effector quaternion in [w, x, y, z] (in wolrd caretian frame)
        world_q_handle = self._get_door_handle_ori()
        world_q_eef = self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name)
        handle_q_eef = quat_mul(quat_conjugate(world_q_handle), world_q_eef)
        gripper = di['robot0_gripper_qpos'][0:1]

        knob = self._get_door_knob_hinge_pos()
        door_v = self.sim.data.get_joint_qvel("hinge0")
        knob_v = self.sim.data.get_joint_qvel("hinge1") if self.handle_type != "pull" else 0

        dii = {"robot0_robot-state": np.concatenate([dist, handle_q_eef, gripper, 
                    di['robot0_joint_pos'], di['robot0_joint_vel']]),
               "object-state": np.array([door_v, knob_v, door, knob]) }
        return dii

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        pass

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        pass

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

if __name__ == "__main__":
    # Create dict to hold options that will be passed to env creation call
    options = {}
 
    options["env_name"] = "PandaDoor"

    options["handle_type"] = "lever"

    options["robots"] = "Panda"
    
    # Choose controller
    controller_name = "OSC_POSE"
    options["pose_control"] = True

    from robosuite.controllers import load_controller_config
    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    import robosuite as suite

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=10,
    )
    print("Model Timestep", env.model_timestep, "Control Timestep", env.control_timestep)
    x_t = env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    print(x_t, low, high)

    # do visualization
    for i in range(10000):
        env.render()
        # action = np.random.uniform(low, high)
        # obs, reward, done, _ = env.step(action)
        # env.render()
        # if done:
        #     env.reset()