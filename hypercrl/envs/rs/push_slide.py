from collections import OrderedDict
import numpy as np
import os

from robosuite.utils.transform_utils import convert_quat, quat2axisangle
from robosuite.utils.mjcf_utils import new_site

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import TableTopTask, UniformRandomSampler

from robosuite.models.grippers import PandaGripper
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE

from .generated_objects import TwoCubeV2

class PandaSlide(RobotEnv):
    """
    This class corresponds to the lifting task for a single robot arm.
    """

    def __init__(
        self,
        robots,
        start_poses = [[-0.04, 0, 0.84029956], [0, 0, 0, 0], 
                      [0.04, 0., 0.84029956], [0, 0, 0, 0]],
        density=[50, 50],
        controller_configs=None,
        gripper_types="PandaGripper",
        gripper_visualizations=False,
        initialization_noise="default",
        table_full_size=(1.2, 0.8, 0.8),
        table_friction=(0.01, 1, 1),
        box2_friction=0.001,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=4.0,
        reward_shaping=True,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=50,
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

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

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

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.box2_friction = box2_friction
        self.density = density

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = True

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        self.include_z_obs = True
        
        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.4, 0.4],
                y_range=[-0.4, 0.4],
                ensure_object_boundary_in_range=False
            )

        # start pose [pos1, quat1, pos2, quat2]
        self.start_poses = start_poses
        self.gripper_start_x = min(start_poses[0][0], start_poses[1][0]) - 0.06

        self.goal_locations = [[0.25, -0.03], [0.25, 0.03], [0.31, -0.03], [0.31, 0.03]]
        self.goal_indicator_loc = [0.25, 0., 0.82]

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

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            # HACK: remove the z_dim and gripper open dim
            lo = lo[:-2]
            hi = hi[:-2]
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    def step(self, action, move_z=False):
        if move_z:
            action = np.array(list(action) + [1])
        else:
            action = np.array(list(action) + [0, 1])
        
        obs, reward, done, info = super().step(action)
        info['success'] = self._check_success()
        return obs, reward, done, info

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

        # # sparse completion reward
        # if self._check_success():
        #     print("success")
        #     reward = 2.0

        # use a shaping reward
        if self.reward_shaping:
            
            # distance between the cube and the goal cube location
            for i in range(4):
                goal = self.goal_locations[i]
                corner = self.sim.data.site_xpos[self.corner_ids[4 + i]][:2]
                dist = np.linalg.norm(goal - corner)
                reward += 1 - np.tanh(10.0 * dist)
            
            reward -= 0.1 * np.linalg.norm(action[:2])

        return reward * self.reward_scale / 4.0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # # Add goal indicator
        # self.mujoco_arena.worldbody.append(new_site(name="goal", rgba=(0.8, 0, 0, 1),
        #      pos=self.goal_indicator_loc, size=(0.02,), group="1"))

        box = TwoCubeV2("box1", density_left=self.density[0], density_right=self.density[1],
            rgba_handle_1=RED, rgba_handle_2=GREEN, friction="0.011 1 0.0001")

        box2 = TwoCubeV2("box2", density_body=self.density[0], density_right=self.density[1],
            rgba_handle_1=RED, rgba_handle_2=BLUE, friction=f"0.011 1 {self.box2_friction}")

        self.mujoco_objects = OrderedDict([("box", box), ("box2", box2)])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            [robot.robot_model for robot in self.robots],
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.right_finger_geoms
        ]

        name = "box1"
        name2 = "box2"
        self.center1_id = self.sim.model.site_name2id(name + "_pot_center")
        self.center2_id = self.sim.model.site_name2id(name2 + "_pot_center")

        self.corner_ids = [self.sim.model.site_name2id(name + f"_corner_{i}") for i in range(1, 5)]
        self.corner_ids.extend([self.sim.model.site_name2id(name2 + f"_corner_{i}") for i in range(1, 5)])

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Determineistic reset
        obj_pos, obj_quat = np.ones((2,3)),np.ones((2,4))
        obj_pos[0] = self.start_poses[0]
        obj_quat[0] = self.start_poses[1]
        obj_pos[1] = self.start_poses[2]
        obj_quat[1] = self.start_poses[3]

        # obj_pos[0] = [-0.04, 0, 0.84029956]
        # obj_quat[0] = [0, 0, 0, 0]
        # obj_pos[1] = [0.04, 0, 0.84029956]
        # obj_quat[1] = [0, 0, 0, 0]

        # Loop through all objects and reset their positions
        for i, (obj_name, _) in enumerate(self.mujoco_objects.items()):
            self.sim.data.set_joint_qpos(obj_name, np.concatenate([obj_pos[i], obj_quat[i]]))

    def reset(self):
        self.include_z_obs = True
        super().reset()

        # Move the gripper to a fixed position in the beginning
        reset_pos = np.array([self.gripper_start_x, 0, 0.84])

        # calculate the offset to goal pos
        direction = reset_pos - self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for _ in range(20):
            # move to the goal in 20 steps
            self.step(direction * 10, move_z=True)
            direction = reset_pos - self.sim.data.site_xpos[self.robots[0].eef_site_id]
        
        # print("Reset arm at ", obs['robot0_robot-state'], "to the cube")
        self.include_z_obs = False
        return self._get_observation()

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
        if self.use_object_obs:
            # Get robot prefix
            # pr = self.robots[0].robot_model.naming_prefix
            
            if self.include_z_obs:
                # position of object relatively
                cent1_pos = np.array(self.sim.data.site_xpos[self.center1_id])
                cent2_pos = np.array(self.sim.data.site_xpos[self.center2_id])
                gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            else:
                cent1_pos = np.array(self.sim.data.site_xpos[self.center1_id][:2])
                cent2_pos = np.array(self.sim.data.site_xpos[self.center2_id][:2])
                gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id][:2])

            relative_gripper_pos = cent1_pos - gripper_site_pos 

            # position of corners
            corners = np.concatenate([
                np.array(self.sim.data.site_xpos[self.corner_ids[j]][:2])
                for j in range(0, 8)
            ])

        dii = {"robot0_robot-state": relative_gripper_pos, "object-state": corners}
        return dii

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        success = True
        for i in range(4):
            corner_pos = np.array(self.sim.data.site_xpos[self.corner_ids[4+i]])
            dist = np.linalg.norm(corner_pos[:2] - self.goal_locations[i])
            success &= (dist <= 0.05)
        return success

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.robots[0].gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
                    - self.sim.data.get_site_xpos(self.robots[0].gripper.visualization_sites["grip_site"])
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.robots[0].eef_site_id] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

if __name__ == "__main__":
    # Create dict to hold options that will be passed to env creation call
    options = {}
 
    options["env_name"] = "PandaSlide"

    options["robots"] = "Panda"
    
    # Choose controller
    controller_name = "OSC_POSITION"

    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper
    import robosuite as suite

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")


    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=10,
    )

    # Get action limits
    low, high = env.action_spec

    env = GymWrapper(env)
    env.reset()

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
