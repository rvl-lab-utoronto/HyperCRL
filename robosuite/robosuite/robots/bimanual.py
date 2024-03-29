import numpy as np

from collections import OrderedDict

import robosuite.utils.transform_utils as T

from robosuite.models.grippers import gripper_factory
from robosuite.controllers import controller_factory, load_controller_config

from robosuite.robots.robot import Robot

import os


class Bimanual(Robot):
    """Initializes a bimanual robot, as defined by a single corresponding XML"""

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        gripper_type="default",
        gripper_visualization=False,
        control_freq=10
    ):
        """
        Args:
            robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

            idn (int or str): Unique ID of this robot. Should be different from others

            controller_config (dict or list of dict): If set, contains relevant controller parameters for creating
                custom controllers. Else, uses the default controller for this specific task. Should either be single
                dict if same controller is to be used for both robot arms or else it should be a list of length 2.
                NOTE: In the latter case, assumes convention of [right, left]

            initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
                instantiated for the task

            initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
                corresponding value types are specified below:
                "magnitude": The scale factor of uni-variate random noise applied to each of a robot's given initial
                    joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                    If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                    If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
                "type": Type of noise to apply. Can either specify "gaussian" or "uniform"
                Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

            gripper_type (str or list of str): type of gripper, used to instantiate
                gripper models from gripper factory. Default is "default", which is the default gripper associated
                within the 'robot' specification. None removes the gripper, and any other (valid) model overrides the
                default gripper. Should either be single str if same gripper type is to be used for both arms or else
                it should be a list of length 2
                NOTE: In the latter case, assumes convention of [right, left]

            gripper_visualization (bool or list of bool): True if using gripper visualization.
                Useful for teleoperation. Should either be single bool if gripper visualization is to be used for both
                arms or else it should be a list of length 2
                NOTE: In the latter case, assumes convention of [right, left]

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.
        """

        self.controller = self._input2dict(None)
        self.controller_config = self._input2dict(controller_config)
        self.gripper = self._input2dict(None)
        self.gripper_type = self._input2dict(gripper_type)
        self.has_gripper = self._input2dict([gripper_type is not None for _, gripper_type in self.gripper_type.items()])
        self.gripper_visualization = self._input2dict(gripper_visualization)
        self.control_freq = control_freq

        self.gripper_joints = self._input2dict(None)                            # xml joint names for gripper
        self._ref_gripper_joint_pos_indexes = self._input2dict(None)            # xml gripper joint position indexes in mjsim
        self._ref_gripper_joint_vel_indexes = self._input2dict(None)            # xml gripper joint velocity indexes in mjsim
        self._ref_joint_gripper_actuator_indexes = self._input2dict(None)       # xml gripper (pos) actuator indexes for robot in mjsim
        self.eef_site_id = self._input2dict(None)                               # xml element id for eef in mjsim
        self.eef_cylinder_id = self._input2dict(None)                           # xml element id for eef cylinder in mjsim
        self.torques = None                                                     # Current torques being applied

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # Flag for loading urdf once (only applicable for IK controllers)
        urdf_loaded = False

        # Load controller configs for both left and right arm
        for arm in self.arms:
            # First, load the default controller if none is specified
            if not self.controller_config[arm]:
                # Need to update default for a single agent
                controller_path = os.path.join(os.path.dirname(__file__), '..',
                                               'controllers/config/{}.json'.format(
                                                   self.robot_model.default_controller_config[arm]))
                self.controller_config[arm] = load_controller_config(custom_fpath=controller_path)

            # Assert that the controller config is a dict file:
            #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            #                                           OSC_POSITION, OSC_POSE, IK_POSE}
            assert type(self.controller_config[arm]) == dict, \
                "Inputted controller config must be a dict! Instead, got type: {}".format(
                    type(self.controller_config[arm]))

            # Add to the controller dict additional relevant params:
            #   the robot name, mujoco sim, eef_name, actuator_range, joint_indexes, timestep (model) freq,
            #   policy (control) freq, and ndim (# joints)
            self.controller_config[arm]["robot_name"] = self.name
            self.controller_config[arm]["sim"] = self.sim
            self.controller_config[arm]["eef_name"] = self.gripper[arm].visualization_sites["grip_site"]
            self.controller_config[arm]["ndim"] = self._joint_split_idx
            self.controller_config[arm]["policy_freq"] = self.control_freq
            (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)
            self.controller_config[arm]["joint_indexes"] = {
                "joints": self.joint_indexes[start:end],
                "qpos": self._ref_joint_pos_indexes[start:end],
                "qvel": self._ref_joint_vel_indexes[start:end]
                                                  }
            self.controller_config[arm]["actuator_range"] = (self.torque_limits[0][start:end],
                                                             self.torque_limits[1][start:end])

            # Only load urdf the first time this controller gets called
            self.controller_config[arm]["load_urdf"] = True if not urdf_loaded else False
            urdf_loaded = True

            # Instantiate the relevant controller
            self.controller[arm] = controller_factory(self.controller_config[arm]["type"], self.controller_config[arm])

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

        # Verify that the loaded model is of the correct type for this robot
        if self.robot_model.arm_type != "bimanual":
            raise TypeError("Error loading robot model: Incompatible arm type specified for this robot. "
                            "Requested model arm type: {}, robot arm type: {}"
                            .format(self.robot_model.arm_type, type(self)))

        # Now, load the gripper if necessary
        for arm in self.arms:
            if self.has_gripper[arm]:
                if self.gripper_type[arm] == 'default':
                    # Load the default gripper from the robot file
                    self.gripper[arm] = gripper_factory(self.robot_model.gripper[arm],
                                                        idn="_".join((str(self.idn), arm)))
                else:
                    # Load user-specified gripper
                    self.gripper[arm] = gripper_factory(self.gripper_type[arm],
                                                        idn="_".join((str(self.idn), arm)))
            else:
                # Load null gripper
                self.gripper[arm] = gripper_factory(None, idn="_".join((str(self.idn), arm)))
            # Use gripper visualization if necessary
            if not self.gripper_visualization[arm]:
                self.gripper[arm].hide_visualization()
            self.robot_model.add_gripper(self.gripper[arm], self.robot_model.eef_name[arm])

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        """
        # First, run the superclass method to reset the position and controller
        super().reset(deterministic)

        if not deterministic:
            # Now, reset the gripper if necessary
            for arm in self.arms:
                if self.has_gripper[arm]:
                    self.sim.data.qpos[
                        self._ref_gripper_joint_pos_indexes[arm]
                    ] = self.gripper[arm].init_qpos

        for arm in self.arms:
            # Update base pos / ori references in controller (technically only needs to be called once)
            self.controller[arm].update_base_pose(self.base_pos, self.base_ori)

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to gripper if necessary
        # indices for grippers in qpos, qvel
        for arm in self.arms:
            if self.has_gripper[arm]:
                self.gripper_joints[arm] = list(self.gripper[arm].joints)
                self._ref_gripper_joint_pos_indexes[arm] = [
                    self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints[arm]
                ]
                self._ref_gripper_joint_vel_indexes[arm] = [
                    self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints[arm]
                ]
                self._ref_joint_gripper_actuator_indexes[arm] = [
                    self.sim.model.actuator_name2id(actuator)
                    for actuator in self.gripper[arm].actuators
                ]

            # IDs of sites for eef visualization
            self.eef_site_id[arm] = self.sim.model.site_name2id(
                self.gripper[arm].visualization_sites["grip_site"])
            self.eef_cylinder_id[arm] = self.sim.model.site_name2id(
                self.gripper[arm].visualization_sites["grip_cylinder"])

    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.robot_model.dof dimensions should be the desired
                normalized joint velocities and if the robot has
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
                NOTE: Assumes inputted actions are of form:
                    [right_arm_control, right_gripper_control, left_arm_control, left_gripper_control]

            policy_step (bool): Whether a new policy step (action) is being taken
        """

        # clip actions into valid range
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        self.torques = np.array([])
        # Now execute actions for each arm
        for arm in self.arms:
            # Make sure to split action space correctly
            (start, end) = (None, self._action_split_idx) if arm == "right" else (self._action_split_idx, None)
            sub_action = action[start:end]

            gripper_action = None
            if self.has_gripper[arm]:
                # get all indexes past controller dimension indexes
                gripper_action = sub_action[self.controller[arm].control_dim:]
                sub_action = sub_action[:self.controller[arm].control_dim]

            # Update the controller goal if this is a new policy step
            if policy_step:
                self.controller[arm].set_goal(sub_action)

            # Now run the controller for a step and add it to the torques
            self.torques = np.concatenate((self.torques, self.controller[arm].run_controller()))

            # Get gripper action, if applicable
            if self.has_gripper[arm]:
                self.grip_action(gripper_action, arm)

        # Clip the torques
        low, high = self.torque_limits
        self.torques = np.clip(self.torques, low, high)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_torq_actuator_indexes] = self.torques

    def grip_action(self, gripper_action, arm):
        """
        Executes gripper @action for specified @arm

        Args:
            gripper_action (float): Value between [-1,1]
            arm (str): "left" or "right"; arm to execute action
        """
        gripper_action_actual = self.gripper[arm].format_action(gripper_action)
        # rescale normalized gripper action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange[self._ref_joint_gripper_actuator_indexes[arm]]
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_gripper_action = bias + weight * gripper_action_actual
        self.sim.data.ctrl[self._ref_joint_gripper_actuator_indexes[arm]] = applied_gripper_action

    def visualize_gripper(self):
        """
        Do any needed visualization here.
        """
        for arm in self.arms:
            if self.gripper_visualization[arm]:
                # By default, color the ball red
                self.sim.model.site_rgba[self.eef_site_id[arm]] = [1., 0., 0., 1.]

    def get_observations(self, di: OrderedDict):
        """
        Returns an OrderedDict containing robot observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """
        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robot_model.naming_prefix

        # proprioceptive features
        di[pf + "joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di[pf + "joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di[pf + "joint_pos"]),
            np.cos(di[pf + "joint_pos"]),
            di[pf + "joint_vel"],
        ]

        for arm in self.arms:
            # Add in eef info
            di[pf + "_{}_".format(arm) + "eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id[arm]])
            di[pf + "_{}_".format(arm) + "eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat(self.robot_model.eef_name[arm]), to="xyzw"
            )
            robot_states.extend([di[pf + "_{}_".format(arm) + "eef_pos"],
                                 di[pf + "_{}_".format(arm) + "eef_quat"]])

            # add in gripper information
            if self.has_gripper[arm]:
                di[pf + "_{}_".format(arm) + "gripper_qpos"] = np.array(
                    [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes[arm]]
                )
                di[pf + "_{}_".format(arm) + "gripper_qvel"] = np.array(
                    [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes[arm]]
                )
                robot_states.extend([di[pf + "_{}_".format(arm) + "gripper_qpos"],
                                     di[pf + "_{}_".format(arm) + "gripper_qvel"]])

        di[pf + "robot-state"] = np.concatenate(robot_states)
        return di

    def _input2dict(self, inp):
        """
        Helper function that converts an input that is either a single value or a list into a dict with keys for
        each arm: "right", "left"
        @inp (str or list): Input value to be converted to dict
        Note: If inp is a list, then assumes format is [right, left]
        """
        # First, convert to list if necessary
        if type(inp) is not list:
            inp = [inp for _ in range(2)]
        # Now, convert list to dict and return
        return {key: value for key, value in zip(self.arms, inp)}

    @property
    def arms(self):
        """
        Returns name of arms used as naming convention throughout this module
        """
        return "right", "left"

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.
        """
        # Action limits based on controller limits
        low, high = [], []
        for arm in self.arms:
            low_g, high_g = ([-1] * self.gripper[arm].dof, [1] * self.gripper[arm].dof) \
                if self.has_gripper[arm] else ([], [])
            low_c, high_c = self.controller[arm].control_limits
            low, high = np.concatenate([low, low_c, low_g]), \
                np.concatenate([high, high_c, high_g])
        return low, high

    @property
    def torque_limits(self):
        """
        Action lower/upper limits per dimension.
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 1]

        return low, high

    @property
    def action_dim(self):
        """
        Action space dimension for this robot (controller dimension + gripper dof)
        """
        dim = 0
        for arm in self.arms:
            dim += self.controller[arm].control_dim + self.gripper[arm].dof if \
                self.has_gripper[arm] else self.controller[arm].control_dim
        return dim

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        # Get the dof of the base robot model
        dof = super().dof
        for arm in self.arms:
            if self.has_gripper[arm]:
                dof += self.gripper[arm].dof
        return dof

    @property
    def _hand_pose(self, arm="right"):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name(self.robot_model.eef_name[arm])

    @property
    def _right_hand_quat(self, arm="right"):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._hand_orn(arm))

    def _hand_total_velocity(self, arm="right"):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """
        # Determine correct start, end points based on arm
        (start, end) = (None, self._joint_split_idx) if arm == "right" else (self._joint_split_idx, None)

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp(self.robot_model.eef_name[arm]).reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[start:end]]

        Jr = self.sim.data.get_body_jacr(self.robot_model.eef_name[arm]).reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[start:end]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _hand_pos(self, arm="right"):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._hand_pose(arm)
        return eef_pose_in_base[:3, 3]

    @property
    def _hand_orn(self, arm="right"):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._hand_pose(arm)
        return eef_pose_in_base[:3, :3]

    @property
    def _hand_vel(self, arm="right"):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity(arm)[:3]

    @property
    def _hand_ang_vel(self, arm="right"):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity(arm)[3:]

    @property
    def _action_split_idx(self):
        """
        Returns the index that correctly splits the right arm from the left arm actions
        NOTE: Assumes inputted actions are of form:
            [right_arm_control, right_gripper_control, left_arm_control, left_gripper_control]
        """
        return self.controller["right"].control_dim + self.gripper["right"].dof if self.has_gripper["right"] \
            else self.controller["right"].control_dim

    @property
    def _joint_split_idx(self):
        """
        Returns the index that correctly splits the right arm from the left arm joints
        """
        return int(len(self.robot_joints) / 2)
