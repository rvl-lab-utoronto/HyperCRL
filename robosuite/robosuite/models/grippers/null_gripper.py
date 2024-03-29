"""
Null Gripper (if we don't want to attach gripper to robot eef).
"""
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class NullGripper(GripperModel):
    """
    Dummy Gripper class to represent no gripper
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/null_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 0

    @property
    def init_qpos(self):
        return None

    @property
    def _joints(self):
        return []

    @property
    def _actuators(self):
        return []

    @property
    def _contact_geoms(self):
        return []

    @property
    def _left_finger_geoms(self):
        return []

    @property
    def _right_finger_geoms(self):
        return []
