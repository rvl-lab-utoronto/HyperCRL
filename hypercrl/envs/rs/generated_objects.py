import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE

class TwoCube(MujocoGeneratedObject):
    """
    Generates the Pot object with side handles (used in BaxterLift)
    """

    def __init__(
        self,
        name,
        rgba_handle_1=None,
        rgba_handle_2=None,
        solid_handle=True,
        thickness=0.04,  # For body
        density_left=2000,
        density_right=10000,
        density_body=1000,
        rotation=90
    ):
        super().__init__(name)

        self.rotation = rotation
        if self.rotation == 0:
            self.body_half_size = np.array([0.04, 0.08, 0.04])
        else:
            self.body_half_size = np.array([0.08, 0.04, 0.04])
        self.thickness = thickness 

        if rgba_handle_1:
            self.rgba_handle_1 = np.array(rgba_handle_1)
        else:
            self.rgba_handle_1 = BLUE # deisnty left
        if rgba_handle_2:
            self.rgba_handle_2 = np.array(rgba_handle_2)
        else:
            self.rgba_handle_2 = GREEN # density right
        self.solid_handle = solid_handle

        self.density_left=density_left
        self.density_right=density_right
        self.density_body=density_body

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    def get_horizontal_radius(self):
        l = w = 0.08
        return np.sqrt(l**2 + w**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        if self.rotation == 0:
            pos1 = [self.body_half_size[0], 0, 0]
            pos2 = [-self.body_half_size[0], 0, 0]
        else:
            pos1 = [0, self.body_half_size[1], 0]
            pos2 = [0, -self.body_half_size[1], 0]
        handle_1 = new_body(name="handle_1")
        handle_1.append(
            new_geom(
                geom_type="box",
                name="handle_1",
                pos=pos1,
                size=self.body_half_size,
                rgba=self.rgba_handle_1,
                group=1,
                density=str(self.density_left),
            )
        )

        handle_2 = new_body(name="handle_2")
        handle_2.append(
            new_geom(
                geom_type="box",
                name="handle_2",
                pos=pos2,
                size=self.body_half_size,
                rgba=self.rgba_handle_2,
                group=1,
                density=str(self.density_right),
            )
        )

        main_body.append(handle_1)
        main_body.append(handle_2)
        main_body.append(new_site(name="pot_center", pos=[0, 0, 0.02], rgba=[1, 0, 0, 0]))

        if self.rotation == 0:
            handle_1.append(new_site(name="corner_1", pos=[-self.body_half_size[0] * 2, -self.body_half_size[1], 0], rgba=[1, 0, 0, 0]))
            handle_1.append(new_site(name="corner_2", pos=[-self.body_half_size[0] * 2, self.body_half_size[1], 0],  rgba=[1, 0, 0, 0]))
            handle_2.append(new_site(name="corner_3", pos=[self.body_half_size[0] * 2,  -self.body_half_size[1], 0], rgba=[1, 0, 0, 0]))
            handle_2.append(new_site(name="corner_4", pos=[self.body_half_size[0] * 2,  self.body_half_size[1], 0],  rgba=[1, 0, 0, 0]))

            handle_1.append(new_site(name="pot_left", pos=[self.body_half_size[0] * 2, 0, 0.02], rgba=[1, 0, 0, 0]))
            handle_2.append(new_site(name="pot_right", pos=[-self.body_half_size[0] * 2, 0, 0.02], rgba=[1, 0, 0, 0]))
        else:
            handle_1.append(new_site(name="corner_1", pos=[-self.body_half_size[0], -self.body_half_size[1] * 2, 0], rgba=[1, 0, 0, 0]))
            handle_1.append(new_site(name="corner_3", pos=[self.body_half_size[0], -self.body_half_size[1] * 2, 0],  rgba=[1, 0, 0, 0]))
            handle_2.append(new_site(name="corner_2", pos=[-self.body_half_size[0], self.body_half_size[1] * 2, 0], rgba=[1, 0, 0, 0]))
            handle_2.append(new_site(name="corner_4", pos=[self.body_half_size[0], self.body_half_size[1] * 2, 0],  rgba=[1, 0, 0, 0]))

            handle_1.append(new_site(name="pot_left", pos=[self.body_half_size[0], 0., 0.02], rgba=[1, 0, 0, 0]))
            handle_2.append(new_site(name="pot_right", pos=[-self.body_half_size[0], 0., 0.02], rgba=[1, 0, 0, 0]))

        return main_body

    def handle_geoms(self):
        return self.handle_1_geoms() + self.handle_2_geoms()

    def handle_1_geoms(self):
        if self.solid_handle:
            return ["handle_1"]
        return ["handle_1_c", "handle_1_+", "handle_1_-"]

    def handle_2_geoms(self):
        if self.solid_handle:
            return ["handle_2"]
        return ["handle_2_c", "handle_2_+", "handle_2_-"]

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class TwoCubeV2(TwoCube):
    """
    Used in rotation task with two cubes. Each cube is an instance of this class.
    This two colors can be used to indicate orientation more easily
    """
    def __init__(
        self,
        name,
        rgba_handle_1=None,
        rgba_handle_2=None,
        solid_handle=True,
        thickness=0.03,  # For body
        density_left=2000,
        density_right=10000,
        density_body=1000,
        rotation=90,
        friction="1 0.005 0.0001",
    ):
        super().__init__(name, rgba_handle_1, rgba_handle_2, solid_handle, thickness,
            density_left, density_right, density_body, rotation)
        

        self.friction = friction
        self.body_half_size = np.array([0.03, 0.03, thickness])
        self.body_bulk_size = np.array([0.03, 0.025, thickness])
        self.body_tip_size = np.array([0.03, 0.005, thickness])

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is None:
            name = self.name
     
        main_body.set("name", name) 
 
        pos1 = [0, 0.025, 0]
        pos2 = [0, -0.005, 0]

        handle_1 = new_body(name=name + "_handle_1")
        handle_1.append(
            new_geom(
                geom_type="box",
                name=name + "_handle_1",
                pos=pos1,
                size=self.body_tip_size,
                rgba=self.rgba_handle_1,
                group=1,
                density=str(self.density_left),
                friction=self.friction,
            )
        )

        handle_2 = new_body(name= name + "_handle_2")
        handle_2.append(
            new_geom(
                geom_type="box",
                name=name + "_handle_2",
                pos=pos2,
                size=self.body_bulk_size,
                rgba=self.rgba_handle_2,
                group=1,
                density=str(self.density_right),
                friction=self.friction,
            )
        )

        main_body.append(handle_1)
        main_body.append(handle_2)
        main_body.append(new_site(name= name + "_pot_center", pos=[0, 0, 0.02], rgba=[1, 0, 0, 0]))

        main_body.append(new_site(name=name + "_corner_1", pos=[-self.body_half_size[0], -self.body_half_size[1], 0], rgba=[1, 0, 0, 0]))
        main_body.append(new_site(name=name + "_corner_2", pos=[-self.body_half_size[0], self.body_half_size[1], 0],  rgba=[1, 0, 0, 0]))
        main_body.append(new_site(name=name + "_corner_3", pos=[self.body_half_size[0], -self.body_half_size[1], 0], rgba=[1, 0, 0, 0]))
        main_body.append(new_site(name=name + "_corner_4", pos=[self.body_half_size[0], self.body_half_size[1], 0],  rgba=[1, 0, 0, 0]))

        return main_body

def five_sided_box(size, rgba, group, thickness,density=None):
    """
    Args:
        size ([float,flat,float]):
        rgba ([float,float,float,float]): color
        group (int): Mujoco group
        thickness (float): wall thickness

    Returns:
        []: array of geoms corresponding to the
            5 sides of the pot used in BaxterLift
    """
    geoms = []
    x, y, z = size
    r = thickness / 2
    geoms.append(
        new_geom(
            geom_type="box", size=[x, y, r], pos=[0, 0, -z + r], rgba=rgba, group=group,density=density
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, -y + r, 0], rgba=rgba, group=group,density=density
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, y - r, 0], rgba=rgba, group=group,density=density
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[x - r, 0, 0], rgba=rgba, group=group,density=density
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[-x + r, 0, 0], rgba=rgba, group=group,density=density
        )
    )
    return geoms


DEFAULT_DENSITY_RANGE = [200, 500, 1000, 3000, 5000]
DEFAULT_FRICTION_RANGE = [0.25, 0.5, 1, 1.5, 2]

def _get_size(size,
              size_max,
              size_min,
              default_max,
              default_min):
    """
        Helper method for providing a size,
        or a range to randomize from
    """
    if len(default_max) != len(default_min):
        raise ValueError('default_max = {} and default_min = {}'
                         .format(str(default_max), str(default_min)) +
                         ' have different lengths')
    if size is not None:
        if (size_max is not None) or (size_min is not None):
            raise ValueError('size = {} overrides size_max = {}, size_min = {}'
                             .format(size, size_max, size_min))
    else:
        if size_max is None:
            size_max = default_max
        if size_min is None:
            size_min = default_min
        size = np.array([np.random.uniform(size_min[i], size_max[i])
                         for i in range(len(default_max))])
    return size


def _get_randomized_range(val,
                          provided_range,
                          default_range):
    """
        Helper to initialize by either value or a range
        Returns a range to randomize from
    """
    if val is None:
        if provided_range is None:
            return default_range
        else:
            return provided_range
    else:
        if provided_range is not None:
            raise ValueError('Value {} overrides range {}'
                             .format(str(val), str(provided_range)))
        return [val]

class PotWithHandlesObject(MujocoGeneratedObject):
    """
    A randomized cylinder object.
    """

    def __init__(
        self,
        name,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        handle_radius=0.01,
        handle_length=0.04,
        handle_width=0.04,
        rgba_body=None,
        solid_handle=False,
        thickness=0.025,  # For body
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.05, 0.05],
                         [0.05, 0.05])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)

        self.body_half_size = np.array([0.04, 0.04, 0.04])
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        if rgba_body:
            self.rgba_body = np.array(rgba_body)
        else:
            self.rgba_body = RED
        self.rgba_handle = GREEN
        self.solid_handle = solid_handle

        super().__init__(
            name,
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction_range=friction_range,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "cylinder size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        main_body =  self._get_collision(name=name, site=site, ob_type="cylinder")

        handle_z = self.body_half_size[2] - self.handle_radius
        handle_z -= 0.03
        handle_center = [
            0,
            -1 * (self.body_half_size[1] + self.handle_length),
            handle_z,
        ]
        # the bar on handle horizontal to body
        main_bar_size = [
            self.handle_width / 2 + self.handle_radius,
            self.handle_radius,
            self.handle_radius,
        ]
        side_bar_size = [self.handle_radius, self.handle_length / 2, self.handle_radius]

        handle = new_body(name="handle")
        handle.append(
            new_geom(
                geom_type="box",
                name="handle_c",
                pos=handle_center,
                size=main_bar_size,
                rgba=self.rgba_handle,
                group=1,
            )
        )
        handle.append(
            new_geom(
                geom_type="box",
                name="handle_+",  # + for positive x
                pos=[
                    self.handle_width / 2,
                    -self.body_half_size[1] - self.handle_length / 2,
                    handle_z,
                ],
                size=side_bar_size,
                rgba=self.rgba_handle,
                group=1,
            )
        )
        handle.append(
            new_geom(
                geom_type="box",
                name="handle_-",
                pos=[
                    -self.handle_width / 2,
                    -self.body_half_size[1] - self.handle_length / 2,
                    handle_z,
                ],
                size=side_bar_size,
                rgba=self.rgba_handle,
                group=1,
            )
        )

        main_body.append(handle)
        main_body.append(
            new_site(
                name="pot_handle",
                rgba=self.rgba_handle,
                pos=handle_center - np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(new_site(name="pot_center", pos=[0, 0, 0], rgba=[1, 0, 0, 0]))

        return main_body

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="cylinder")