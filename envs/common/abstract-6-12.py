import copy
import os
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.utils import seeding
import numpy as np

from highway_env import utils
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.risk_field import initialize_plot, Risk_field
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.mpc_controller2 import MPC
import casadi as ca
import casadi.tools as ca_tools
Observation = TypeVar("Observation")


class AbstractEnv(gym.Env):

    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    observation_type: ObservationType
    action_type: ActionType
    _record_video_wrapper: Optional[RecordVideo]
    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }

    PERCEPTION_DISTANCE = 5.0 * Vehicle.MAX_SPEED
    """The maximum distance of any vehicle present in the observation [m]"""

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.enable_auto_render = False
        self.reset()
        self.mpc_controller = MPC()
        # self.fig, self.ax = initialize_plot()
        self.risk_field = Risk_field()
        self.x0 = np.array([self.vehicle.position[0],self.vehicle.position[1],self.vehicle.speed,self.vehicle.heading]).reshape(-1, 1)
        self.u0 = np.array([0, 0]*self.mpc_controller.N).reshape(-1, 2)

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            # "screen_width": 600,  # [px]
            # "screen_height": 150,  # [px]
            "screen_width": 1800,  # [px]
            "screen_height": 600,  # [px]
            # "centering_position": [0.3, 0.5],
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = self.config["simulation_frequency"] \
            if self._record_video_wrapper else self.config["policy_frequency"]
        self.metadata['render_fps'] = video_real_time_ratio * frames_freq

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Returns a multi-objective vector of rewards.

        If implemented, this reward vector should be aggregated into a scalar in _reward().
        This vector value should only be returned inside the info dict.

        :param action: the last action performed
        :return: a dict of {'reward_name': reward_value}
        """
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        raise NotImplementedError

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "crashed": self.vehicle.crashed,
            "action": action,
            # "cost": 5 * float(self.vehicle.crashed) +
            #         0.05 * np.clip(utils.lmap(self.vehicle.speed * np.cos(self.vehicle.heading), self.config["cost_speed_range"], [0, 1]), 0, 1)
            # "cost": 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road)
            #         + self._compute_headway_cost_ego(self.vehicle),
            "cost": 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road)
                    + self._cost(),
            # "cost": self._cost(),
            "time": self.time,
            "position": self.vehicle.position,
            "speed": self.vehicle.speed,
            "heading": self.vehicle.heading,
            "acceleration": self.vehicle.action['acceleration'],
            "steering": self.vehicle.action['steering'],
        }
        try:
            info["rewards"] = self._rewards(action),
            # info["follow_speed"] = self._follow_car_speed(),
        except NotImplementedError:
            pass
        return info

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration
        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == 'human':
            self.render()
        return obs, info

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.config['usempc_controller']:
            xs = action
            y_ref = utils.lmap(xs[0], [-1, 1], [3,9])
            v_ref = utils.lmap(xs[1], [-1, 1], [10, 30])
            xs = np.array([y_ref, v_ref]).reshape(-1, 1)
            # print(xs)
            init_control = ca.reshape(self.u0, -1, 1)
            c_p = np.concatenate((self.x0, xs))
            u_sol, u_attach, f = self.mpc_controller.sovler_mpc(init_control, c_p)
            action = np.array(u_attach)
            x0 = np.array([self.vehicle.position[0], self.vehicle.position[1],
                           self.vehicle.speed, self.vehicle.heading]).reshape(-1, 1)
            self.x0 = ca.reshape(x0, -1, 1)
            self.u0 = ca.vertcat(u_sol[1:, :], u_sol[-1, :])
            action = action.flatten()  # 这一步非常重要
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        obs = self.observation_type.observe()
        reward = self._reward(action)
        cost = self._cost()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()
        done = terminated or truncated
        U_field, vehicles_obs, X, Y = self.plot_cost()
        # self.risk_field.update_plot(self.fig, self.ax, X, Y, U_field, vehicles_obs, done)
        return obs, reward, terminated, truncated, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        # print(frames)
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self) -> List[int]:
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.
        If a RecordVideo wrapper has been set, use it to capture intermediate frames.
        """
        if self.viewer is not None and self.enable_auto_render:

            if self._record_video_wrapper and self._record_video_wrapper.video_recorder:
                self._record_video_wrapper.video_recorder.capture_frame()
            else:
                self.render()

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        """
        Change the type of all vehicles on the road

        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behavior(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def _compute_headway_cost_ego(self, vehicle):
        ego_lane_index = vehicle.lane_index  #自车所在车道线
        ego_v = vehicle.speed  #自车车速
        ego_p = vehicle.position[0]  #自车纵向位置
        f_d = []
        f_v = []
        f_p = []
        for car in self.road.vehicles:
            if car.lane_index == ego_lane_index:
                d = car.position[0] - vehicle.position[0]
                if d > 0:
                    f_d.append(d)
                    f_v.append(car.speed)
                    f_p.append(car.position[0])
        if len(f_d) == 0:
            headway_cost = 0
        else:
            min_f_d = min(f_d)
            Headway_ego = abs(min_f_d) / ego_v
            if 0 <= Headway_ego <= 1:
                headway_cost = -np.log(Headway_ego)
            else:
                headway_cost = 0
        return headway_cost

    def _compute_headway_distance(self, vehicle):
        ego_lane_index = vehicle.lane_index
        ego_v = vehicle.speed
        ego_p = vehicle.position[0]
        f_d = []
        f_v = []
        f_p = []
        r_d = []
        r_v = []
        r_p = []
        for car in self.road.vehicles:
            if self.config["target_lane"] in car.lane_index:
                d = car.position[0] - vehicle.position[0]
                if d > 0:
                    f_d.append(d)
                    f_v.append(car.speed)
                    f_p.append(car.position[0])
                if d < 0:
                    r_d.append(d)
                    r_v.append(car.speed)
                    r_p.append(car.position[0])

        if len(f_d) >= 2 and len(r_d) >= 2: #自车前后大于2辆车
            min_f_d = min(f_d)
            min_index = f_d.index(min_f_d)
            max_r_d = max(r_d)
            max_index = r_d.index(max_r_d)
            Headway_r = abs(max_r_d) / r_v[max_index]
            Headway_ego = abs(min_f_d) / ego_v
            Headway_cost = min(abs(Headway_r), abs(Headway_ego))
            Norm_Headway_cost = np.tanh(5 * (Headway_cost - 0.5))
            leader1_v, leader2_v = f_v[min_index], f_v[min_index + 1]
            Target_v = (leader1_v + leader2_v) / 2

        if len(f_d) < 2 and len(r_d) >= 2: #自车前小于2辆车，自车后大于2辆车
            max_r_d = max(r_d)
            max_index = r_d.index(max_r_d)
            Headway_r = abs(max_r_d) / r_v[max_index]
            if len(f_d) == 1: # 自车前1辆车
                min_f_d = min(f_d)
                min_index = f_d.index(min_f_d)
                leader1_v = f_v[min_index]
                Headway_ego = abs(min_f_d) / ego_v
                Target_v = leader1_v
            if len(f_d) == 0: # 自车前0辆车
                Target_v = 25
                Headway_ego = 10
            Headway_cost = min(abs(Headway_r), abs(Headway_ego))
            Norm_Headway_cost = np.tanh(5 * (Headway_cost - 0.5))

        if len(f_d) >= 2 and len(r_d) < 2: #自车前大于2辆车，自车后小于2辆车
            min_f_d = min(f_d)
            min_index = f_d.index(min_f_d)
            Headway_ego = abs(min_f_d) / ego_v
            if len(r_d) == 1:
                max_r_d = max(r_d)
                max_index = r_d.index(max_r_d)
                Headway_r = abs(max_r_d) / r_v[max_index]
            if len(r_d) == 0:
                Headway_r = 10
            Headway_cost = min(abs(Headway_r), abs(Headway_ego))
            Norm_Headway_cost = np.tanh(5 * (Headway_cost - 0.5))
            leader1_v, leader2_v = f_v[min_index], f_v[min_index + 1]
            Target_v = (leader1_v + leader2_v) / 2

        if len(f_d) < 2 and len(r_d) < 2:  # 自车前小于2辆车，自车后小于2辆车
            if len(f_d) == 1 and len(r_d) == 1:
                min_f_d = min(f_d)
                min_index = f_d.index(min_f_d)
                max_r_d = max(r_d)
                max_index = r_d.index(max_r_d)
                Headway_r = abs(max_r_d) / r_v[max_index]
                Headway_ego = abs(min_f_d) / ego_v
                Headway_cost = min(abs(Headway_r), abs(Headway_ego))
                Norm_Headway_cost = np.tanh(5 * (Headway_cost - 0.5))
                leader1_v = f_v[min_index]
                Target_v = leader1_v
            if len(f_d) == 0 and len(r_d) == 1:
                Target_v = 25
                Headway_ego = 10

            if len(f_d) == 1 and len(r_d) == 0:
                min_f_d = min(f_d)
                min_index = f_d.index(min_f_d)
                Headway_r = 10
                Headway_ego = abs(min_f_d) / ego_v
                Headway_cost = min(abs(Headway_r), abs(Headway_ego))
                Norm_Headway_cost = np.tanh(5 * (Headway_cost - 0.5))
                leader1_v = f_v[min_index]
                Target_v = leader1_v

            if len(f_d) == 0 and len(f_d) == 0:
                Norm_Headway_cost = 1
                Target_v = 25
        return Norm_Headway_cost, Target_v

    def calculate_target_velocity(self, V_self, P_self, V_front, P_front):
        Headway_ego = (P_front - P_self) / V_self
        if Headway_ego < 1.2:
            # 计算安全舒适的减速度（可以根据具体情况调整）
            comfortable_deceleration = 2.0  # 2.0 m/s^2 作为示例

            # 计算目标车速，使自车按照安全舒适的减速度减速
            target_car_speed = V_self - comfortable_deceleration * Headway_ego
        else:
            # 如果Headway_ego不小于1.2，保持原速度或选择其他策略
            target_car_speed = max(V_self, V_front)
        return target_car_speed

    def _follow_car_speed(self):
        ego_lane_index = self.vehicle.lane_index
        ego_v = self.vehicle.speed
        ego_p = self.vehicle.position[0]
        r_d = []
        r_v = []
        r_p = []
        for car in self.road.vehicles:
            if self.config["target_lane"] in car.lane_index:
                d = car.position[0] - self.vehicle.position[0]
                if d <= 0:
                    r_d.append(d)
                    r_v.append(car.speed)
                    r_p.append(car.position[0])
                max_r_d = max(r_d)
                max_index = r_d.index(max_r_d)
        f_car_speed = r_v[max_index]
        # print(f_car_speed)
        return f_car_speed


    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', '_record_video_wrapper']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def plot_cost(self):
        Asta = 1
        Adyn = 1
        Ab = 1  # Field intensity coefficient for the road boundary
        Al = 0.1  # Field intensity coefficient for the lane marking
        σb = 1  # Risk distribution range for the road boundary
        σl = 1  # Risk distribution range for the lane marking
        kx = 1
        ky = 0.8
        kv = 2
        alpha = 0.9
        beta = 2
        L_obs = 5
        W_obs = 2
        positions = [vehicle.position for vehicle in self.road.vehicles]
        speeds = [vehicle.speed for vehicle in self.road.vehicles]
        headings = [vehicle.heading for vehicle in self.road.vehicles]
        vehicles_obs = np.column_stack((np.array(positions), np.array(speeds), np.array(headings)))
        v_ego = vehicles_obs[0, 2]  # 保持v_ego不变
        x_start = vehicles_obs[0, 0]
        # 设置x, y的范围
        x = np.linspace(x_start-100, x_start+100, 400)
        # x = np.linspace(200, 400, 400)
        y = np.linspace(-2, 14, 32)
        x_ego, y_ego = np.meshgrid(x, y)

        # Road boundary positions
        y_boundary = [-2, 14]

        # Lane marking positions
        y_lane = [2, 6, 10]

        # Potential field for road boundaries
        Eb = np.zeros_like(x_ego)
        for boundary in y_boundary:
            Eb += Ab * np.exp(-((y_ego - boundary) ** 2) / (2 * σb ** 2))
        Eb = np.where((y_ego >= 12) | (y_ego <= 0), Eb, 0)

        # Potential field for lane markings
        El = np.zeros_like(x_ego)
        for lane in y_lane:
            El += Al * np.exp(-((y_ego - lane) ** 2) / (2 * σl ** 2))
        El = np.where((y_ego > 0) & (y_ego < 12), El, 0)

        # 初始化场强矩阵
        U_field = np.zeros_like(x_ego)
        for vehicle in vehicles_obs[1:]:
            x_obs, y_obs, v_obs, h_obs = vehicle
            σx = kx * L_obs
            σy = ky * W_obs
            σv = kv * np.abs(v_obs - v_ego)
            # 计算相对速度方向
            if v_obs >= v_ego:
                relv = 1
            else:
                relv = -1

            # 修正坐标旋转公式
            h_obs = -h_obs  # 这一步非常重要，不然有航向角的障碍物的势场就反了
            x_rel = (x_ego - x_obs) * np.cos(h_obs) - (y_ego - y_obs) * np.sin(h_obs) + x_obs
            y_rel = (x_ego - x_obs) * np.sin(h_obs) + (y_ego - y_obs) * np.cos(h_obs) + y_obs

            Usta = Asta * np.exp(-((x_rel - x_obs) ** 2 / σx ** 2) ** beta - ((y_rel - y_obs) ** 2 / σy ** 2) ** beta)
            Udyn = Adyn * (np.exp(-((x_rel - x_obs) ** 2 / σv ** 2 + (y_rel - y_obs) ** 2 / σy ** 2)) /
                    (1 + np.exp(-relv * (x_rel - x_obs - alpha * L_obs * relv))))

            U_field += Usta + Udyn
        U_field1 = U_field + Eb + El
        return U_field1, vehicles_obs,  x_ego, y_ego

class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = info["agents_rewards"]
        terminated = info["agents_terminated"]
        truncated = info["agents_truncated"]
        return obs, reward, terminated, truncated, info
