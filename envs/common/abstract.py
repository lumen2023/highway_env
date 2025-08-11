import copy
import os
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.utils import seeding
import numpy as np
from collections import deque

from highway_env import utils
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.risk_field import initialize_plot, Risk_field
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.controller import LYZ_MDPVehicle
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
        print("\n！！！！！！！！！！！！！！外部库！！！！！！！！！！！！！！！！\n")
        # Configuration
        self.config = self.default_config()
        self.configure(config)
        self.idm_flag =  False

        # 历史记录
        history_len = self.config["history_len"]
        # 保存过去5帧的状态信息
        self.history = deque(maxlen=history_len) # LYZ-history
        self.action_idm = None
        self.is_first_step = True  # 标记是否是第一个时间步
        self.time = 0

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
        self.fig = None
        self.ax = None
        self.mpc_controller = MPC()
        # self.fig, self.ax = initialize_plot()
        self.risk_field = Risk_field()
        self.x0 = np.array([self.vehicle.position[0],self.vehicle.position[1],self.vehicle.speed,self.vehicle.heading]).reshape(-1, 1)
        self.u0 = np.array([0, 0]*self.mpc_controller.N).reshape(-1, 2)
        self.previous_steering_angle = 0

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
            "policy_frequency": 1,  # [Hz]IDVehicle
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "other_vehicles_type1": "highway_env.vehicle.behavior.AggressiveVehicle", # 1是激进风格
            "other_vehicles_type2": "highway_env.vehicle.behavior.DefensiveVehicle", # 2是 passive风格
            "other_vehicles_type3": "highway_env.vehicle.behavior.IDVehicle", # 3是 无避让风格
            # "screen_width": 600,  # [px]
            # "screen_height": 150,  # [px]
            "screen_width": 1800,  # [px]
            "screen_height": 600,  # [px]
            # "centering_position": [0.3, 0.5],
            "centering_position": [0.6, 0.5],
            "scaling": 5.5,
            "usempc_controller": False,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
            "use_history": False,
            # "use_history": True,
            "history_len": 15,
            "num_samples": 3
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

        # 获取单帧 observation space
        single_frame_obs_space = self.observation_type.space()
        single_frame_size = np.prod(single_frame_obs_space.shape)  # 计算单帧观测的总大小 (30,)

        if self.config["use_history"]:
            history_len = self.config["num_samples"]  # 需要的历史帧数 (3)

            # 适配历史观测的空间，使其变成 (90,)
            self.observation_space = gym.spaces.Box(
                low=np.tile(single_frame_obs_space.low, history_len),  # 低值扩展到 (90,)
                high=np.tile(single_frame_obs_space.high, history_len),  # 高值扩展到 (90,)
                dtype=single_frame_obs_space.dtype
            )
        else:
            self.observation_space = single_frame_obs_space  # 仍然是 (30,)

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
            "on_road": self.vehicle.on_road,
            "action": action,
            "action_idm": self.action_idm,
            # "cost": 5 * float(self.vehicle.crashed) +
            #         0.05 * np.clip(utils.lmap(self.vehicle.speed * np.cos(self.vehicle.heading), self.config["cost_speed_range"], [0, 1]), 0, 1)
            # "cost": 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road)
            #         + self._compute_headway_cost_ego(self.vehicle),
            # "cost": 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road)
            #         + min(self._cost(),3),
            "cost": 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road)
                    + min(self._cost(), 3),
            "time": self.time,
            "position": self.vehicle.position,
            "speed": self.vehicle.speed,
            "heading": self.vehicle.heading,
            "acceleration": self.vehicle.action['acceleration'],
            "steering": self.vehicle.action['steering'],
        }
        # if info["cost"] > 1:
        #     print("***\n","风险场代价self._cost() : ",self._cost(),"***\n")
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
        self.time = 0
        self.is_first_step = True  # 重新开始回合时，设置为第一步

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
        if self.config['use_history']:
            # 清空历史记录
            self.history.clear()
            for _ in range(self.config['history_len']): # LYZ-history重复15次
                self.history.append(obs)
            # print(" History !!! ")
            # 如果已经保存了5帧数据，可以将它们作为当前的obs
            # obs = np.array(list(self.history))
            indices = np.linspace(0, self.config["history_len"] - 1, self.config["num_samples"], dtype=int)
            obs = np.array([self.history[i] for i in indices]).ravel()  # 形状 (90,)
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
            y_ref = utils.lmap(xs[0], [-1, 1], [-2,14])
            v_ref = utils.lmap(xs[1], [-1, 1], [15, 35])
            xs = np.array([y_ref, v_ref]).reshape(-1, 1)
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

        # print("idm_flag: ", self.idm_flag)
        ## IDM ##
        steering = self.steering_control()
        self.action_idm = [self.a_ego_idm, self.s_ego_mobil]
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        obs = self.observation_type.observe()
        reward = self._reward(action)
        # cost = self._cost()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        # 为了符合 is_first 的设计，首次调用时是 True，后续都为 False
        info["is_first"] = self.is_first_step
        # 第一次 step 后将 is_first 设置为 False
        if self.is_first_step:
            self.is_first_step = False

        if self.render_mode == 'human':
            self.render()
        done = terminated or truncated
        # 标记当前回合已开始
        done = terminated or truncated

        # LYZ_history
        if self.config["use_history"]:
            # 将当前帧添加到历史记录中
            self.history.append(obs)
            # # 如果已经保存了5帧数据，可以将它们作为当前的obs
            # obs = np.array(list(self.history)) # 将队列转化为单列列表输出
            num_samples = self.config["num_samples"]
            len = self.config["history_len"]
            indices = np.linspace(0, len - 1, num_samples, dtype=int)
            obs = np.array([self.history[i] for i in indices]).ravel()
            # obs = obs.reshape(-1, )

        # U_field, vehicles_obs, X, Y = self.plot_cost()
        # self.risk_field.update_plot(self.fig, self.ax, X, Y, U_field, vehicles_obs, done, self.time)
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

    # def plot_cost(self):
    #     Asta = 1
    #     Adyn = 1
    #     kx = 1
    #     ky = 0.8
    #     kv = 2
    #     alpha = 0.9
    #     beta = 2
    #     L_obs = 5
    #     W_obs = 2
    #     positions = [vehicle.position for vehicle in self.road.vehicles]
    #     speeds = [vehicle.speed for vehicle in self.road.vehicles]
    #     headings = [vehicle.heading for vehicle in self.road.vehicles]
    #     vehicles_obs = np.column_stack((np.array(positions), np.array(speeds), np.array(headings)))
    #     v_ego = vehicles_obs[0, 2]  # 保持v_ego不变
    #     x_start = vehicles_obs[0, 0]
    #     # 设置x, y的范围
    #     x = np.linspace(x_start-100, x_start+100, 400)
    #     y = np.linspace(-2, 14, 32)
    #     x_ego, y_ego = np.meshgrid(x, y)
    #
    #     # 初始化场强矩阵
    #     U_field = np.zeros_like(x_ego)
    #
    #     for vehicle in vehicles_obs[1:]:
    #         x_obs, y_obs, v_obs, h_obs = vehicle
    #         σx = kx * L_obs
    #         σy = ky * W_obs
    #         σv = kv * np.abs(v_obs - v_ego)
    #         # 计算相对速度方向
    #         if v_obs >= v_ego:
    #             relv = 1
    #         else:
    #             relv = -1
    #
    #         # # 修正坐标旋转公式
    #         # x_rel = (x_ego - x_obs) * np.cos(h_obs) + (y_ego - y_obs) * np.sin(h_obs) + x_obs
    #         # y_rel = -(x_ego - x_obs) * np.sin(h_obs) + (y_ego - y_obs) * np.cos(h_obs) + y_obs
    #
    #         # 修正坐标旋转公式
    #         h_obs = -h_obs  # 这一步非常重要，不然有航向角的障碍物的势场就反了
    #         x_rel = (x_ego - x_obs) * np.cos(h_obs) - (y_ego - y_obs) * np.sin(h_obs) + x_obs
    #         y_rel = (x_ego - x_obs) * np.sin(h_obs) + (y_ego - y_obs) * np.cos(h_obs) + y_obs
    #
    #         Usta = Asta * np.exp(-((x_rel - x_obs) ** 2 / σx ** 2) ** beta - ((y_rel - y_obs) ** 2 / σy ** 2) ** beta)
    #
    #         Udyn = Adyn * (np.exp(-((x_rel - x_obs) ** 2 / σv ** 2 + (y_rel - y_obs) ** 2 / σy ** 2)) /
    #                 (1 + np.exp(-relv * (x_rel - x_obs - alpha * L_obs * relv))))
    #
    #         U_field += Usta + Udyn
    #
    #     return U_field, vehicles_obs,  x_ego, y_ego

    def _cost(self):
        """
        计算车辆行驶过程中的成本函数，该成本函数基于潜在场方法，用于路径规划和避障。

        成本函数考虑了静态障碍物（其他车辆）、道路边界和车道线的潜在场影响，
        以及车辆之间的动态相互作用，以计算总的成本（潜在能量）。

        Returns:
            float: 车辆行驶过程中的总成本（潜在能量）。
        """
        # 初始化潜在场强度系数
        Asta = 1
        Adyn = 1
        Ab = 1  # 道路边界的场强系数
        Al = 0.1  # 车道线的场强系数
        σb = 1  # 道路边界的风风险分布范围
        σl = 1  # 车道线的风险分布范围
        kx = 1
        ky = 0.8
        kv = 2
        alpha = 0.9
        beta = 2
        L_obs = 5
        W_obs = 2

        # 获取所有车辆的位置、速度和航向
        positions = [vehicle.position for vehicle in self.road.vehicles]
        speeds = [vehicle.speed for vehicle in self.road.vehicles]
        headings = [vehicle.heading for vehicle in self.road.vehicles]
        vehicles_obs = np.column_stack((np.array(positions), np.array(speeds), np.array(headings)))
        x_ego, y_ego, v_ego, _ = vehicles_obs[0]

        # 道路边界位置
        y_boundary = [-2, 14]
        # 车道线位置
        y_lane = [2, 6, 10]

        # 计算道路边界的潜在场
        Eb = np.zeros_like(x_ego)
        for boundary in y_boundary:
            Eb += Ab * np.exp(-((y_ego - boundary) ** 2) / (1 * σb ** 2))

        # 计算车道线的潜在场
        El = np.zeros_like(x_ego)
        for lane in y_lane:
            El += Al * np.exp(-((y_ego - lane) ** 2) / (1 * σl ** 2))

        Ustas = []
        Udyns = []
        # 计算与其他车辆的相互作用势场
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

            Usta = Asta * np.exp(
                -((x_rel - x_obs) ** 2 / σx ** 2) ** beta - ((y_rel - y_obs) ** 2 / σy ** 2) ** beta)
            Udyn = Adyn * (np.exp(-((x_rel - x_obs) ** 2 / σv ** 2 + (y_rel - y_obs) ** 2 / σy ** 2)) /
                           (1 + np.exp(-relv * (x_rel - x_obs - alpha * L_obs * relv))))
            Ustas.append(Usta)
            Udyns.append(Udyn)

        # 计算总的成本（潜在能量）
        Ustas = np.array(Ustas).sum()
        Udyns = np.array(Udyns).sum()
        Ebs = Eb.sum()
        Els = El.sum()
        U = Ustas + Udyns + Ebs + Els
        # U = Ustas + Udyns
        return U


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
        y = np.linspace(-2, 14, 32)
        x_ego, y_ego = np.meshgrid(x, y)

        # Road boundary positions
        y_boundary = [-2, 14]
        # Lane marking positions
        y_lane = [2, 6, 10]
        # Potential field for road boundaries
        Eb = np.zeros_like(x_ego)
        for boundary in y_boundary:
            Eb += Ab * np.exp(-((y_ego - boundary) ** 2) / (1 * σb ** 2))
        # Potential field for lane markings
        El = np.zeros_like(x_ego)
        for lane in y_lane:
            El += Al * np.exp(-((y_ego - lane) ** 2) / (1 * σl ** 2))
        # 初始化动态障碍物场强矩阵
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

            # # 修正坐标旋转公式
            # x_rel = (x_ego - x_obs) * np.cos(h_obs) + (y_ego - y_obs) * np.sin(h_obs) + x_obs
            # y_rel = -(x_ego - x_obs) * np.sin(h_obs) + (y_ego - y_obs) * np.cos(h_obs) + y_obs

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
    
    # 提取自车和其他车辆信息
    def find_surrounding_vehicles(self):
        # 提取自车和其他车辆信息
        positions = [vehicle.position for vehicle in self.road.vehicles]
        speeds = [vehicle.speed for vehicle in self.road.vehicles]
        headings = [vehicle.heading for vehicle in self.road.vehicles]
        lane_indexs = [vehicle.lane_index for vehicle in self.road.vehicles]
        lane_indexs = [item[-1] for item in lane_indexs]
        vehicles_obs = np.column_stack(
            (np.array(positions), np.array(speeds), np.array(headings), np.array(lane_indexs)))
        ego_vehicle = vehicles_obs[0]
        self.ego_vehicle = ego_vehicle
        other_vehicles = vehicles_obs[1:]
        ego_position = ego_vehicle[:2]
        ego_lane_index = ego_vehicle[-1]
        # 初始化周围车辆信息
        lead_vehicle = None
        rear_vehicle = None
        left_lead_vehicle = None
        left_rear_vehicle = None
        right_lead_vehicle = None
        right_rear_vehicle = None
        min_distance_lead = float('inf')
        min_distance_rear = float('inf')
        min_distance_left_lead = float('inf')
        min_distance_left_rear = float('inf')
        min_distance_right_lead = float('inf')
        min_distance_right_rear = float('inf')

        for vehicle in other_vehicles:
            other_position = vehicle[:2]
            other_lane_index = vehicle[-1]
            distance = np.linalg.norm(other_position - ego_position)

            # 判断当前车道的前后车
            if other_lane_index == ego_lane_index:
                if other_position[0] > ego_position[0] and distance < min_distance_lead:
                    min_distance_lead = distance
                    lead_vehicle = vehicle
                if other_position[0] < ego_position[0] and distance < min_distance_rear:
                    min_distance_rear = distance
                    rear_vehicle = vehicle

            # 判断左车道的前后车
            if other_lane_index == ego_lane_index - 1:
                if other_position[0] > ego_position[0] and distance < min_distance_left_lead:
                    min_distance_left_lead = distance
                    left_lead_vehicle = vehicle
                if other_position[0] < ego_position[0] and distance < min_distance_left_rear:
                    min_distance_left_rear = distance
                    left_rear_vehicle = vehicle

            # 判断右车道的前后车
            if other_lane_index == ego_lane_index + 1:
                if other_position[0] > ego_position[0] and distance < min_distance_right_lead:
                    min_distance_right_lead = distance
                    right_lead_vehicle = vehicle
                if other_position[0] < ego_position[0] and distance < min_distance_right_rear:
                    min_distance_right_rear = distance
                    right_rear_vehicle = vehicle

        # 形成最终矩阵
        surrounding_vehicles = np.array([
            [1] + ego_vehicle.tolist(),
            [1] + lead_vehicle.tolist() if lead_vehicle is not None else [0] + [0, 0, 0, 0, ego_lane_index],
            [1] + rear_vehicle.tolist() if rear_vehicle is not None else [0] + [0, 0, 0, 0, ego_lane_index],
            [1] + left_lead_vehicle.tolist() if left_lead_vehicle is not None else [0] + [0, 0, 0, 0,
                                                                                          ego_lane_index - 1],
            [1] + left_rear_vehicle.tolist() if left_rear_vehicle is not None else [0] + [0, 0, 0, 0,
                                                                                          ego_lane_index - 1],
            [1] + right_lead_vehicle.tolist() if right_lead_vehicle is not None else [0] + [0, 0, 0, 0,
                                                                                            ego_lane_index + 1],
            [1] + right_rear_vehicle.tolist() if right_rear_vehicle is not None else [0] + [0, 0, 0, 0,
                                                                                            ego_lane_index + 1]
        ])

        return surrounding_vehicles
    
    # 基于IDM模型计算出加速度
    def calculate_idm_acceleration(self, ego_vehicle, lead_vehicle):
        # 设定变量
        v0 = 30  # 期望速度 (m/s)
        T = 1.5  # 安全时距 (s)
        a = 3  # 最大加速度 (m/s^2)
        b = 5  # 舒适减速度 (m/s^2)
        s0 = 10  # 最小距离 (m)
        delta = 4  # 加速度指数
        max_acceleration = 6  # 最大允许加速度 (m/s^2)
        min_acceleration = -6  # 最大允许减速度 (m/s^2)

        # 获取自车和前车信息
        v = ego_vehicle[3]  # 自车速度
        s = lead_vehicle[1]-ego_vehicle[1]  # 与前车的距离
        dv = v - lead_vehicle[3]  # 相对前车的速度差

        # 加一个小的正数epsilon来避免除以零
        epsilon = 1e-6
        s = max(s, epsilon)

        # 计算期望距离
        s_star = s0 + max(0, v * T + v * dv / (2 * np.sqrt(a * b)))

        # 计算加速度
        acceleration = a * (1 - (v / v0) ** delta - (s_star / s) ** 2)
        # print(a, v, v0, delta, s_star, s, acceleration)

        # 限制加速度范围
        acceleration = np.clip(acceleration, min_acceleration, max_acceleration)

        return acceleration

    # 基于IDM模型计算出加速度集合
    def calculate_all_accelerations(self):
        # surrounding_vehicles是一个7行6列的数组，
        # 第一行自车、第二行前车、第三行后车，
        # 第四行左侧车道前车、第五行左侧车道后车、第六行右侧车道前车、第七行右侧车道后车，
        # 第一列是否有车、最后一列车道的索引。
        surrounding_vehicles = self.find_surrounding_vehicles()
        ego_vehicle = surrounding_vehicles[0]
        lead_vehicle = surrounding_vehicles[1]
        rear_vehicle = surrounding_vehicles[2]
        left_lead_vehicle = surrounding_vehicles[3]
        left_rear_vehicle = surrounding_vehicles[4]
        right_lead_vehicle = surrounding_vehicles[5]
        right_rear_vehicle = surrounding_vehicles[6]

        # Initialize the results array
        accelerations = np.zeros((10, 3))
        acc_big = 0
        ### 没有变道前 ###
        # 自车加速度
        ego_acceleration0 = self.calculate_idm_acceleration(ego_vehicle, lead_vehicle)
        accelerations[0] = [ego_vehicle[0], ego_acceleration0, ego_vehicle[5]]

        # 自车后车加速度
        rear_acceleration0 = self.calculate_idm_acceleration(rear_vehicle, ego_vehicle) if rear_vehicle[0] == 1 else acc_big
        accelerations[1] = [rear_vehicle[0], rear_acceleration0, rear_vehicle[5]]

        # 左侧后车加速度
        left_rear_acceleration0 = self.calculate_idm_acceleration(left_rear_vehicle, left_lead_vehicle) if left_rear_vehicle[0] == 1 else acc_big
        accelerations[2] = [left_rear_vehicle[0], left_rear_acceleration0, left_rear_vehicle[5]]

        # 右侧后车加速度
        right_rear_acceleration0 = self.calculate_idm_acceleration(right_rear_vehicle, right_lead_vehicle) if right_rear_vehicle[0] == 1 else acc_big
        accelerations[3] = [right_rear_vehicle[0], right_rear_acceleration0, right_rear_vehicle[5]]

        ### 向左变道后 ###
        # 新的自车加速度
        new_ego_acceleration1 = self.calculate_idm_acceleration(ego_vehicle, left_lead_vehicle)
        accelerations[4] = [ego_vehicle[0], new_ego_acceleration1, ego_vehicle[5]-1]

        # 新的自车后车加速度
        new_rear_acceleration1 = self.calculate_idm_acceleration(rear_vehicle, lead_vehicle) if rear_vehicle[0] == 1 else acc_big
        accelerations[5] = [rear_vehicle[0], new_rear_acceleration1, rear_vehicle[5]]

        # 新的左侧后车加速度
        new_left_rear_acceleration1 = self.calculate_idm_acceleration(left_rear_vehicle, ego_vehicle) if left_rear_vehicle[0] == 1 else acc_big
        accelerations[6] = [left_rear_vehicle[0], new_left_rear_acceleration1, left_rear_vehicle[5]]
        # print(new_left_rear_acceleration1)

        ### 向右变道后 ###
        # 新的自车加速度
        new_ego_acceleration2 = self.calculate_idm_acceleration(ego_vehicle, right_lead_vehicle)
        accelerations[7] = [ego_vehicle[0], new_ego_acceleration2, ego_vehicle[5]+1]

        # 新的自车后车加速度
        new_rear_acceleration2 = self.calculate_idm_acceleration(rear_vehicle, lead_vehicle) if rear_vehicle[0] == 1 else acc_big
        accelerations[8] = [rear_vehicle[0], new_rear_acceleration2, rear_vehicle[5]]

        # 新的右侧后车加速度
        new_right_rear_acceleration2 = self.calculate_idm_acceleration(right_rear_vehicle, ego_vehicle) if right_rear_vehicle[0] == 1 else acc_big
        accelerations[9] = [right_rear_vehicle[0], new_right_rear_acceleration2, right_rear_vehicle[5]]
        return accelerations

    def mobil_lane_change_decision(self):
        # 获取所有加速度数据
        accelerations = self.calculate_all_accelerations()
        # 常量定义
        b_safe = -3  # 安全减速度（假设值）
        p = 0  # 权重（假设值）
        a_th = 0.1  # 优势阈值（假设值）

        # accelerations0：自车a
        # accelerations1：自车后车a
        # accelerations2：自车左侧后车a
        # accelerations3：自车右侧后车a
        # 左边道
        # accelerations4：自车a
        # accelerations5：自车后车a
        # accelerations6：自车左侧后车a
        # 右边道
        # accelerations7：自车a
        # accelerations8：自车后车a
        # accelerations9：自车右侧后车a

        # 获取加速度信息
        a_e = accelerations[0][1]
        self.a_ego_idm = a_e
        a_n_left = accelerations[2][1]
        a_o_left = accelerations[1][1]
        a_n_right = accelerations[3][1]
        a_o_right = accelerations[1][1]

        a_tilde_e_left = accelerations[4][1]
        a_tilde_n_left = accelerations[6][1]
        a_tilde_o_left = accelerations[5][1]

        a_tilde_e_right = accelerations[7][1]
        a_tilde_n_right = accelerations[9][1]
        a_tilde_o_right = accelerations[8][1]

        # 车道索引
        ego_lane_index = accelerations[0][2]
        left_lane_index = accelerations[4][2]
        right_lane_index = accelerations[7][2]

        # 初始化安全性和优势值
        left_safe = False
        right_safe = False
        left_advantage = float('-inf')
        right_advantage = float('-inf')
        # print(a_tilde_e_left, a_e, a_tilde_n_left, a_n_left, a_tilde_o_left, a_o_left)
        # 检查左变道安全性
        if left_lane_index in [0, 1, 2, 3] :
            left_safe = a_tilde_n_left >= b_safe
            if left_safe:
                left_advantage = (a_tilde_e_left - a_e + p * (a_tilde_n_left - a_n_left + a_tilde_o_left - a_o_left))

        # 检查右变道安全性
        if right_lane_index in [0, 1, 2, 3]:
            right_safe = a_tilde_n_right >= b_safe
            if right_safe:
                right_advantage = (
                            a_tilde_e_right - a_e + p * (a_tilde_n_right - a_n_right + a_tilde_o_right - a_o_right))

        # 选择最大优势值的变道方向
        self.best_direction = "保持车道"
        if left_safe and left_advantage >= a_th and left_advantage > right_advantage:
            self.best_direction = "左变道"
        elif right_safe and right_advantage >= a_th:
            self.best_direction = "右变道"
        if self.best_direction == "保持车道":
            target_lane_index = ego_lane_index
        elif self.best_direction == "左变道":
            target_lane_index = left_lane_index
        elif self.best_direction == "右变道":
            target_lane_index = right_lane_index
        # print("left_advantage:", left_advantage,"\n",
        #       "right_advantage:", right_advantage, "\n",
        #       "best_direction:",  self.best_direction, "\n",
        #       )
        return target_lane_index, a_e

    # 转向控制函数，根据IDM和MOBIL模型进行车道变换决策，并计算合适的转向角度。
    def steering_control(self):
        """
        转向控制函数，根据IDM和MOBIL模型进行车道变换决策，并计算合适的转向角度。

        参数:
        self: 当前对象实例，包含车辆状态和其他相关信息。

        返回:
        float: 计算得到的转向角度（单位：弧度）。
        """
        LENGTH = 5  # 车辆长度 [m]
        TAU_HEADING = 0.2  # 航向时间常数 [s]
        TAU_LATERAL = 0.6  # 横向时间常数 [s]
        KP_HEADING = 1 / TAU_HEADING  # 航向控制增益
        KP_LATERAL = 1 / TAU_LATERAL  # 横向控制增益 [1/s]
        MAX_STEERING_ANGLE = np.pi / 3  # 最大转向角度 [rad]
        MAX_STEERING_RATE = np.pi / 18  # 最大转角变化率 [rad/s]，可以根据需要调整

        # 根据MOBIL模型进行车道变换决策
        target_lane_index, _ = self.mobil_lane_change_decision()

        # 获取当前车辆的状态
        ego_x, ego_y, ego_v, ego_heading, ego_lane_index = self.ego_vehicle

        # 计算目标车道的y坐标
        target_y = target_lane_index * 4

        # 计算当前车辆与目标车道的横向偏差
        delta_y = ego_y - target_y
        delta_y1 = 12 - ego_y
        delta_y2 = 0 - ego_y

        # 当前车辆在车道上的位置坐标
        lane_coords = [ego_x, delta_y]

        # 横向位置控制
        lateral_speed_command = -KP_LATERAL * lane_coords[1]

        # 将横向速度转换为航向命令
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(ego_v), -1, 1))
        heading_ref = np.clip(heading_command, -np.pi / 4, np.pi / 4)

        # 航向控制
        heading_rate_command = KP_HEADING * utils.wrap_to_pi(heading_ref - ego_heading)

        # 将航向变化率转换为转向角度
        slip_angle = np.arcsin(np.clip(LENGTH / 2 / utils.not_zero(ego_v) * heading_rate_command, -1, 1))
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        # 在特定条件下增加转向角度
        if delta_y1 < 0 and ego_heading > 0:
            steering_angle = 4 * steering_angle
        if delta_y2 > 0 and ego_heading < 0:
            steering_angle = 4 * steering_angle

        # 限制转角变化率
        if hasattr(self, 'previous_steering_angle'):
            steering_angle_change = steering_angle - self.previous_steering_angle
            if abs(steering_angle_change) > MAX_STEERING_RATE:
                steering_angle = self.previous_steering_angle + np.sign(steering_angle_change) * MAX_STEERING_RATE

        # 存储当前的转角用于下次计算
        self.previous_steering_angle = steering_angle

        # 更新当前车辆的转向角度
        self.s_ego_mobil = float(steering_angle)

        # 打印调试信息
        # print(delta_y1, delta_y2, ego_heading, self.s_ego_mobil, target_lane_index)

        return float(steering_angle)




class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = info["agents_rewards"]
        terminated = info["agents_terminated"]
        truncated = info["agents_truncated"]
        return obs, reward, terminated, truncated, info
