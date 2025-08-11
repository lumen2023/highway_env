from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    一个高速公路驾驶环境。

    车辆在有多个车道的直道高速公路上行驶，通过达到高速度、保持在最右侧车道以及避免碰撞来获得奖励。
    """

    @classmethod
    def default_config(cls) -> dict:
        # 定义默认的配置参数
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,  # 车道数量
            "vehicles_count": 50,  # 车辆总数
            "controlled_vehicles": 1,  # 控制的车辆数量
            "initial_lane_id": None,  # 初始车道ID
            "duration": 40,  # 环境持续时间（秒）
            "ego_spacing": 2,  # 自车与前车的初始间距
            "vehicles_density": 1,  # 车辆密度
            "collision_reward": 0,  # 碰撞时获得的奖励
            "right_lane_reward": 0,  # 在最右侧车道行驶时获得的奖励
            "high_speed_reward": 1,  # 高速行驶时获得的奖励
            "lane_change_reward": 0,  # 每次变道时获得的奖励
            # "risk_reward": 0,  # 风险奖励
            "risk_reward": -0.5,  # 风险奖励
            "comfort_reward": 0,  # 舒适性奖励
            "reward_speed_range": [20, 30],  # 奖励速度范围
            "normalize_reward": True,  # 是否归一化奖励
            "offroad_terminal": True,  # 是否在驶出道路时终止
            "usempc_controller": False,  # 是否使用MPC控制器
        })
        return config

    def _reset(self) -> None:
        # 重置环境，创建道路和车辆
        self._create_road()
        self._create_vehicles()

    """创建由直线相邻车道组成的道路。"""
    def _create_road(self) -> None:
        """创建由直线相邻车道组成的道路。"""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
    
    """创建一些新的随机车辆并将它们添加到道路上。"""
    def _create_vehicles(self) -> None:
        """创建一些新的随机车辆并将它们添加到道路上。"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            # 创建被控车辆
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                # 创建其他随机车辆
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        定义奖励函数，以促进高速行驶、保持在最右侧车道以及避免碰撞。
        :param action: 上次执行的动作
        :return: 相应的奖励值
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            # 归一化奖励值
            reward = utils.lmap(reward,
                                [0,
                                 self.config["high_speed_reward"]],
                                [-1, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        计算并返回车辆的奖励字典。

        参数:
            action (Action): 执行的动作。

        返回:
            Dict[Text, float]: 包含不同奖励类型的字典。
        """
        # 获取当前车辆所在车道的相邻车道
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        # 根据车辆类型确定目标车道
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # 计算车辆的前向速度
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # 将前向速度映射到[-1, 1]区间内
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [-1, 1])

        # 计算风险奖励，包括碰撞、是否在道路上及额外成本
        risk = 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road) + self._cost()
        # risk = 5 * float(self.vehicle.crashed) + 5 * float(not self.vehicle.on_road)
        # risk = self._cost()
        # 返回包含高速奖励、在道路奖励和风险奖励的字典
        return {
            "high_speed_reward": np.clip(scaled_speed, -1, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "risk_reward": risk,
        }

    def _cost(self):
        """
        计算自车的风险代价函数，包括静态和动态风险场。
        """
        # 计算自车的机动性指标
        self.s_ego_mobil = self.steering_control()
        # 计算自车的加速度和车道变更决策
        _, self.a_ego_idm = self.mobil_lane_change_decision()

        # 风险场的权重系数
        Asta = 1
        Adyn = 1
        Ab = 1  # 道路边界的场强系数
        Al = 0.1  # 车道标记的场强系数

        # 风险场的分布范围参数
        σb = 1  # 道路边界的风险分布范围
        σl = 1  # 车道标记的风险分布范围

        # 速度和位置的敏感度参数
        kx = 1
        ky = 0.8
        kv = 2

        # 动态风险场的形状参数
        alpha = 0.9
        beta = 2

        # 障碍物的尺寸参数
        L_obs = 5
        W_obs = 2

        # 获取所有车辆的位置、速度和航向
        positions = [vehicle.position for vehicle in self.road.vehicles]
        speeds = [vehicle.speed for vehicle in self.road.vehicles]
        headings = [vehicle.heading for vehicle in self.road.vehicles]
        vehicles_obs = np.column_stack((np.array(positions), np.array(speeds), np.array(headings)))

        # 提取自车的信息
        x_ego, y_ego, v_ego, _ = vehicles_obs[0]

        # 道路边界的位置
        y_boundary = [-2, 14]
        # 车道标记的位置
        y_lane = [2, 6, 10]

        # 道路边界的势场
        Eb = np.zeros_like(x_ego)
        for boundary in y_boundary:
            Eb += Ab * np.exp(-((y_ego - boundary) ** 2) / (1 * σb ** 2))

        # 车道标记的势场
        El = np.zeros_like(x_ego)
        for lane in y_lane:
            El += Al * np.exp(-((y_ego - lane) ** 2) / (1 * σl ** 2))

        # 初始化静态和动态风险场的值
        Ustas = []
        Udyns = []

        # 遍历其他车辆，计算它们对自车的风险场
        for vehicle in vehicles_obs[1:]:
            x_obs, y_obs, v_obs, h_obs = vehicle

            # 计算风险场的分布范围
            σx = kx * L_obs
            σy = ky * W_obs
            σv = kv * np.abs(v_obs - v_ego)

            # 计算相对速度方向
            relv = 1 if v_obs >= v_ego else -1

            # 修正坐标旋转公式
            h_obs = -h_obs  # 修正障碍物的航向角
            x_rel = (x_ego - x_obs) * np.cos(h_obs) - (y_ego - y_obs) * np.sin(h_obs) + x_obs
            y_rel = (x_ego - x_obs) * np.sin(h_obs) + (y_ego - y_obs) * np.cos(h_obs) + y_obs
            # 设置一个最大输入值限制，防止溢出
            MAX_EXP_INPUT = 500  # 根据需求调整这个值

            # 计算静态风险场
            exp_input_sta = -((x_rel - x_obs) ** 2 / σx ** 2) ** beta - ((y_rel - y_obs) ** 2 / σy ** 2) ** beta
            exp_input_sta = np.clip(exp_input_sta, -MAX_EXP_INPUT, MAX_EXP_INPUT)  # 限制指数输入范围
            Usta = Asta * np.exp(exp_input_sta)

            # 计算动态风险场
            exp_input_dyn = -((x_rel - x_obs) ** 2 / σv ** 2 + (y_rel - y_obs) ** 2 / σy ** 2)
            exp_input_dyn = np.clip(exp_input_dyn, -MAX_EXP_INPUT, MAX_EXP_INPUT)  # 限制指数输入范围
            # 处理第二部分的指数计算，并确保避免溢出
            exp_input_dyn2 = -relv * (x_rel - x_obs - alpha * L_obs * relv)
            exp_input_dyn2 = np.clip(exp_input_dyn2, -MAX_EXP_INPUT, MAX_EXP_INPUT)  # 限制第二部分的输入范围
            Udyn = Adyn * (np.exp(exp_input_dyn) / (1 + np.exp(exp_input_dyn2)))

            # 将结果保存到列表中
            Ustas.append(Usta)
            Udyns.append(Udyn)

            # # 计算静态和动态风险场
            # Usta = Asta * np.exp(
            #     -((x_rel - x_obs) ** 2 / σx ** 2) ** beta - ((y_rel - y_obs) ** 2 / σy ** 2) ** beta)
            # Udyn = Adyn * (np.exp(-((x_rel - x_obs) ** 2 / σv ** 2 + (y_rel - y_obs) ** 2 / σy ** 2)) /
            #                (1 + np.exp(-relv * (x_rel - x_obs - alpha * L_obs * relv))))
            #
            # Ustas.append(Usta)
            # Udyns.append(Udyn)

        # 计算总的风险代价
        Ustas = np.array(Ustas).sum()
        Udyns = np.array(Udyns).sum()
        Ebs = Eb.sum()
        Els = El.sum()
        # 边界 + 车道中线 + 静态风险场 + 动态风险场
        U = Ustas + Udyns + Ebs + Els
        U = min(3, U)
        return U

    def _is_terminated(self) -> bool:
        """当自车发生碰撞时，结束当前回合。"""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """当达到时间限制时，截断当前回合。"""
        return self.time >= self.config["duration"]


class SafeHighwayEnvFast(HighwayEnv):
    """
    SafeHighwayEnvFast 类继承了 HighwayEnv，
    是 highway-v0 环境的一个变种，主要进行了一些修改以提高执行速度：
    - 降低仿真频率以减少计算量
    - 减少场景中的车辆数量，以及减少车道数量，缩短每个 episode 的持续时间
    - 只检测被控制车辆与其他车辆的碰撞
    """

    @classmethod
    def default_config(cls) -> dict:
        # 获取父类的默认配置，并进行修改
        cfg = super().default_config()
        cfg.update({
            "action": {
                "type": "ContinuousAction",  # 使用连续动作类型
                "acceleration_range": [-4, 4],  # 加速度范围设置为 -4 到 4
                "steering_range": [-0.1, 0.1]  # 转向角范围设置为 -0.1 到 0.1
            },
            "simulation_frequency": 25,  # 仿真频率设置为 25 Hz，降低计算量
            "policy_frequency": 10,  # 策略执行频率设置为 5 Hz
            "lanes_count": 4,  # 车道数量设置为 4 条
            "vehicles_count": 20,  # 场景中车辆的总数量设置为 20 辆
            "duration": 20,  # 每个 episode 的持续时间设置为 20 秒
            "ego_spacing": 1.5,  # 控制车辆的间距设置为 1.5 个单位距离
        })
        return cfg

    def _create_vehicles(self) -> None:
        # 调用父类的方法创建车辆
        super()._create_vehicles()
        # 禁用对未控制车辆的碰撞检测
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False  # 对于非受控车辆，不进行碰撞检测
