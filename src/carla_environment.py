import carla
from gymnasium import spaces
import random
import time
import datetime
import math
import numpy as np
import tensorflow as tf
from collections import deque
import transforms3d
from src.waypoints import *


class CarlaEnv:
    """
        Carla environment used for training RL agents for ethical decision-making or collision avoidance tasks.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Carla environment.

        Parameters:
        kwargs (dict): Dictionary of configuration parameters.
        """
        # Setting up Carla simulator
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()

        self.episode_max_time = kwargs['episode_timeout']
        self.episode_start_time = None
        self.actor_list = []

        # Ego vehicle parameters
        self.a_x_max_acc = kwargs['a_x_max_acceleration']
        self.a_x_max_braking = kwargs['a_x_max_braking']
        self.a_y_max = kwargs['a_y_max']
        self.mass = kwargs['mass']
        self.mu = kwargs['mu']
        self.t_gear_change = kwargs['t_gear_change']

        # Chrono physics parameters
        self.chrono_enabled = kwargs['use_chrono']
        self.chrono_path = kwargs['chrono_path']
        self.vehicle_json = kwargs['vehicle_json']
        self.powertrain_json = kwargs['powertrain_json']
        self.tire_json = kwargs['tire_json']

        # Inputted frames parameters
        self.image_width = kwargs['image_width']
        self.image_height = kwargs['image_height']
        self.history_length = kwargs['history_length']
        self.knob = kwargs['knob_value']
        self.action_space = spaces.Discrete(9)  # Actions from 1 to 9 as defined
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.image_width, self.image_height, self.history_length), dtype=np.float32)
        self.state_buffer = None

        # Obstacle blueprints which are being strategically placed to restrict the potential driving space of the agent
        self.obstacle_bp = self.bp.filter("chainbarrierend")[0]

        # Road friction parameters
        friction_bp = self.bp.find('static.trigger.friction')
        extent = carla.Location(700.0, 700.0, 700.0)
        friction_bp.set_attribute('friction', str(self.mu))
        friction_bp.set_attribute('extent_x', str(extent.x))
        friction_bp.set_attribute('extent_y', str(extent.y))
        friction_bp.set_attribute('extent_z', str(extent.z))

        # Blueprints of ego vehicle etc.
        self.bp_ego = self.bp.filter(blueprints_dict['ego_car'][0])[0]
        # self.bp_ego.set_attribute('color', '0, 0, 255') # set blue color for ego car
        self.collision_bp = self.bp.find('sensor.other.collision')
        self.camera_bp = self.bp.find('sensor.camera.rgb')
        self.camera_bp.set_attribute("image_size_x", f"{self.image_width}")
        self.camera_bp.set_attribute("image_size_y", f"{self.image_height}")
        self.sp_sensors = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.ego = None
        self.collision_sensor = None
        self.camera = None
        self.spawned = None
        self.crossed = False
        self.cross_points = [0, 0, 0, 0]
        self.collisions = []
        self.collision_history = []
        self.collision_speed = 0
        self.cars = []
        self.motorcycles = []
        self.bicycles = []
        self.pedestrians1 = []
        self.pedestrians2 = []
        self.evaluate = False
        self.seed = None

        # Spawn obstacles in the environment
        for intersection, transforms in obstacles.items():
            for i, transform in enumerate(transforms):
                obstacle_spawn = transform
                self.world.spawn_actor(self.obstacle_bp, obstacle_spawn)
        self.world.spawn_actor(friction_bp, carla.Transform(
            carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0)))

    def reset(self, evaluate):
        """
        Reset the environment to start a new episode.

        Parameters:
        evaluate (bool): Flag indicating whether to evaluate the environment.

        Returns:
        np.array: Initial state of the environment.
        """
        self.seed = random.randint(100, 400)
        self.state_buffer = deque([np.zeros((self.image_width, self.image_height), dtype=np.float32)
                                   for _ in range(self.history_length)], maxlen=self.history_length)
        self.cars = []
        self.motorcycles = []
        self.bicycles = []
        self.pedestrians1 = []
        self.pedestrians2 = []
        self.crossed = False
        self.cross_points = [0, 0, 0, 0]
        self.collisions = []
        self.collision_history = []
        if bool(self.actor_list):
            self.destroy_actors()
        self._generate_traffic(evaluate)
        self.episode_start_time = datetime.datetime.now()
        return self._get_state()

    def step(self, action):
        """
        Perform the action and return the next state, reward, done flag, and additional info.

        Parameters:
        action (int): Action to be performed.

        Returns:
        tuple: (observation, reward, done, info)
        """
        self._convert_action_index_to_control(action)  # Perform selected action to ego car

        observation = self._get_state()
        done = self._check_done()
        reward = self._compute_step_reward(self.seed) if not done else self._compute_terminal_reward(self.seed)
        return observation, reward, done, {}

    def _generate_traffic(self, evaluate):
        """
        Spawn the ego vehicle, other actors, and set up sensors.

        Parameters:
        evaluate (bool): Flag indicating whether to evaluate the environment.
        """
        self.evaluate = False if evaluate == 0 else True
        spawn_points_new = spawn_points()
        location = random.choice(list(spawn_points_new)[1:]) if not self.evaluate else list(spawn_points_new)[0]
        sp_ego = spawn_points_new[location]['ego_car']
        self.cross_points[0],  self.cross_points[1] = sp_ego.location.x, sp_ego.location.y

        control = carla.WalkerControl()
        control.direction.x = 0
        control.direction.y = 0
        control.direction.z = 0
        control.speed = 1.55  # 5.6 km/h

        if spawn_points_new[location]['pedestrian_direction'][0] == 0:
            control.direction.y = spawn_points_new[location]['pedestrian_direction'][1]
            self.cross_points[2] = spawn_points_new[location]['pedestrian_direction'][2]
        else:
            control.direction.x = spawn_points_new[location]['pedestrian_direction'][0]
            self.cross_points[3] = spawn_points_new[location]['pedestrian_direction'][2]

        self.spawned = False
        while not self.spawned:
            try:
                self.ego = self.world.spawn_actor(self.bp_ego, sp_ego)
                self.ego.set_simulate_physics(enabled=True)
                physics_control = self.ego.get_physics_control()
                physics_control.gear_switch_time = self.t_gear_change
                physics_control.mass = self.mass
                physics_control.center_of_mass = carla.Vector3D(x=-0.100000, y=0.000000, z=-0.350000)
                self.ego.apply_physics_control(physics_control)
                self.actor_list.append(self.ego)
                self.spectator.set_transform(carla.Transform(sp_ego.location + carla.Location(z=12),
                                                             carla.Rotation(pitch=-53, yaw=sp_ego.rotation.yaw)))

                self.camera = self.world.spawn_actor(self.camera_bp, self.sp_sensors, attach_to=self.ego)
                self.actor_list.append(self.camera)
                self.camera.listen(lambda data: self._get_camera_image(data))

                self.collision_sensor = self.world.spawn_actor(self.collision_bp, self.sp_sensors, attach_to=self.ego)
                self.actor_list.append(self.collision_sensor)
                self.collision_sensor.listen(lambda event: self._collision_data(event))

                if evaluate == 0:
                    for i in range(random.randint(1, len(spawn_points_new[location]['pedestrians1']))):
                        wildcard = random.choice(blueprints_dict['pedestrians1'])
                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[location]['pedestrians1'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrians1.append(wildcard)

                    for i in range(random.randint(1, len(spawn_points_new[location]['pedestrians2']))):
                        wildcard = random.choice(blueprints_dict['pedestrians2'])
                        bp_ped = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[location]['pedestrians2'][i]
                        pedestrian = self.world.spawn_actor(bp_ped, spawn_point)
                        self.actor_list.append(pedestrian)
                        self.pedestrians2.append(wildcard)

                    non_autopilot_actors_num = len(self.actor_list)

                    for i in range(random.randint(0, len(spawn_points_new[location]['bicycles']))):
                        wildcard = random.choice(blueprints_dict['bicycles'])
                        bp_bicycle = self.bp.filter(wildcard)[0]
                        spawn_point = spawn_points_new[location]['bicycles'][i]
                        bicycle = self.world.spawn_actor(bp_bicycle, spawn_point)
                        self.actor_list.append(bicycle)
                        self.bicycles.append(wildcard)

                    autopilot_actors_bicycle = len(self.actor_list)

                    for i in range(random.randint(1, len(spawn_points_new[location]['cars_motorcycles']))):
                        spawn_point = spawn_points_new[location]['cars_motorcycles'][i]
                        if random.random() < 0.6:  # 60% chance to spawn a car
                            wildcard = random.choice(blueprints_dict['cars'])
                            bp_car = self.bp.filter(wildcard)[0]
                            car = self.world.spawn_actor(bp_car, spawn_point)
                            self.actor_list.append(car)
                            self.cars.append(wildcard)
                        else:  # 40% chance to spawn a motorcycle
                            wildcard = random.choice(blueprints_dict['motorcycles'])
                            bp_motorcycle = self.bp.filter(wildcard)[0]
                            motorcycle = self.world.spawn_actor(bp_motorcycle, spawn_point)
                            self.actor_list.append(motorcycle)
                            self.motorcycles.append(wildcard)

                else:
                    bp_red_car = self.bp.filter("tesla")[0]
                    # bp_red_car.set_attribute('color', '255, 0, 0') set Red color
                    bp_motorcycle = self.bp.filter("yamaha")[0]
                    bp_bicycle = self.bp.filter("gazelle")[0]
                    bp_ped_adult_man = self.bp.filter("0028")[0]
                    bp_ped_adult_mother = self.bp.filter("0008")[0]
                    bp_ped_child = self.bp.filter("0011")[0]
                    bp_ped_old_man = self.bp.filter("0017")[0]
                    sp_red_car = spawn_points_new[location]['cars_motorcycles'][0]
                    sp_motorcycle = spawn_points_new[location]['cars_motorcycles'][1]
                    sp_bicycle = spawn_points_new[location]['bicycles'][0]
                    sp_ped_adult_man = spawn_points_new[location]['pedestrians1'][0]

                    self.cars.append("tesla")
                    self.motorcycles.append("yamaha")
                    self.bicycles.append("gazelle")
                    self.pedestrians1.append("0028")
                    self.pedestrians2.append("0028")
                    self.pedestrians2.append("0008")
                    self.pedestrians2.append("0011")
                    self.pedestrians2.append("0017")

                    if evaluate == 1:
                        ped_adult_man = self.world.spawn_actor(bp_ped_adult_man, sp_ped_adult_man)
                        self.actor_list.append(ped_adult_man)
                    else:
                        sp_ped_adult_man = spawn_points_new[location]['pedestrians2'][0]
                        sp_ped_adult_mother = spawn_points_new[location]['pedestrians2'][1]
                        sp_ped_child = spawn_points_new[location]['pedestrians2'][2]
                        sp_ped_old_man = spawn_points_new[location]['pedestrians2'][3]

                        ped_adult_man = self.world.spawn_actor(bp_ped_adult_man, sp_ped_adult_man)
                        ped_adult_mother = self.world.spawn_actor(bp_ped_adult_mother, sp_ped_adult_mother)
                        ped_child = self.world.spawn_actor(bp_ped_child, sp_ped_child)
                        ped_old_man = self.world.spawn_actor(bp_ped_old_man, sp_ped_old_man)

                        self.actor_list.append(ped_adult_man)
                        self.actor_list.append(ped_adult_mother)
                        self.actor_list.append(ped_child)
                        self.actor_list.append(ped_old_man)

                    non_autopilot_actors_num = len(self.actor_list)

                    bicycle = self.world.spawn_actor(bp_bicycle, sp_bicycle)

                    autopilot_actors_bicycle = len(self.actor_list)

                    motorcycle = self.world.spawn_actor(bp_motorcycle, sp_motorcycle)
                    red_car = self.world.spawn_actor(bp_red_car, sp_red_car)

                    self.actor_list.append(red_car)
                    self.actor_list.append(motorcycle)
                    self.actor_list.append(bicycle)

                time.sleep(1)
                self.ego.apply_control(carla.VehicleControl(throttle=0, brake=0, manual_gear_shift=True, gear=3))
                self.ego.enable_constant_velocity(carla.Vector3D(60 / 3.6, 0, 0))
                time.sleep(1)
                self.ego.apply_control(carla.VehicleControl(manual_gear_shift=False))
                self.ego.disable_constant_velocity()

                if self.chrono_enabled:
                    self.ego.enable_chrono_physics(5000, 0.002, self.vehicle_json, self.powertrain_json, self.tire_json,
                                                   self.chrono_path)

                for actor in self.actor_list[3:non_autopilot_actors_num]:
                    actor.apply_control(control)

                for actor in self.actor_list[non_autopilot_actors_num:autopilot_actors_bicycle]:
                    actor.enable_constant_velocity(carla.Vector3D(3.05, 0, 0))
                for actor in self.actor_list[autopilot_actors_bicycle:]:
                    random.seed(self.seed)
                    speed = random.uniform(5, 12) if not self.evaluate else 6.94  # 25 km/h for red car and motorcycle
                    actor.enable_constant_velocity(carla.Vector3D(speed, 0, 0))

                self.spawned = True
            except RuntimeError:
                print("Spawn failed, Retrying in 10s. Please check server connection and whether simulator is running.")
                time.sleep(10)
                self.destroy_actors()
                self.spawned = False
            pass

    def _convert_action_index_to_control(self, action):
        """
        Map the action index to vehicle control commands.

        Parameters:
        action (int): Action index.
        """
        if action == 1:  # No Action
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        elif action == 2:  # Braking
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        elif action == 3:  # Accelerating
            self.ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        elif action == 4:  # Turning left
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.8, brake=0.0))
        elif action == 5:  # Turning left and accelerating
            self.ego.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.8, brake=0.0))
        elif action == 6:  # Turning left and braking
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.8, brake=1.0))
        elif action == 7:  # Turning right
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.8, brake=0.0))
        elif action == 8:  # Turning right and accelerating
            self.ego.apply_control(carla.VehicleControl(throttle=1.0, steer=0.8, brake=0.0))
        elif action == 9:  # Turning right and braking
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.8, brake=1.0))

    def _get_state(self):
        """
        Get the current state of the environment.

        Returns:
        np.array: State buffer as an array.
        """
        # Stack frames along the last dimension to form (84, 84, 4)
        state = np.stack(list(self.state_buffer), axis=-1)
        return state

    def _get_camera_image(self, data):
        """
        Convert the image from the camera sensor to the state observation and add it to state buffer.

        Parameters:
        data (carla.Image): Image data from the camera sensor.
        """
        raw_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        raw_data = np.reshape(raw_data, (data.height, data.width, 4))
        raw_data = raw_data[:, :, :3]  # Exclude alpha channel

        image = tf.image.rgb_to_grayscale(raw_data)
        image = tf.image.resize(image, [self.image_width, self.image_height])
        image = tf.cast(image, tf.float32)
        # image = image / 255.0 # implemented in dqn_agent.build_cnn_model
        image = np.squeeze(image, axis=-1)

        self.state_buffer.append(image)
        pass

    def add_noise_to_image(self, image, variance=1.4):
        noise = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=(4, 84, 84))
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255)

    def get_noisy_image(self):
        # method to get noisy image
        images = self.state_buffer
        noisy_image = self.add_noise_to_image(images, variance=1.4)
        return np.stack(list(noisy_image), axis=-1)

    def change_lighting_conditions(self, weather_params):
        weather = carla.WeatherParameters(
            cloudiness=weather_params['cloudiness'],
            sun_altitude_angle=weather_params['sun_altitude_angle']
        )
        self.world.set_weather(weather)
        time.sleep(1)
        pass

    def _collision_data(self, event):
        """
        Handle collision data from the collision sensor.

        Parameters:
        event (carla.CollisionEvent): Collision event data.
        """
        speed = self._get_speed() * 3.6
        self.collision_speed = speed
        wildcard = event.other_actor.type_id
        found = False
        categories = ['cars', 'motorcycles', 'bicycles', 'pedestrians1', 'pedestrians2']
        for category in categories:
            for actor in blueprints_dict[category]:
                if wildcard.find(actor) != -1:
                    if not self.evaluate:
                        if category == 'pedestrians1':
                            for p in self.pedestrians1:
                                self._collision_history(p)
                        elif category == 'pedestrians2':
                            for p in self.pedestrians2:
                                self._collision_history(p)
                        else:
                            self._collision_history(actor)
                    elif wildcard == "0011" or wildcard == "0008":
                        self._collision_history("0011")
                        self._collision_history("0008")
                    else:
                        self._collision_history(actor)
                    found = True
                    break  # Exit the inner loop if a match is found
            if found:
                break  # Exit the outer loop if a match is found
        if not found:  # If no match was found after checking all categories
            self._collision_history("static")
        pass

    def _collision_history(self, item):
        if item not in self.collisions:
            self.collisions.append(item)
        pass

    def get_collision_history(self):
        # For evaluation scenario only.
        if not bool(self.collisions):
            return 1  # no collision
        elif self.collisions[0] == "static":
            return 2  # self-sacrifice
        elif self.collisions[0] == "tesla":
            return 3  # the red car
        elif self.collisions[0] == "yamaha":
            return 4  # motorcycle
        elif self.collisions[0] == "gazelle":
            return 5  # bicycle
        elif self.collisions[0] == "0028":
            return 6  # adult man
        elif self.collisions[0] == "0008" or self.collisions[0] == "0011":
            return 7  # mother + child
        elif self.collisions[0] == "0017":
            return 8  # old man
        else:
            print("something wierd!")
            return None

    def _compute_step_reward(self, seed):
        """
        Compute the reward for non-terminal steps.

        Parameters:
        seed (int): Random seed generated in each episode for generating same age of actors at each step.

        Returns:
        float: Step reward.
        """
        step_reward = 0
        speed = self._get_speed() * 3.6
        random.seed(seed)
        r1 = random.randint(20, 51)
        r2 = random.randint(20, 51)
        r3 = random.randint(5, 10)
        r4 = random.randint(60, 70)
        r5 = random.randint(20, 51)
        r6 = random.randint(5, 10)
        r7 = random.randint(60, 70)
        r8 = random.randint(20, 51)

        x = 1 - self.knob  # eq. 12

        ego_prob = []
        other_prob = []

        a_x, a_y = self.get_acceleration()

        for car_passenger in self.cars:  # assuming one person in each car
            prob = self._car_passenger_injury_probability(a=-6.6986, b=0.0333, v=speed)
            other_prob.append(prob)
            ego_prob.append(prob)

        for motorcyclist in self.motorcycles:
            age = r1 if not self.evaluate else 40
            prob = self._ped_cyclists_injury_probability(a=-4.555, b=0.040, c=0.011, v=speed, age=age)
            other_prob.append(prob)

        for bicyclist in self.bicycles:
            age = r2 if not self.evaluate else 25
            prob = self._ped_cyclists_injury_probability(a=-7.467, b=0.079, c=0.047, v=speed, age=age)
            other_prob.append(prob)

        for pedestrian in self.pedestrians1:
            if pedestrian in pedestrians_age_gp['child']:
                age = r3
            elif pedestrian in pedestrians_age_gp['old']:
                age = r4
            else:
                age = r5 if not self.evaluate else 38  # Man Pedestrian
            prob = self._ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, v=speed, age=age)
            other_prob.append(prob)

        for pedestrian in self.pedestrians2:
            if pedestrian in pedestrians_age_gp['child']:
                age = r6 if not self.evaluate else 7  # child
            elif pedestrian in pedestrians_age_gp['old']:
                age = r7 if not self.evaluate else 65  # Old Man
            else:
                age = r8 if not self.evaluate else 35  # Mother
            prob = self._ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, v=speed, age=age)
            other_prob.append(prob)

        a = 0 if self.a_x_max_braking < a_x < self.a_x_max_acc else 1
        b = 0 if  a_y < abs(self.a_y_max) else 1

        inversed_prob = [1 / p_v for p_v in other_prob]
        inversed_prob_ego = [1 / p_v for p_v in ego_prob]

        step_reward = ((1-0.2*a)*(1-0.8*b)/10000) * ((sum(inversed_prob)/x)+(sum(inversed_prob_ego)/self.knob))  # eq. 8
        return step_reward

    def _compute_terminal_reward(self, seed):
        """
        Compute the reward at the end of the episode.

        Parameters:
        seed (int): Random seed for generating the reward.

        Returns:
        float: Terminal reward.
        """
        if self._check_crossed() and not bool(self.collisions): #  eq. 11
            return 200

        terminal_reward = 0
        speed = self.collision_speed
        random.seed(seed)
        r1 = random.randint(20, 51)
        r2 = random.randint(20, 51)
        r3 = random.randint(5, 10)
        r4 = random.randint(60, 70)
        r5 = random.randint(20, 51)
        r6 = random.randint(5, 10)
        r7 = random.randint(60, 70)
        r8 = random.randint(20, 51)

        x = 1 - self.knob  # eq. 12

        ego_prob = []
        other_prob = []

        for i in self.collisions:
            if i in self.cars:
                prob = self._car_passenger_injury_probability(a=-6.6986, b=0.0333, v=speed)
                other_prob.append(prob)
                ego_prob.append(prob)
            elif i in self.motorcycles:
                age = r1 if not self.evaluate else 40
                prob = self._ped_cyclists_injury_probability(a=-4.555, b=0.040, c=0.011, v=speed, age=age)
                other_prob.append(prob)
            elif i in self.bicycles:
                age = r2 if not self.evaluate else 25
                prob = self._ped_cyclists_injury_probability(a=-7.467, b=0.079, c=0.047, v=speed, age=age)
                other_prob.append(prob)
            elif i in self.pedestrians1:
                if i in pedestrians_age_gp['child']:
                    age = r3
                elif i in pedestrians_age_gp['old']:
                    age = r4
                else:
                    age = r5 if not self.evaluate else 38  # Man Pedestrian
                prob = self._ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, v=speed, age=age)
                other_prob.append(prob)
            elif i in self.pedestrians2:
                if i in pedestrians_age_gp['child']:
                    age = r6 if not self.evaluate else 7  # child
                elif i in pedestrians_age_gp['old']:
                    age = r7 if not self.evaluate else 65  # Old Man
                else:
                    age = r8 if not self.evaluate else 35  # Mother
                prob = self._ped_cyclists_injury_probability(a=-6.190, b=0.078, c=0.038, v=speed, age=age)
                other_prob.append(prob)
            else:
                prob = self._car_passenger_injury_probability(a=-5.7641, b=0.0239, v=speed)  # Single car crash
                ego_prob.append(prob)

        terminal_reward = -400 * ((sum(other_prob) * x) + (sum(ego_prob)*self.knob))  # eq. 11

        return terminal_reward

    def _ped_cyclists_injury_probability(self, a, b, c, v, age):  # Eq. 5
        prob = 1/(1+math.exp(-1*(a + b*v + c*age)))
        return prob

    def _car_passenger_injury_probability(self, a, b, v):  # Eq. 6
        prob = 1/(1+math.exp(-1*(a + b*v)))
        return prob

    def get_location(self):  # for test scenario only
        x = 50.9  # ego start position at intersection_1 (evaluation)
        y = 28.35
        location = self.ego.get_transform().location
        return location.x-x, y-location.y

    def get_velocity(self):
        velocity = self.ego.get_velocity()
        return velocity.x, velocity.y

    def _get_speed(self):  # m/s
        velocity = self.ego.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def get_acceleration(self):  # m/s^2
        acc = self.ego.get_acceleration()
        rot = self.ego.get_transform().rotation
        pitch = np.radians(rot.pitch)  # deg to radian
        roll = np.radians(rot.roll)
        yaw = np.radians(rot.yaw)
        global_acc = np.array([acc.x, acc.y, acc.z])
        r = transforms3d.euler.euler2mat(roll, pitch, yaw).T
        acc_relative = np.dot(r, global_acc)
        return acc_relative[0], acc_relative[1]

    def _check_done(self):
        """
        Check if the episode is done.

        Returns:
        bool: True if the episode is done, False otherwise.
        """
        elapsed_time = datetime.datetime.now() - self.episode_start_time
        if bool(self.collisions):
            return True  # Stop the episode if there are any collisions
        elif self._check_crossed():
            return True  # Stop the episode if the ego car is crossed from generated scenario
        elif elapsed_time.total_seconds() > self.episode_max_time:
            return True  # Stop the episode if the maximum episode time is exceeded
        return False

    def _check_crossed(self):
        ego_location = self.ego.get_transform().location
        if self.cross_points[2] != 0:
            if np.sign(self.cross_points[0]-self.cross_points[2]) != np.sign(ego_location.x-self.cross_points[2]):
                self.crossed = True
        elif np.sign(self.cross_points[1]-self.cross_points[3]) != np.sign(ego_location.y-self.cross_points[3]):
            self.crossed = True
        return self.crossed

    def destroy_actors(self):
        self.collision_sensor.stop()
        self.camera.stop()
        for actor in self.actor_list:
            actor.destroy()
        time.sleep(1)
        self.actor_list = []
