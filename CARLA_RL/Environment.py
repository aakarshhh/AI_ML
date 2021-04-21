import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import argparse
import logging
import random
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError :
    pass
import carla
IM_WIDTH = 640
IM_HEIGHT = 480
spe = 20
class CarlaEnv:
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):

        argparser = argparse.ArgumentParser(
        description=__doc__)
        argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
        args = argparser.parse_args()
        
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x = 2.7 ,y = 0, z =0.5))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(0.2)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.005)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        self.front_camera = i2
    def step(self, action):
        if action == 0 :
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= -0.7))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0.7))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(brake = 1))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=-1))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer= 0))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=1))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(brake = 0.25))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -1
        elif kmh < 30 :
            done = False
            reward = -0.2
        else:
            done = False
            reward = 0.3
        if self.episode_start + spe < time.time():
            done = True
        return self.front_camera, reward, done, None

