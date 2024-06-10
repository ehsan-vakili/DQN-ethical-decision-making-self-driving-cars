import os
import sys
import glob
import random
from threading import Thread
import carla

stop = False
actors_list = []


def initialize_carla_client():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return world


def get_blueprints(world):
    blueprint_library = world.get_blueprint_library()
    return {
        'walker': blueprint_library.filter('0028')[0],
        'motorcycle': blueprint_library.filter('kawasaki')[0],
        'vehicle': blueprint_library.filter("bmw")[0],
    }


def input_handler():
    global stop, actors_list
    world = initialize_carla_client()
    blueprints = get_blueprints(world)
    spectator = world.get_spectator()
    map = world.get_map()

    while not stop:
        user_input = input().strip().lower()
        loc = spectator.get_transform()
        loc1 = spectator.get_location()

        if user_input in ['c', 'p', 'b']:
            lane_type = {
                'c': carla.LaneType.Driving,
                'p': carla.LaneType.Driving | carla.LaneType.Biking | carla.LaneType.Sidewalk,
                'b': carla.LaneType.Driving | carla.LaneType.Biking,
            }[user_input]
            blueprint = {
                'c': blueprints['vehicle'],
                'p': blueprints['walker'],
                'b': blueprints['motorcycle'],
            }[user_input]

            waypoint = map.get_waypoint(loc1, project_to_road=True, lane_type=lane_type)
            actor = world.spawn_actor(blueprint, loc)
            actors_list.append(actor)

            entity = "Car" if user_input == 'c' else "Pedestrian" if user_input == 'p' else "Motorcycle"
            print(f"-------------------{entity}-------------------")
            print(f"Spectator Transform: {spectator.get_transform()}")
            print(f"Waypoint Transform: {waypoint.transform}")

        elif user_input == 'q':
            stop = True
        else:
            print("Please input a valid value from {c, p, b, q}")

    for actor in actors_list:
        actor.destroy()


process = Thread(target=input_handler)
process.start()
process.join()

for actor in actors_list:
    actor.destroy()
