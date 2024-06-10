import random
import carla


def spawn_points():
    # intersection2 random loc for pedestrian groups
    c1 = random.uniform(53, 54)
    c2 = random.uniform(48, 49)
    # intersection3 random loc for pedestrian groups
    c3 = random.uniform(3, 4)
    c4 = random.uniform(9, 10)
    # intersection4 random loc for pedestrian groups
    c5 = random.uniform(-98, -99)
    c6 = random.uniform(-103, -104)
    # intersection5 random loc for pedestrian groups
    c7 = random.uniform(4, 5)
    c8 = random.uniform(9, 10)
    # intersection6 random loc for pedestrian groups
    c9 = random.uniform(57, 58)
    c10 = random.uniform(62, 63)
    # intersection7 random loc for pedestrian groups
    c11 = random.uniform(-58.5, -57.5)
    c12 = random.uniform(-53.5, -54.5)

    spawn_points_dict = {
        'intersection1': {
            'ego_car': carla.Transform(carla.Location(x=50.9, y=28.35, z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=77.5, y=24.9, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
                                 carla.Transform(carla.Location(x=72.5, y=24.9, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=71, y=31.5, z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=87.5, y=32, z=1.0), carla.Rotation(pitch=0.0, yaw=-90, roll=0.0))],
            'pedestrians2': [carla.Transform(carla.Location(x=87.5, y=29.5, z=1.0),
                                             carla.Rotation(pitch=0.0, yaw=-90, roll=0.0)),
                             carla.Transform(carla.Location(x=87.5, y=31, z=1.0),
                                             carla.Rotation(pitch=0.0, yaw=-90, roll=0.0)),
                             carla.Transform(carla.Location(x=88, y=32, z=1.0),
                                             carla.Rotation(pitch=0.0, yaw=-90, roll=0.0)),
                             carla.Transform(carla.Location(x=87.5, y=33.5, z=1.0),
                                             carla.Rotation(pitch=0.0, yaw=-90, roll=0.0))],
            'pedestrian_direction': [0, -1, 92]
        },
        'intersection2': {
            'ego_car': carla.Transform(carla.Location(x=44.35, y=random.uniform(66, 70), z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=-89, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=43.846821, y=random.uniform(38, 50), z=1.0),
                                     carla.Rotation(pitch=0.0, yaw=269.934875, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=47.071621, y=random.uniform(45, 55), z=1.0),
                                      carla.Rotation(pitch=0.0, yaw=-90, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=random.uniform(-0.2, 0.2) + c1 + i * 0.7, y=35.5+random.uniform(-0.5, 0.5), z=1.0),
                carla.Rotation(pitch=0.0, yaw=180, roll=0.0)) for i in range(5)],
            'pedestrians2': [carla.Transform(
                carla.Location(x=random.uniform(-0.2, 0.2) + c2 + i * 0.7, y=35+random.uniform(-0.5, 0.5), z=1.0),
                carla.Rotation(pitch=0.0, yaw=180, roll=0.0)) for i in range(5)],
            'pedestrian_direction': [-1, 0, 29]
        },
        'intersection3': {
            'ego_car': carla.Transform(carla.Location(x=random.uniform(-5, 5), y=13.191462, z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=random.uniform(0, -7.5), y=16.691898, z=1.0),
                                     carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                     carla.Transform(carla.Location(x=random.uniform(-12.5, -19), y=16.691898, z=1.0),
                                     carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                     carla.Transform(carla.Location(x=random.uniform(-24, -28), y=16.691898, z=1.0),
                                     carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                     carla.Transform(carla.Location(x=random.uniform(-24, -27), y=13.110546, z=1.0),
                                     carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=random.uniform(-17, -23), y=9.763391, z=1.0),
                                      carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=-32 + random.uniform(-0.5, 0.5), y=random.uniform(-0.2, 0.2) + c3 + i * 0.7, z=1.0),
                carla.Rotation(pitch=0.0, yaw=90, roll=0.0)) for i in range(5)],
            'pedestrians2': [carla.Transform(
                carla.Location(x=-32 + random.uniform(-0.5, 0.5), y=random.uniform(-0.2, 0.2) + c4 + i * 0.7, z=1.0),
                carla.Rotation(pitch=0.0, yaw=90, roll=0.0)) for i in range(5)],
            'pedestrian_direction': [0, 1, -38]
        },
        'intersection4': {
            'ego_car': carla.Transform(carla.Location(x=-104, y=random.uniform(70, 80), z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=-107.320274, y=random.uniform(45, 50), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0)),
                                 carla.Transform(carla.Location(x=-107.3, y=random.uniform(57, 60), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0)),
                                 carla.Transform(carla.Location(x=-107.459697, y=random.uniform(66, 70), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0)),
                                 carla.Transform(carla.Location(x=-103.8, y=random.uniform(45, 47), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=-101, y=random.uniform(45, 50), z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0)),
                         carla.Transform(carla.Location(x=-101, y=random.uniform(55, 60), z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=-89.357765, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=random.uniform(-0.2, 0.2) + c5 + i * 0.7, y=random.uniform(-0.5, 0.5) + 39.1, z=1.0),
                carla.Rotation(pitch=0.0, yaw=180, roll=0.0)) for i in range(5)],
            'pedestrians2': [carla.Transform(
                carla.Location(x=random.uniform(-0.2, 0.2) + c6 + i * 0.7, y=random.uniform(-0.5, 0.5) + 39.1, z=1.0),
                carla.Rotation(pitch=0.0, yaw=180, roll=0.0)) for i in range(5)],
            'pedestrian_direction': [-1, 0, 35]
        },
        'intersection5': {
            'ego_car': carla.Transform(carla.Location(x=random.uniform(-70, -63), y=12.945686, z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=random.uniform(-79, -85), y=16.446157, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                                 carla.Transform(carla.Location(x=random.uniform(-68, -72), y=16.691898, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                                 carla.Transform(carla.Location(x=random.uniform(-60, -63), y=16.691898, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                                 carla.Transform(carla.Location(x=random.uniform(-79, -85), y=13.041880, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=random.uniform(-85, -79), y=9.5, z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0)),
                         carla.Transform(carla.Location(x=random.uniform(-65, -75), y=9.5, z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=180.159195, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=-93 + random.uniform(-0.5, 0.5), y=random.uniform(-0.2, 0.2) + c7 + i * -0.7, z=1.0),
                carla.Rotation(pitch=0.0, yaw=90, roll=0.0)) for i in range(5)],
            'pedestrians2': [carla.Transform(
                carla.Location(x=-93 + random.uniform(-0.5, 0.5), y=random.uniform(-0.2, 0.2) + c8 + i * -0.7, z=1.0),
                carla.Rotation(pitch=0.0, yaw=90, roll=0.0)) for i in range(5)],
            'pedestrian_direction': [0, 1, -98]
        },
        'intersection6': {
            'ego_car': carla.Transform(carla.Location(x=random.uniform(0, 10), y=66.280106, z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=-179.926727, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=random.uniform(-22, -25), y=66.196953, z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=-179.926727, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=random.uniform(-20, -25), y=62.3, z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=-179.926727, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=-31.5 + random.uniform(-0.5, 0.5), y=random.uniform(-0.2, 0.2) + c9 + i * -0.7, z=1.0),
                carla.Rotation(pitch=0.0, yaw=90, roll=0.0)) for i in range(5)],
            'pedestrians2': [carla.Transform(
                carla.Location(x=-31 + random.uniform(-0.5, 0.5), y=random.uniform(-0.2, 0.2) + c10 + i * -0.7, z=1.0),
                carla.Rotation(pitch=0.0, yaw=90, roll=0.0)) for i in range(5)],
            'pedestrian_direction': [0, 1, -36]
        },
        'intersection7': {
            'ego_car': carla.Transform(carla.Location(x=-52.106274, y=random.uniform(80, 87), z=1.0),
                                       carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0)),
            'cars_motorcycles': [carla.Transform(carla.Location(x=-48.505871, y=random.uniform(107, 112), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0)),
                                 carla.Transform(carla.Location(x=-48.54, y=random.uniform(101, 96), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0)),
                                 carla.Transform(carla.Location(x=-48.585056, y=random.uniform(87, 90), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0)),
                                 carla.Transform(carla.Location(x=-51.990807, y=random.uniform(107, 112), z=1.0),
                                                 carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0))],
            'bicycles': [carla.Transform(carla.Location(x=-54.8, y=random.uniform(111, 114), z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0)),
                         carla.Transform(carla.Location(x=-54.8, y=random.uniform(103, 107), z=1.0),
                                         carla.Rotation(pitch=0.0, yaw=89.838768, roll=0.0))],
            'pedestrians1': [carla.Transform(
                carla.Location(x=random.uniform(-0.2, 0.2) + c11 + i * -0.7, y=119 + random.uniform(-0.5, 0.5), z=1.0),
                carla.Rotation(pitch=0.0, yaw=0, roll=0.0)) for i in range(5)],
            'pedestrians2': [carla.Transform(
                carla.Location(x=random.uniform(-0.2, 0.2) + c12 + i * -0.7, y=119 + random.uniform(-0.5, 0.5), z=1.0),
                carla.Rotation(pitch=0.0, yaw=0, roll=0.0)) for i in range(5)],
            'pedestrian_direction': [1, 0, 125]
        }
    }
    return spawn_points_dict


pedestrians = [f"{i:04d}" for i in range(1, 29)]

blueprints_dict = {
        'ego_car': ["mkz_2017"],
        'cars_motorcycles': ["a2", "etron", "tt", "grandtourer", "impala", "c3",
                             "mustang", "charger_2020", "charger_police", "wrangler_rubicon", "mkz_2020", "coupe",
                             "cooper_s", "tesla", "patrol", "prius", "low_rider", "kawasaki", "vespa", "yamaha"],
        'cars': ["a2", "etron", "tt", "grandtourer", "impala", "c3", "mustang", "charger_2020", "charger_police",
                 "wrangler_rubicon", "mkz_2020", "coupe", "cooper_s", "tesla", "patrol", "prius"],
        'motorcycles': ["low_rider", "kawasaki", "vespa", "yamaha"],
        'bicycles': ["crossbike", "diamondback", "gazelle"],
        'pedestrians1': pedestrians[::2],
        'pedestrians2': pedestrians[1::2]
}

pedestrians_age_gp = {
    'child': ["0009", "0010", "0011", "0012", "0013", "0014"],
    'old': ["0015", "0016", "0017", "0020", "0021", "0022", "0023"]
}

obstacles = {'intersection1_1': [carla.Transform(carla.Location(x=53.5+i, y=34, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(70-53)+1)],
             'intersection1_2': [carla.Transform(carla.Location(x=84, y=33+i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(39-33)+1)],
             'intersection1_3': [carla.Transform(carla.Location(x=73+1.5*i, y=35, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((96-73)/1.5)+1)],
             'intersection1_4': [carla.Transform(carla.Location(x=89+i, y=22.4, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(96-89)+1)],
             'intersection2_1': [carla.Transform(carla.Location(x=42.1, y=32+1.5*i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((56-32)/1.5)+1)],
             'intersection2_2': [carla.Transform(carla.Location(x=50.118519, y=32+2*i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((57-32)/2)+1)],
             'intersection3_1': [carla.Transform(carla.Location(x=5-1.5*i, y=6, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(55/1.5)+1)],
             'intersection3_2': [carla.Transform(carla.Location(x=-45+i, y=20, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(12)+1)],
             'intersection4_1': [carla.Transform(carla.Location(x=-109.4, y=30+1.5 * i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((70-30)/1.5)+1)],
             'intersection4_2': [carla.Transform(carla.Location(x=-96, y=30+1.5 * i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((37.5-30)/1.5)+1)],
             'intersection5_1': [carla.Transform(carla.Location(x=-53-1.5*i, y=18.8, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(50/1.5)+1)],
             'intersection5_2': [carla.Transform(carla.Location(x=-53-1.5*i, y=6, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int(50/1.5)+1)],
             'intersection6_1': [carla.Transform(carla.Location(x=-37+1.5*i, y=68, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((37+30)/1.5)+1)],
             'intersection6_2': [carla.Transform(carla.Location(x=-37+1.5*i, y=60, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((37-32)/1.5)+1)],
             'intersection7_1': [carla.Transform(carla.Location(x=-46.7, y=75+1.5*i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((128-75)/1.5)+1)],
             'intersection7_2': [carla.Transform(carla.Location(x=-58, y=103+2.5*i, z=0.0),
                                 carla.Rotation(pitch=0, yaw=0, roll=0)) for i in range(int((128-103)/2.5)+1)]
             }
