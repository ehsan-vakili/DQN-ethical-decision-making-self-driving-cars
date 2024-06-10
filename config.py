import argparse


def get_args():
    """
    Function to parse command line arguments and return them as a dictionary.
    The users can change "default" parameters to their own desired values.

    Returns:
    - dict: Configuration parameters for the DQN agent.
    """

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="DQN agent configuration")

    # Add command line arguments
    parser.add_argument('--image-width', type=int, default=84, help='Image width')
    parser.add_argument('--image-height', type=int, default=84, help='Image height')
    parser.add_argument('--history-length', type=int, default=4, help='Number of frames in history for state')
    parser.add_argument('--replay-buffer-capacity', type=int, default=20000, help='Capacity of replay buffer')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.996, help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.1, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.99992, help='Epsilon decay rate')
    parser.add_argument('--target-update', type=int, default=500, help='Target network update frequency')
    parser.add_argument('--episodes', type=int, default=8000, help='Number of training episodes')
    parser.add_argument("--episode-timeout", type=int, default=10, help='maximum episode time allowed (s)')
    parser.add_argument('--evaluation-interval', type=int, default=5000, help='Steps between evaluations')
    parser.add_argument('--evaluation-steps', type=int, default=250, help='Number of evaluation steps')
    parser.add_argument('--save-model-freq', type=int, default=500, help='save the model once per X training episodes')

    parser.add_argument('--knob-value', type=float, default=0.5, help='k ∈ {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}')

    parser.add_argument('--uncertainty-check', type=bool, default=False, help='uncertainty check during test phase')
    parser.add_argument('--sample-nets', type=int, default=5, help='number of sample networks during uncertainty test')

    parser.add_argument('--mass', type=int, default=1900, help='ego vehicle mass')
    parser.add_argument('--a-x-max-acceleration', type=float, default=3.5, help='max allowed longitudinal acceleration')
    parser.add_argument('--a-x-max-braking', type=float, default=-8.0, help='max allowed braking acceleration')
    parser.add_argument('--a-y-max', type=float, default=7.8, help='max allowed lateral acceleration')
    parser.add_argument('--mu', type=float, default=0.8, help='friction coefficient')
    parser.add_argument('--t-gear-change', type=float, default=0.2, help='duration of gear change')

    parser.add_argument('--use-chrono', type=bool, default=False, help='Enable chrono physics?')
    parser.add_argument('--chrono-path', type=str, default="C:/Program Files (x86)/ProjectChrono/data/vehicle/")
    parser.add_argument('--vehicle-json', type=str, default="sedan/vehicle/Sedan_Vehicle.json")
    parser.add_argument('--powertrain-json', type=str, default="sedan/powertrain/Sedan_SimpleMapPowertrain.json")
    parser.add_argument('--tire-json', type=str, default="sedan/tire/Sedan_Pac02Tire.json")

    # Parse arguments and return as dictionary
    args = parser.parse_args()
    return vars(args)
