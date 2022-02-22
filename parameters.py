import argparse
from functools import reduce

parser = argparse.ArgumentParser(description='tsn_marl')

# -- Basic information --
parser.add_argument('--nodes_num', type=int, default=6)
parser.add_argument('--data_path', type=str, default='Vehicle_NetWork',
                    help='the path archi data saved as ./DataSaved/{}/')

# unit: ms->time_slot
parser.add_argument('--slot_per_millisecond', type=int, default=4,
                    help='number of slots per millisecond')

# -- Architecture->archi.py --
parser.add_argument('--tt_flow_cycles', type=int, nargs='+', default=[64, 128, 256, 512],
                    help='flow_cycles(ms)')

# -- Node information during schedule->node.py --
# calculate at the end
parser.add_argument('--global_cycle', type=int, help='global_cycle(ms)')

# -- basic scheduler env: c++ engine --
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--render_every", type=int, default=10)
parser.add_argument("--n_round", type=int, default=30)
parser.add_argument("--render", type=bool, default=False)
parser.add_argument("--load_from", type=int, default=None)
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--name", type=str, default="test_scheduler")
parser.add_argument("--eval", type=bool, default=True)
parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])

# -- basic DDPG parameters --
# sigma control the rate of exploration
parser.add_argument("--sigma", type=float, default=0.02)


args = parser.parse_args()


def gcd(a, b):
    """
    calculate the greatest common divisor
    """
    if a < b:
        temp = b
        b = a
        a = temp
    remainder = a % b
    if remainder == 0:
        return b
    else:
        return gcd(remainder, b)


def lcm(a, b):
    """
    calculate the least common multiple
    """
    remainder = gcd(a, b)
    return int(a * b / remainder)


args.global_cycle = reduce(lcm, args.tt_flow_cycles)


if __name__ == '__main__':
    print(args.global_cycle)
