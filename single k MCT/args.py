import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('network parameters')
group.add_argument('--n_in', type=int, default=102, help='number of inputs')
group.add_argument('--n_out', type=int, default=1, help='number of outputs')
group.add_argument('--net_depth', type=int, default=8, help='number of total layers')
group.add_argument('--net_width','--list', nargs='+', default=(200, 100, 60, 40, 40, 40), help='size of the hidden layers')
group.add_argument('--bias', action='store_true', default=True, help='use bias?')
group.add_argument(
    '--dtype',
    type=str,
    default='float64',
    choices=['float32', 'float64'],
    help='dtype')





group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--seed', type=int, default=0, help='random seed, 0 for randomized')
group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer')
group.add_argument('--lr', type=float, default=5e-4, help='learning rate')
group.add_argument('--tolerance', type=float, default=1e-3, help='tolerance for training')
group.add_argument('--batch_size', type=int, default=1000, help='training batch size')

group.add_argument(
    '--max_step', type=int, default=10**3, help='maximum number of steps')
group.add_argument(
    '--clip_grad',
    type=float,
    default=0,
    help='global norm to clip gradients, 0 for disabled')

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance')
group.add_argument(
    '--clear_checkpoint', action='store_true', help='clear checkpoint')
group.add_argument(
    '--print_step',
    type=int,
    default=5,
    help='number of steps to print log, 0 for disabled')
group.add_argument(
    '--save_step',
    type=int,
    default=100,
    help='number of steps to save network weights, 0 for disabled')
group.add_argument(
    '--cuda', type=int, default=-1, help='ID of GPU to use, -1 for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled')

args = parser.parse_args()
