import os
from glob import glob

import numpy as np
import torch

from args import args


if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True

if not args.seed:
    args.seed = np.random.randint(1, 10**8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')

args.out_filename = None


def get_args_features():
    features = 'MLP_nd{net_depth}_nw'
    for element in args.net_width:
        features += '-%s'%element

    if args.bias:
        features += '_bias'

    if args.optimizer != 'adam':
        features += '_{optimizer}'
    if args.clip_grad:
        features += '_cg{clip_grad:g}'

    features = features.format(**vars(args))

    return features


def init_out_filename():
    # if not args.out_dir:
    #     return
    features = get_args_features()
    template = '{args.out_dir}/{features}/out{args.out_infix}'
    args.out_filename = template.format(**{**globals(), **locals()})


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    # if not args.out_dir:
    #     return
    init_out_filename()
    ensure_dir(args.out_filename)
    if args.save_step:
        ensure_dir(args.out_filename + '_save/')


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def clear_err():
    if args.out_filename:
        open(args.out_filename + '.err', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def my_err(s):
    if args.out_filename:
        with open(args.out_filename + '.err', 'a', newline='\n') as f:
            f.write(s + u'\n')
    if not args.no_stdout:
        print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        print_fn('{} = {}'.format(k, v))
    print_fn('')


def parse_checkpoint_name(filename):
    filename = os.path.basename(filename)
    filename = filename.replace('.state', '')
    step = int(filename)
    return step


def get_last_checkpoint_step():
    if not (args.out_filename and args.save_step):
        return -1
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    if not filename_list:
        return -1
    step = max([parse_checkpoint_name(x) for x in filename_list])
    return step


def clear_checkpoint():
    if not (args.out_filename and args.save_step):
        return
    filename_list = glob('{}_save/*.state'.format(args.out_filename))
    for filename in filename_list:
        os.remove(filename)



def sample_store(sample):
    ensure_dir(args.out_filename + '_samples/')
    for sample_n in range(args.batch_size):
        with open(args.out_filename + '_samples/sample%d_N%d_c%s_s%s_T%.2f'%(sample_n,args.n,args.c,args.graph_seed,1./args.beta), "w+") as text_file:
            for spin in sample[sample_n,:]:
                text_file.write("%s\n" %int(spin))


def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)


def moving_average(x,window):
    window=10000
    mean_x = np.zeros(len(x))
    count_x = np.zeros(len(x))
    for i,x_value in enumerate(x):
        mean_x[i:(i+window)]+=x_value
        count_x[i:(i+window)] += 1
    for i,m in enumerate(mean_x):
        mean_x[i]=mean_x[i]/count_x[i]
    return mean_x


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
