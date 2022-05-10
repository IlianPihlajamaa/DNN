import os
import shutil
os.chdir(os.path.dirname(os.path.realpath(__file__))) # change working directory to where this file is located
import pandas as pd
import time
import glob
import math
import sys
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import numpy as np
from MLP import MLP
from args import args
import torch.utils.data as utils_data
from utils import (
    r2_loss,
    default_dtype_torch,
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    init_out_dir,
    my_log,
    print_args,
    msq_loss,
    find_analytical_S_k
)
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy

def load_data(num_files, path, N_t_perfile=200, rhoname="rho", without="0", silent=False):

    # The input data consists of the following informations
    N_input = 64 + 4 # order, density, logt, k
    N_output = 1
    if not silent:
        print('\n*******************\n\t Creating input database: ')

    full_data_path1 =  os.path.join(path, 'Data','Sampled Data','F(k,t)','Fkt_%s*'%rhoname)
    full_data_path2 =  os.path.join(path, 'Data 2','Sampled Data','F(k,t)','Fkt_%s*'%rhoname)

    listFfiles1 = shuffle(glob.glob(full_data_path1))
    listFfiles2 = shuffle(glob.glob(full_data_path2))

    listFfiles = []
    picked1 = 0
    picked2 = 0
    tried1 = 0
    tried2 = 0
    while len(listFfiles) < num_files:
        if np.random.rand() > 0.5:
            Ffile = listFfiles1[tried1]
            tried1 += 1
            if Ffile.split('_')[4][:-4] in str(without) :
                continue
        
            listFfiles.append(Ffile)
            picked1 += 1
        else:
            Ffile = listFfiles2[tried2]
            tried2 += 1
            if Ffile.split('_')[4][:-4] in str(without):
                continue
            listFfiles.append(Ffile)
            picked2 += 1

    listFfiles = shuffle(listFfiles)
    X = np.zeros((num_files*N_t_perfile, N_input))
    y = np.zeros((num_files*N_t_perfile, N_output))
    i = 0 
    for filesopened, Ffile in enumerate(listFfiles):
        rho = Ffile.split('_')[2] # we get rho from the file name
        order = Ffile.split('_')[4][:-4] # we get rho from the file name

        # print('Reading data for rho = ', rho)
        Sfile = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(Ffile))), "Full Data", "S(k)", 'Sk_%s_%s.txt' %(rhoname, rho))
        new_in_data = np.loadtxt(Sfile) # Structure factors
        new_out_data = np.loadtxt(Ffile)
        X[i:i+N_t_perfile, :-4] = new_in_data[None, :] #Structure factors+rho
        X[i:i+N_t_perfile, -4] = 1/float(order) # order
        X[i:i+N_t_perfile, -3] = float(rho) # rho
        X[i:i+N_t_perfile, -2] = new_out_data[:, 1] # logt
        X[i:i+N_t_perfile, -1] = new_out_data[:, 0] # k
        y[i:i+N_t_perfile, 0] = new_out_data[:, -1] # F
        i+= N_t_perfile
        if filesopened % 20 == 0 and not silent:
            print(filesopened,"/", num_files, "files opened")

    # Shuffle
    permutation = np.random.permutation(num_files*N_t_perfile)
    X = X[permutation, :]   
    y = y[permutation, :]
    return X, y

def rescale_and_split_data(X, y):
    # # Here we can normalize the data or do any pre-processing
    # min_max_scalerX = sklearn.preprocessing.MinMaxScaler()
    # min_max_scalery = sklearn.preprocessing.MinMaxScaler()
    
    X_scaled = X.copy()
    # X_scaled = min_max_scalerX.fit_transform(X_scaled)
    
    y_scaled = y.copy()
    # y_scaled = min_max_scalery.fit_transform(y_scaled)
    
    # Use sklearn to make train and test sets
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=5)

    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled#, min_max_scalerX, min_max_scalery



print('\n\n\t\t*** Starting Machine Learning ***')
def define_training_model(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled):
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    print_args()
    args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')
    init_out_dir()
    # Convert them to torch tensors
    X_train_scaled = (torch.from_numpy(X_train_scaled)).to(device=args.device,dtype=default_dtype_torch)
    X_test_scaled =  (torch.from_numpy(X_test_scaled)).to(device=args.device ,dtype=default_dtype_torch)
    y_train_scaled = (torch.from_numpy(y_train_scaled)).to(device=args.device,dtype=default_dtype_torch)
    y_test_scaled =  (torch.from_numpy(y_test_scaled)).to(device=args.device ,dtype=default_dtype_torch)
    
    net = MLP(**vars(args))
    net.to(args.device)
    torch.device(args.device)

    my_log('{}\n'.format(net))
    
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.00001)
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))
    
    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename,
                                                      last_step))
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
    return net, optimizer, args, params, last_step, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled


def print_output(net, cum_loss, step, optimizer, args, train_time, start_time, r2_test, r2_train):
    if step > 0:
        train_time /= args.print_step
    used_time = time.time() - start_time
    my_log(
        'step = {}, loss = {:.8g}, r2_train = {:.8g}, r2_test = {:.8g}, train_time = {:.3f}, used_time = {:.3f}'
        .format(
            step,
            cum_loss,
            r2_train,
            r2_test,
            train_time,
            used_time,
        ))
    return
    
def my_loss(output, target, sigma, logt):
    loss = torch.mean(torch.sqrt((output - target)**2) * torch.exp(-(target - 0.5)**2 / sigma**2) * (1 + torch.exp((logt+2)/5)))
    return loss 
    
def my_weight(target, sigma, logt):
    loss =  np.exp(-(target - 0.5)**2 / sigma**2) * (1 + np.exp((logt+2)/5))
    return loss 

def load_model(net, optimizer, args, params):
    last_step = get_last_checkpoint_step()
    print("Loading model at checkpoint", last_step)

    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename,
                                                      last_step),
                           map_location=torch.device('cpu'))
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])



def train_model(net, optimizer, args, params):
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename,
                                                      last_step),
                           map_location=torch.device('cpu'))
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])


    print_args()
    # criterion = nn.L1Loss() 

    criterion = my_loss # loss function
    train_time = 0
    start_time = time.time()
    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))
    
    my_log('Training...')
    for step in range(last_step + 1, args.max_step + 1):
        cum_loss = 0.0
        train_start_time = time.time()
        #divide training in batches
        for batch_idx, (data, target) in enumerate(data_loader_trn):
        # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            logt = data[:, -2]
            # calculate the loss
            sigma = 0.4
            loss = criterion(output, target, sigma, logt)
            optimizer.zero_grad()
    
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(params, args.clip_grad)
        
            # perform a single optimization step (parameter update)
            optimizer.step()
        
            # update running training loss
            cum_loss += loss.item() 
            
        train_time += time.time() - train_start_time
        cum_loss /= ((data.shape)[0]/args.batch_size)

        if args.print_step and step % args.print_step == 0:
            r2_test = msq_loss(net(X_test_scaled), y_test_scaled)
            r2_train = msq_loss(net(X_train_scaled), y_train_scaled)
            print_output(net, cum_loss, step, optimizer, args, train_time, start_time, r2_test, r2_train)
            train_time = 0
            if r2_train < args.tolerance:
                print("Training has coverged with required tolerance")
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, '{}_save/{}.state'.format(
                    args.out_filename, step))
                break
        if (args.out_filename and args.save_step
                and step % args.save_step == 0):
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, '{}_save/{}.state'.format(
                args.out_filename, step))

def plot_data(net, path):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)
    

    k_array = np.loadtxt("Data\\k_array.txt")
    logtarray = np.loadtxt("Data\\logt_array.txt")
    relative_full_data_path =  '\\Data\\Full Data\\F(k,t)\\Fkt*'
    listFfiles = glob.glob(path + relative_full_data_path)
    colors = ["red", "blue", "green", "black", "purple"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8,4])
    ax1.set_ylabel("$S(k)$")
    ax1.set_xlabel("$k$")
    for i in range(5):
        colori = colors[i]
        index = np.random.randint(len(listFfiles))
        k_index = np.random.randint(len(k_array))
        k = k_array[k_index]
        Ffile = listFfiles[index]
        rho = Ffile.split('_')[2] # we get rho from the file name
        order = Ffile.split('_')[4][:-4] # we get order from the file name
        net_prediction = np.zeros_like(logtarray)
        Sk = np.loadtxt(path+'\\Data\\Full Data\\S(k)\\Sk_rho_%s.txt' %rho)
        input_prediction = np.array([np.hstack((Sk, [1/float(order), float(rho), logt, k] )).ravel() for logt in logtarray])
        input_prediction_scaled = input_prediction
        output_prediction_scaled = net(torch.tensor(input_prediction_scaled))
        output_prediction = output_prediction_scaled.detach().numpy()
        F = np.loadtxt(Ffile)
        ax1.plot(k, Sk[k_index], "*", color = colori)
        ax1.plot(k_array, Sk, "-", color = colori, label="$\\rho=%.1f$"%float(rho))
        ax2.plot(logtarray, output_prediction, "--", color = colori)
        ax2.plot(logtarray, F[k_index, :], color = colori)
    ax1.legend()
    ax2.set_ylim([0, 1])
    ax2.set_xlim([-6, 10])
    ax2.set_xlabel(r"$\log_{10}(t)$")
    ax2.set_ylabel(r"$F(k, t)$")

    plt.savefig("Plots\\example_test_%d.png"%np.random.randint(10**9), dpi = 500)
    plt.show()
    


path = os.path.dirname(os.path.realpath(__file__))
num_files = 1
X, y = load_data(num_files, path)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = rescale_and_split_data(X, y)
net, optimizer, args, params, last_step, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = define_training_model(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
training_samples = utils_data.TensorDataset(X_train_scaled, y_train_scaled)
data_loader_trn = utils_data.DataLoader(training_samples, batch_size=args.batch_size, drop_last=False, shuffle=True)
# train_model(net, optimizer, args, params, last_step)
load_model(net, optimizer, args, params)
# find_phase_diagram(net, path)
plot_data(net, path)





