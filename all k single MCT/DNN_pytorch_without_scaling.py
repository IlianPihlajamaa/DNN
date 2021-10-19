import os
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
)
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(NS_to_open, path, N_t_perfile=1000):

    # The input data consists of the following informations
    N_input = 103
    N_output = 1
    
    relative_data_path =  '\\Data\\Sk_data\\Sk_eta*'
    listSfiles = shuffle(glob.glob(path + relative_data_path))
    X = np.zeros((num_files*N_t_perfile, N_input))
    y = np.zeros((num_files*N_t_perfile, N_output))
    print('\n*******************\n\t Creating input database: ')
    i = 0 
    for Sfile in listSfiles[:NS_to_open]:
        eta = (Sfile.split('_eta_')[1]).split('.txt')[0] # we get eta from the file name
        print('Reading data for eta = ', eta)
        new_in_data = np.loadtxt(Sfile) # Structure factors
        phifile = path + '\\Data\\phi_data\\phi_eta_' + eta + '.txt'
        new_out_data = np.loadtxt(phifile)
        X[i:i+N_t_perfile, :-3] = new_in_data[None, :] #Structure factors+eta
        X[i:i+N_t_perfile, -3] = float(eta) # eta
        X[i:i+N_t_perfile, -2] = new_out_data[:, 0] # logt
        X[i:i+N_t_perfile, -1] = new_out_data[:, 1] # k
        y[i:i+N_t_perfile, 0] = new_out_data[:, -1] # F
        i+= N_t_perfile

    # Shuffle
    permutation = np.random.permutation(num_files*N_t_perfile)
    X = X[permutation, :]   
    y = y[permutation, :]  
    return X, y

def rescale_and_split_data(X, y):
    # # Here we can normalize the data or do any pre-processing
    min_max_scalerX = sklearn.preprocessing.MinMaxScaler()
    min_max_scalery = sklearn.preprocessing.MinMaxScaler()
    

    
    # Use sklearn to make train and test sets
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, test_size=0.2, random_state=5)

    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, min_max_scalerX, min_max_scalery



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
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
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
    
    
def train_model(net, optimizer, args, params, last_step):
    criterion = nn.MSELoss() # loss function
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
        
            # calculate the loss
            loss = criterion(output,target)
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
        cum_loss /= (num_files*1000 / args.batch_size)
        r2_test = r2_loss(net(X_test_scaled), y_test_scaled)

        r2_train = r2_loss(net(X_train_scaled), y_train_scaled)
        if args.print_step and step % args.print_step == 0:
            print_output(net, cum_loss, step, optimizer, args, train_time, start_time, r2_test, r2_train)
            train_time = 0

        if r2_train > 1-args.tolerance:
            print("Training has coverged with required tolerance")
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, '{}_save/{}.state'.format(
                args.out_filename, step))
            break

#function below was written for binary mixture. Needs to be adapted for single component
# def plot_data(net, min_max_scalerX, min_max_scalery):
#     # This function plots example predictions at three different densities
    
#     eta05 = 0.53
#     eta0515 = 0.5159
#     eta055 = 0.55
#     # load data
#     SK05 = np.loadtxt(path+'\\Data\\Sk_data\\example_Sk_eta_%.10f.txt' %eta05)
#     Ft05 = np.loadtxt(path+'\\Data\\phi_data\\example_phi_eta_%.10f.txt' %eta05)
#     SK055 = np.loadtxt(path+'\\Data\\Sk_data\\example_Sk_eta_%.10f.txt' %eta055)
#     Ft055 = np.loadtxt(path+'\\Data\\phi_data\\example_phi_eta_%.10f.txt' %eta055)
#     SK0515 = np.loadtxt(path+'\\Data\\Sk_data\\example_Sk_eta_%.10f.txt' %eta0515)
#     Ft0515 = np.loadtxt(path+'\\Data\\phi_data\\example_phi_eta_%.10f.txt' %eta0515)
    
#     # first figure
#     plt.figure(figsize=[5,5])
#     example_X = min_max_scalerX.transform(np.array([np.hstack((SK05, [eta05, logt, Ft05[1, 1]] )).ravel() for logt in np.linspace(-6, 10, 1000)]))
#     example_prediction = net(torch.tensor(example_X))
#     plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:, 0], label="DNN FAA", lw=2)
#     # plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAB", lw=2)
#     plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FBB", lw=2)
#     plt.plot(Ft05[:,0], Ft05[:,2], 'o', ms=2, label="MCT FBB")
#     # plt.plot(Ft05[:,0], Ft05[:,3], 'o', ms=2, label="MCT FAB")
#     plt.plot(Ft05[:,0], Ft05[:,4], 'o', ms=2, label="MCT FAA")
#     plt.legend()
#     plt.ylim([0, 1])
#     plt.xlabel("log10(t)")
#     plt.ylabel("phi(k,t) at kD=%.2f" %Ft05[1, 1])
#     plt.title("volume fraction = %.2f"%eta05)
#     plt.savefig(path+"\\predictionDNN050.png", dpi=500, bbox_inches="tight")
#     plt.show()
    
#     #second figure
#     plt.figure(figsize=[5,5])
#     example_X = min_max_scalerX.transform(np.array([np.hstack((SK055, [eta055, logt, Ft055[1, 1]] )).ravel() for logt in np.linspace(-6, 10, 1000)]))
#     example_prediction = net(torch.tensor(example_X))
#     plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,0], label="DNN FAA", lw=2)
#     # plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAB", lw=2)
#     plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FBB", lw=2)
#     plt.plot(Ft055[:,0], Ft055[:,2], 'o',ms=2, label="MCT FBB")
#     # plt.plot(Ft055[:,0], Ft055[:,3], 'o',ms=2, label="MCT FAB")
#     plt.plot(Ft055[:,0], Ft055[:,4], 'o',ms=2, label="MCT FAA")
#     plt.legend()
#     plt.xlabel("log10(t)")
#     plt.ylim([0, 1])
#     plt.ylabel("phi(k,t) at kD=%.2f" %Ft055[1, 1])
#     plt.title("volume fraction = %.2f"%eta055)
#     plt.savefig(path+"\\predictionDNN055.png", dpi=500, bbox_inches="tight")
#     plt.show()
    
#     # third figure
#     plt.figure(figsize=[5,5])
#     example_X = min_max_scalerX.transform(np.array([np.hstack((SK0515, [eta0515, logt, Ft0515[1, 1]] )).ravel() for logt in np.linspace(-6, 10, 1000)]))
#     example_prediction = net(torch.tensor(example_X))
#     plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,0], label="DNN FAA", lw=2)
#     # plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAB", lw=2)
#     plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FBB", lw=2)
#     plt.plot(Ft0515[:,0], Ft0515[:,2], 'o',ms=2, label="MCT FBB")
#     # plt.plot(Ft0515[:,0], Ft0515[:,3], 'o',ms=2, label="MCT FAB")
#     plt.plot(Ft0515[:,0], Ft0515[:,4], 'o',ms=2, label="MCT FAA")
#     plt.ylim([0, 1])
#     plt.legend()
#     plt.xlabel("log10(t)")
#     plt.ylabel("phi(k,t) at kD=%.2f" %Ft0515[1, 1])
#     plt.title("volume fraction = %.2f"%eta0515)
#     plt.savefig(path+"\\predictionDNN0515.png", dpi=500, bbox_inches="tight")
#     plt.show()
    
#     # plot the error as function of density
#     plt.figure(figsize=[5,5])
#     error =np.sum((net(torch.tensor(X_test_scaled)).detach().numpy() - y_test_scaled.detach().numpy())**2, axis=1)
#     ax = plt.scatter(min_max_scalerX.inverse_transform(X_test_scaled)[:, -3], error, s=1)
#     plt.xlabel("volume fraction")
#     plt.ylabel("error")
#     plt.show()
    
#     # plot the error as function of density and time
#     error[error<10**-6] = 10**-6
#     plt.figure(figsize=[5, 5])
#     ax = plt.scatter(min_max_scalerX.inverse_transform(X_test_scaled)[:, -3],min_max_scalerX.inverse_transform(X_test_scaled)[:, -2], c=np.log10(error), s=2)
#     plt.xlabel("volume fraction")
#     plt.ylabel("log(time)")
#     plt.title("log10(Error)")
#     plt.colorbar()
#     # plt.savefig(path + "error_as_function_of_time_and_density_%d.png" %(NS_to_open*0.8), dpi=500, bbox_inches="tight")
#     plt.show()
#     return


path = os.path.dirname(os.path.realpath(__file__))
num_files = 800
X, y = load_data(num_files, path)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, min_max_scalerX, min_max_scalery = rescale_and_split_data(X, y)
net, optimizer, args, params, last_step, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = define_training_model(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
training_samples = utils_data.TensorDataset(X_train_scaled, y_train_scaled)
data_loader_trn = utils_data.DataLoader(training_samples, batch_size=args.batch_size, drop_last=False, shuffle=True)
train_model(net, optimizer, args, params, last_step)
# plot_data(net, min_max_scalerX, min_max_scalery)


## Shap

import shap

# Calculate shaply values
begin = time.time()
(data, target) = next(iter(data_loader_trn))
explainer = shap.DeepExplainer(net, X_train_scaled[1:1000])
shap_values = explainer.shap_values(X_test_scaled[1:1000])
print("elapsed time = ", time.time()-begin)

# visualize 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)
k_array = np.linspace(0.2, 39.8, 100)
plt.plot(k_array, np.std(shap_values[:, :-3], axis=0))
plt.xlabel(r"Wavelength $k$")
plt.ylabel("std of S(k) Shapley values")
plt.savefig("shaply_withoutMinMax.png", dpi=300, bbox_inches="tight")
plt.show()

labels = []
for i in range(100):
    labels.append("S(k=" + str(k_array[i]) + ")")

    
sortedlabels = [labels[i] for i in np.argsort(np.std(shap_values[:, :-3], axis=0))[::-1]]
print("The most important structure factor values are:")
print(sortedlabels[:5])