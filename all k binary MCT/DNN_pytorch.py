import os
# path = 'D:\\Program Files\\OneDrive - TU Eindhoven\\PhD project\\Machine Learning\\'
path = "C:\\Users\\s158686\\OneDrive - TU Eindhoven\\PhD project\\Machine Learning\\DNN-pytorch\\all k binary MCT\\"
os.chdir(path)
import pandas as pd
import time
import glob
import seaborn as sns
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

# leave to -1 to read all the data in the folder
NS_to_open=-1

# The input data consists of the following informations
input_labels=[]
for si in range(100):
    input_labels.append('S11%d'%si)
for si in range(100):
    input_labels.append('S12%d'%si)
for si in range(100):
    input_labels.append('S22%d'%si)
input_labels.append('eta')
# and those are the output 
input_labels.append('log t')
input_labels.append('k')

full_in = []
full_out = []


print('\n*******************\n\t Creating input database: ')
listSfiles = shuffle(glob.glob(path + 'Data\\Sk_data\\Sk_eta*'))
# listSfiles = shuffle(glob.glob( 'Data/Sk_data/Sk_eta*'))
num_files = len(listSfiles[:NS_to_open])
for Sfile in listSfiles[:NS_to_open]:
    # The first 100 + 1 columns are for the input data
    eta = (Sfile.split('_eta_')[1]).split('.txt')[0]
    print('Reading data for eta = ', eta)

    new_in_data = np.loadtxt(Sfile)
    new_in_data = np.append(new_in_data, eta)
    # new_in_data = np.array([eta])

    # then for this input we have all the time dependent output that I store as additional columns 
    phifile = path + 'Data/phi_data/phi_eta_'+eta+'.txt'
    # phifile = 'Data/phi_data/phi_eta_'+eta+'.txt'
    new_out_data = np.loadtxt(phifile)
    for index in range(new_out_data.shape[0]):
        full_in.append(np.append(new_in_data, (new_out_data[index, 0], new_out_data[index, 1])))
        full_out.append(new_out_data[index, 2:])
        # Add the line that contains IN + OUT to the full database
#        print(full_df)


full_df = pd.DataFrame(np.array(full_in))
full_df.columns=input_labels
full_df["F11"] = [full_out[i][0] for i in range(new_out_data.shape[0]*num_files)]
# full_df["F12"] = [full_out[i][1] for i in range(new_out_data.shape[0]*num_files)]
full_df["F22"] = [full_out[i][2] for i in range(new_out_data.shape[0]*num_files)]



full_df = full_df.reset_index(drop=True)
full_df = shuffle(full_df)

# print(full_df)

# Here we can shuffle the database as we want
#   [ ... ]

# Separate the data in X,Y
X = full_df.drop(['F11', 'F22'], axis=1)
# print(X.head())
# print(X.shape)


y = full_df[['F11', 'F22']]
# print(y.head())
# print(y.shape)






# # Here we can normalize the data or do any pre-processing
import sklearn.preprocessing
min_max_scalerX = sklearn.preprocessing.MinMaxScaler()
min_max_scalery = sklearn.preprocessing.MinMaxScaler()


X_scaled = X.copy()
X_scaled = min_max_scalerX.fit_transform(X_scaled)

y_scaled = y.copy()
y_scaled = min_max_scalery.fit_transform(y_scaled)

# Use sklearn to make train and test sets
from sklearn.model_selection import train_test_split
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=5)




print('\n\n\t\t*** MLP ***')
init_out_dir()
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


net = MLP(**vars(args))
net.to(args.device)
# and make dataframes into torch tensors
torch.device(args.device)
X_train_scaled = (torch.from_numpy(X_train_scaled)).to(device=args.device,dtype=default_dtype_torch)
X_test_scaled =  (torch.from_numpy(X_test_scaled)).to(device=args.device ,dtype=default_dtype_torch)
y_train_scaled = (torch.from_numpy(y_train_scaled)).to(device=args.device,dtype=default_dtype_torch)
y_test_scaled =  (torch.from_numpy(y_test_scaled)).to(device=args.device ,dtype=default_dtype_torch)
print(X_train_scaled)
my_log('{}\n'.format(net))

params = list(net.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = int(sum([np.prod(p.shape) for p in params]))
my_log('Total number of trainable parameters: {}'.format(nparams))
named_params = list(net.named_parameters())

training_samples = utils_data.TensorDataset(X_train_scaled, y_train_scaled)
data_loader_trn = utils_data.DataLoader(training_samples, batch_size=args.batch_size, drop_last=False, shuffle=True)


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


criterion = nn.L1Loss()

train_time = 0
train_loss = 0
start_time = time.time()
init_time = time.time() - start_time
my_log('init_time = {:.3f}'.format(init_time))
tolerance = args.tolerance

my_log('Training...')
for step in range(last_step + 1, args.max_step + 1):
    # clear the gradients of all optimized variables
    cum_loss = 0.0
    train_start_time = time.time()
    optimizer = torch.optim.Adam(params, lr=args.lr/(step+1)**1.5, betas=(0.9, 0.999), weight_decay=0.0001)
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
    
    if args.print_step and step % args.print_step == 0:
        if step > 0:
            train_time /= args.print_step
        used_time = time.time() - start_time
        r2_test = r2_loss(net(X_test_scaled), y_test_scaled)
        
        r2_train = r2_loss(net(X_train_scaled), y_train_scaled)
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
        train_time = 0
        if r2_train > 1-tolerance:
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



# sys.exit()


import matplotlib.pyplot as plt
eta05 = 0.50
eta0515 = 0.5159
eta055 = 0.55
SK05 = np.loadtxt(path+'Data\\Sk_data\\example_Sk_eta_%.10f.txt' %eta05)
Ft05 = np.loadtxt(path+'Data\\phi_data\\example_phi_eta_%.10f.txt' %eta05)
SK055 = np.loadtxt(path+'Data\\Sk_data\\example_Sk_eta_%.10f.txt' %eta055)
Ft055 = np.loadtxt(path+'Data\\phi_data\\example_phi_eta_%.10f.txt' %eta055)
SK0515 = np.loadtxt(path+'Data\\Sk_data\\example_Sk_eta_%.10f.txt' %eta0515)
Ft0515 = np.loadtxt(path+'Data\\phi_data\\example_phi_eta_%.10f.txt' %eta0515)

plt.figure(figsize=[5,5])
example_X = min_max_scalerX.transform(np.array([np.hstack((SK05, [eta05, logt, Ft05[1, 1]] )).ravel() for logt in np.linspace(-6, 10, 1000)]))
example_prediction = net(torch.tensor(example_X))
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:, 0], label="DNN FBB", lw=2)
# plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAB", lw=2)
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAA", lw=2)
plt.plot(Ft05[:,0], Ft05[:,2], 'o', ms=2, label="MCT FBB")
# plt.plot(Ft05[:,0], Ft05[:,3], 'o', ms=2, label="MCT FAB")
plt.plot(Ft05[:,0], Ft05[:,3], 'o', ms=2, label="MCT FAA")
plt.legend()
# plt.ylim([0, 1])

plt.xlabel("log10(t)")
plt.ylabel("phi(k,t) at kD=%.2f" %Ft05[1, 1])
plt.title("volume fraction = %.2f"%eta05)
# plt.savefig(path+"predictionDNN_%d.png" %(NS_to_open*0.8), dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=[5,5])
example_X = min_max_scalerX.transform(np.array([np.hstack((SK055, [eta055, logt, Ft055[1, 1]] )).ravel() for logt in np.linspace(-6, 10, 1000)]))
example_prediction = net(torch.tensor(example_X))
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,0], label="DNN FBB", lw=2)
# plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAB", lw=2)
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAA", lw=2)

plt.plot(Ft055[:,0], Ft055[:,2], 'o',ms=2, label="MCT FBB")
# plt.plot(Ft055[:,0], Ft055[:,3], 'o',ms=2, label="MCT FAB")
plt.plot(Ft055[:,0], Ft055[:,3], 'o',ms=2, label="MCT FAA")
plt.legend()
plt.xlabel("log10(t)")
plt.ylim([0, 1])
plt.ylabel("phi(k,t) at kD=%.2f" %Ft055[1, 1])
plt.title("volume fraction = %.2f"%eta055)

# plt.savefig(path+"predictionDNN_%d.png" %(NS_to_open*0.8), dpi=500, bbox_inches="tight")
plt.show()


plt.figure(figsize=[5,5])
example_X = min_max_scalerX.transform(np.array([np.hstack((SK0515, [eta0515, logt, Ft0515[1, 1]] )).ravel() for logt in np.linspace(-6, 10, 1000)]))
example_prediction = net(torch.tensor(example_X))
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,0], label="DNN FBB", lw=2)
# plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAB", lw=2)
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-2], min_max_scalery.inverse_transform(example_prediction.detach().numpy())[:,1], label="DNN FAA", lw=2)

plt.plot(Ft0515[:,0], Ft0515[:,2], 'o',ms=2, label="MCT FBB")
# plt.plot(Ft0515[:,0], Ft0515[:,3], 'o',ms=2, label="MCT FAB")
plt.plot(Ft0515[:,0], Ft0515[:,3], 'o',ms=2, label="MCT FAA")

plt.ylim([0, 1])

plt.legend()
plt.xlabel("log10(t)")
plt.ylabel("phi(k,t) at kD=%.2f" %Ft0515[1, 1])
plt.title("volume fraction = %.2f"%eta0515)

# plt.savefig(path+"predictionDNN_%d.png" %(NS_to_open*0.8), dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=[5,5])
error =np.sum((net(torch.tensor(X_test_scaled)).detach().numpy() - y_test_scaled.detach().numpy())**2, axis=1)
ax = plt.scatter(min_max_scalerX.inverse_transform(X_test_scaled)[:, -3], error, s=1)
plt.xlabel("volume fraction")
plt.ylabel("error")
plt.show()

error[error<10**-6] = 10**-6
plt.figure(figsize=[5, 5])
ax = plt.scatter(min_max_scalerX.inverse_transform(X_test_scaled)[:, -3],min_max_scalerX.inverse_transform(X_test_scaled)[:, -1], c=np.log10(error), s=2)
plt.xlabel("volume fraction")
plt.ylabel("log(time)")
plt.title("log10(Error)")
plt.colorbar()
# plt.savefig(path + "error_as_function_of_time_and_density_%d.png" %(NS_to_open*0.8), dpi=500, bbox_inches="tight")
plt.show()

