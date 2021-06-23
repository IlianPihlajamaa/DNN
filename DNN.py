import pandas as pd
import glob
import seaborn as sns
import math
import sys
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import numpy as np

#this is the path to where the data is stored
path = 'D:\\Program Files\\OneDrive - TU Eindhoven\\PhD project\\Machine Learning\\DNN1 sample\\'


# leave to -1 to read all the data in the folder
NS_to_open=500

# The input data consists of the following information: we have 100 values of S(k) for different k
# logt, at which we want the intermediate scattering function, and the volume fraction eta.
input_labels=[]
for si in range(100):
    input_labels.append('S%d'%si)
input_labels.append('eta')
# and those are the output 
input_labels.append('log t')

full_in = []
full_out = []

print('\n*******************\n\t Creating input database: ')
listSfiles = shuffle(glob.glob(path + 'Data\\Sk_data\\Sk_eta*'))
for Sfile in listSfiles[:NS_to_open]:
    # The first 100 + 1 columns are for the input data
    eta = (Sfile.split('_eta_')[1]).split('.txt')[0]
    print('Reading data for eta = ', eta)

    new_in_data = np.loadtxt(Sfile)
    new_in_data = np.append(new_in_data, eta)
    # new_in_data = np.array([eta])

    # then for this input we have all the time dependent output that I store as additional columns 
    phifile = path + 'Data/phi_data/phi_eta_'+eta+'.txt'
    new_out_data = np.loadtxt(phifile)
    for index in range(new_out_data.shape[0]):
        full_in.append(np.append(new_in_data, new_out_data[index, 0]))
        full_out.append(new_out_data[index, 1])
        # Add the line that contains IN + OUT to the full database
#        print(full_df)


full_df = pd.DataFrame(np.array(full_in))
del full_in
full_df.columns=input_labels
full_df["F"] = full_out
del full_out


full_df = full_df.reset_index(drop=True)
full_df = shuffle(full_df)

print(full_df)

# Here we can shuffle the database as we want
#   [ ... ]

# Separate the data in X,Y
X = full_df.drop(['F'], axis=1)
print(X.head())
print(X.shape)


y = full_df[['F']]
print(y.head())
print(y.shape)






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




from sklearn.neural_network import MLPRegressor
# model = MLPRegressor(hidden_layer_sizes=(50,10), activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
                      # learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=100000, 
                      # shuffle=True, random_state=666, tol=0.0001, verbose=True, warm_start=False, 
                      # momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                      # beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=150000)

model = MLPRegressor(random_state=1, max_iter=500, verbose=True, shuffle=True, hidden_layer_sizes=(100, 100, 10))
model.fit(X_train_scaled, np.ravel(y_train_scaled))

print('R2 score over the training set: %f'%model.score(X_train_scaled, y_train_scaled))
print('R2 score over the test set: %f'%model.score(X_test_scaled, y_test_scaled))


import matplotlib.pyplot as plt
# visualize the prediction with example curves
SK05 = np.loadtxt(path+'Data\\Sk_data\\Sk_eta_0.5017710322.txt')
Ft05 = np.loadtxt(path+'Data\\phi_data\\phi_eta_0.5017710322.txt')
SK055 = np.loadtxt(path+'Data\\Sk_data\\Sk_eta_0.5486267326.txt')
Ft055 = np.loadtxt(path+'Data\\phi_data\\phi_eta_0.5486267326.txt')
SK0515 = np.loadtxt(path+'Data\\Sk_data\\Sk_eta_0.5152075135.txt')
Ft0515 = np.loadtxt(path+'Data\\phi_data\\phi_eta_0.5152075135.txt')


plt.plot(Ft05[:,0], Ft05[:,1], 'o', ms=3, label="MCT eta=0.5")
plt.plot(Ft055[:,0], Ft055[:,1], 'o',ms=3, label="MCT eta=0.55")
plt.plot(Ft0515[:,0], Ft0515[:,1], 'o',ms=3, label="MCT eta=0.515")


example_X = min_max_scalerX.transform(np.array([np.hstack((SK05, [0.5017710322, logt] )).ravel() for logt in np.linspace(-10, 10, 100)]))
example_prediction = model.predict(example_X)
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-1], min_max_scalery.inverse_transform(example_prediction[:,None]), label="DNN eta=0.5", lw=2)



example_X = min_max_scalerX.transform(np.array([np.hstack((SK055, [0.5486267326, logt] )).ravel() for logt in np.linspace(-10, 10, 100)]))
example_prediction = model.predict(example_X)
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-1], min_max_scalery.inverse_transform(example_prediction[:,None]), label="DNN eta=0.55", lw=2)


example_X = min_max_scalerX.transform(np.array([np.hstack((SK0515, [0.5152075135, logt] )).ravel() for logt in np.linspace(-10, 10, 100)]))
example_prediction = model.predict(example_X)
plt.plot(min_max_scalerX.inverse_transform(example_X)[:,-1], min_max_scalery.inverse_transform(example_prediction[:,None]), label="DNN eta=0.515", lw=2)


plt.legend()
plt.xlabel("log10(t)")
plt.ylabel("phi(k,t) at kD=7.4")
plt.show()

# plot the error as function of the volume fraction
error = (model.predict(X_test_scaled) - np.ravel(y_test_scaled))**2
ax = plt.scatter(min_max_scalerX.inverse_transform(X_test_scaled)[:, -2], error, s=1)
plt.xlabel("volume fraction")
plt.ylabel("error")
plt.show()

