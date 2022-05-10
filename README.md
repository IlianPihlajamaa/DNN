# Mode coupling theory with a deep neural network

This repository contains a trained deep neural network that can reproduce generalized mode coupling theory results with a high accuracy. 
It was trained on a data set of machine generated structure factors and their full GMCT solutions for a randomly chosen order 1-5. The model was trained using PyTorch.

The data on which it is trained can be found at this link: ....
That repository also contains the code to solve the GMCT equations.

To run the model, put the contents of the data in the folders "Data" and "Data 2" and run DN_pytorch.py.

The model requires 68 input parameters. 
 - the first 64 are the discretised structure factors on the grid given by "k_array.txt" which resides in the Data folders. 
 - the 65th input parameter is 1/order, where order is the last level of GMCT that is taken into account
 - the 66th input parameter is the number density
 - the 67th input parameter is log10(t)
 - the 68th input parameter is k, not necessarily on the grid at which the structure factor was sampled.
 
 It outputs the intermediate scattering function F(k, t)