import torch
import signatory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from InvertSignatorySignatures import invert_signature
import gc
import matplotlib.pyplot as plt

gc.collect()
torch.cuda.empty_cache() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth = 10

# File paths for training and testing sets
train_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\train_stock_data.csv"
test_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\test_stock_data.csv"

# Gets training and testing data
traindf = pd.read_csv(train_path)
testdf = pd.read_csv(test_path)

# Performs transpose and adds batch dimension for signature
train_data = torch.unsqueeze(torch.tensor(traindf.values), 0)
test_data = torch.unsqueeze(torch.tensor(testdf.values), 0)

print(train_data.shape)

# Initiating empty list which will hold all signatures
train_signature_data = []
test_signature_data = []

# Getting dimensions of data to be looped over
n = train_data.shape[1]
print(n)
m = test_data.shape[1]
print(m)
num_samples = int(n/depth)

tempdf = train_data[:,0:depth,:]

signature = signatory.signature(path = tempdf, depth = depth)

inverse = invert_signature(signature=signature, depth=depth, channels = 2)[:,1:,]

for i in range(num_samples):
    temp_data = train_data[:,i*depth:(i+1)*depth,:]
    temp_signature = signatory.signature(path = temp_data, depth = depth)
    train_signature_data.append(temp_signature)

for j in range(int(m/depth)):
    temp_data = test_data[:,j*depth:(j+1)*depth,:]
    temp_signature = signatory.signature(path = temp_data, depth = depth)
    test_signature_data.append(temp_signature)

print("finished")


train_signature_data = torch.stack(train_signature_data, dim = 0)
test_signature_data = torch.stack(test_signature_data, dim = 0)

'''
torch.save(train_signature_data, 'train_sig_data.pt')
torch.save(test_signature_data, 'test_sig_data.pt')
loaded_tensor = torch.load('train_sig_data.pt')
print(loaded_tensor.shape)
print(loaded_tensor[0,0,:])
'''


# for i in range(int(loaded_tensor.shape[2]/4)):
#     print(loaded_tensor[0,0,i])
#print(train_signature_data.shape)
