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


def main():

    depth = 10

    # File paths for training and testing sets
    train_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\train_stock_data.csv"
    test_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\test_stock_data.csv"

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
    rolling_train_data = []

    # Getting dimensions of data to be looped over
    n = train_data.shape[1]
    print(n)
    m = test_data.shape[1]

    num_samples = int(n/depth)

    # Testing on first 10 days of data
    tempdf = train_data[:,0:depth,:]
    signature = signatory.signature(path = tempdf, depth = depth)
    inverse = invert_signature(signature=signature, depth=depth, channels = 2)[:,1:,]

    for i in range(n-depth+1):
        temp_data = train_data[:,i:i+10,:]
        temp_signature = signatory.signature(path = temp_data, depth = depth)
        rolling_train_data.append(temp_signature)


    for i in range(num_samples):
        temp_data = train_data[:,i*depth:(i+1)*depth,:]
        temp_signature = signatory.signature(path = temp_data, depth = depth)
        train_signature_data.append(temp_signature)

    for j in range(int(m/depth)):
        temp_data = test_data[:,j*depth:(j+1)*depth,:]
        temp_signature = signatory.signature(path = temp_data, depth = depth)
        test_signature_data.append(temp_signature)

    print("finished")

    # Converting to tensors
    train_signature_data = torch.stack(train_signature_data, dim = 0)
    test_signature_data = torch.stack(test_signature_data, dim = 0)
    rolling_train_data = torch.stack(rolling_train_data, dim = 0)

    # Ensuring all the data is captured by the rolling window
    print(rolling_train_data[0])
    print(train_signature_data[0])
    print(rolling_train_data[-1])
    print(train_signature_data[-1])
    print(rolling_train_data.shape)

    torch.save(train_signature_data, '../VAE/data/train_sig_data.pt')
    torch.save(test_signature_data, '../VAE/data/test_sig_data.pt')
    torch.save(rolling_train_data, '../VAE/data/rolling_train_data.pt')
    

if __name__ == "__main__":
    main()
