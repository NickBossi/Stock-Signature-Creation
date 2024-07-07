import pandas as pd 
import numpy as np
import torch

#Loading csv's

apple_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\AAPL.csv"
microsoft_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\MSFT.csv"
file_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\stock_data.csv"

appledf = pd.read_csv(apple_path)
microdf = pd.read_csv(microsoft_path)

print(microdf.shape)

#Truncating apple data rows to ensure same length as microsoft data
appledf = appledf.tail(microdf.shape[0])

#Extracting only the opening price from the stocks
applecolumn = appledf['Open'].values
microcolumn = microdf['Open'].values

file = pd.DataFrame(np.stack((applecolumn, microcolumn), axis=1), columns=['apple', 'microsoft'])
#file.to_csv(file_path, index=False)

#creating a torch tensor of the stocks
path = torch.from_numpy(np.stack((applecolumn, microcolumn), axis=1))


#path = torch.tensor([applecolumn, microcolumn])
# def getpath():
#     return path