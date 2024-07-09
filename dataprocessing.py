import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt

#Loading csv's

apple_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\AAPL.csv"
microsoft_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\MSFT.csv"
file_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\stock_data.csv"
train_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\train_stock_data.csv"
test_path = r"d:\Documents\UCT\Honours\Honours_Project\Code\Stocks\data\test_stock_data.csv"

appledf = pd.read_csv(apple_path)
microdf = pd.read_csv(microsoft_path)

print(microdf.shape)

#Truncating apple data rows to ensure same length as microsoft data
appledf = appledf.tail(microdf.shape[0])
print(appledf.tail())


#Extracting only the opening price from the stocks
applecolumn = appledf['Open'].values
microcolumn = microdf['Open'].values

#Creating a dataframe of the stocks
file = pd.DataFrame(np.stack((applecolumn, microcolumn), axis=1), columns=['apple', 'microsoft'])

#Splitting the data into training and testing sets
test_stock_data = file.tail(30)
train_stock_data = file.iloc[3:-30]

#Plotting the stock data
plt.plot(file)
plt.show()

#Saving the data to csv files
file.to_csv(file_path, index=False)
train_stock_data.to_csv(train_path, index=False)
test_stock_data.to_csv(test_path, index=False)

