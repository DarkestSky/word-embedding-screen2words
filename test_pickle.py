import pickle
import torch
import numpy as np

pkl_file = 'result/screen_summary.pkl'

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)
    # print(type(data))
    # print(data['vector'][0])
    data = data.query("screenId == 13974")
    # print(data)
    data = data.iloc[0]['vector']
    # data = np.array(data)
    print(data)
    print(torch.tensor(data, dtype=torch.float32))
    # print(len(data[0]['vector'][0]))
    