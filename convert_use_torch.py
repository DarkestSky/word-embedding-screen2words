import pandas as pd
import numpy as np
import pickle
import torchtext
import torch

glove_data_path = 'glove_data/'
output_data_path = 'result/'
word_dim = 300    # keep the same dimension as the pre-trained glove file
screen_summary_file = 'screen_summaries/screen_summaries_fixed.csv'

pkl_output_file = output_data_path + 'screen_summary_torchtext.pkl'

glove = torchtext.vocab.GloVe(name="840B", dim=300)
    
data = pd.read_csv(screen_summary_file)

error_count = 0;
max_length = 0;

examples = pd.DataFrame(columns=['screenId', 'vector'])
for index, row in data.iterrows():
    max_length = max(max_length, len(row['summary'].split()))
    gen_token = []
    for word in row['summary'].split(' '):
        try:
            gen_token.append(glove[word])
        except:
            # Some words may not be able to convert correctly, just ignore them
            gen_token.append(torch.tensor([0] * word_dim))
            error_count += 1
            # print(index, word)
    gen_token = torch.stack(gen_token)
    new_example = {'screenId': row['screenId'], 'vector': [gen_token]}
    new_example = pd.DataFrame(new_example)
    examples = pd.concat([examples, new_example], ignore_index=True)

# save in pkl file for further use in the model
with open(pkl_output_file, 'wb') as f:
    pickle.dump(examples, f)
    