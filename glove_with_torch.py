import torch
import torchtext
import pandas as pd

# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="840B", # trained on Wikipedia 2014 corpus
                              dim=300)   # embedding size = 100

gen_token = []
gen_token.append(glove['cat'])
gen_token.append(glove['dog'])

gen_token = torch.stack(gen_token)

new_example = {'screenId': 114514, 'vector': [gen_token]}
new_example = pd.DataFrame(new_example)

data = new_example.iloc[0]['vector']
print(data)
