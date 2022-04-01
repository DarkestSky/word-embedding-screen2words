from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

glove_data_path = 'glove_data/'
output_data_path = 'result/'
glove_filename = 'glove.42B.300d.txt'
word_dim = 300    # keep the same dimension as the pre-trained glove file
screen_summary_file = 'screen_summaries/screen_summaries_fixed.csv'

word2vec_output_file = output_data_path + glove_filename + '.word2vec'
pkl_output_file = output_data_path + 'screen_summary.pkl'
glove_filename = glove_data_path + glove_filename

'''
    We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format 
    that will be loaded in the next section
'''
# glove2word2vec(glove_filename, word2vec_output_file)

# load the Stanford GloVe model
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    
data = pd.read_csv(screen_summary_file)

error_count = 0;
max_length = 0;

examples = pd.DataFrame(columns=['screenId', 'vector'])
for index, row in data.iterrows():
    max_length = max(max_length, len(row['summary'].split()))
    gen_token = []
    for word in row['summary'].split(' '):
        try:
            gen_token.append(model.get_vector(word))
        except:
            # Some words may not be able to convert correctly, just ignore them
            gen_token.append([0] * word_dim)
            error_count += 1
            # print(index, word)
    gen_token = np.array(gen_token)
    new_example = {'screenId': row['screenId'], 'vector': [gen_token]}
    new_example = pd.DataFrame(new_example)
    examples = pd.concat([examples, new_example], ignore_index=True)

# save in pkl file for further use in the model
with open(pkl_output_file, 'wb') as f:
    pickle.dump(examples, f)
    