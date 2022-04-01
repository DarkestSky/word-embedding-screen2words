from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_data_path = 'glove_data/'
output_data_path = 'result/'
glove_filename = 'glove.42B.300d.txt'

word2vec_output_file = output_data_path + glove_filename + '.word2vec'
glove_filename = glove_data_path + glove_filename

'''
    We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format 
    that will be loaded in the next section
'''
# glove2word2vec(glove_filename, word2vec_output_file)

# load the Stanford GloVe model
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# #Show a word embedding
print('King: ',model.get_vector('king'))
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print('Most similar word to King + Woman: ', result)

result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print('King - Man + Woman = ',result)
result = model.most_similar(positive=['rome', 'france'], negative=['paris'], topn=1)
print('France - Paris + Rome = ',result)
result = model.most_similar(positive=['english', 'france'], negative=['french'], topn=1)
print('France - french + english = ',result)
result = model.most_similar(positive=['june', 'december'], negative=['november'], topn=1)
print('December - November + June = ',result)
result = model.most_similar(positive=['sister', 'man'], negative=['woman'], topn=1)
print('Man - Woman + Sister = ',result)
