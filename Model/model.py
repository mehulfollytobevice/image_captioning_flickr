#import libraries
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import add
from tensorflow.keras.layers import (LSTM, Embedding, 
    TimeDistributed, Dense, RepeatVector, 
    Activation, Flatten, Reshape, concatenate,  
    Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def create_model(OUTPUT_DIM,vocab_size,embedding_dim,max_length,embedding_matrix):
    """
    :param OUTPUT_DIM: the output dimension 
    :param vocab_size: the size of the vocabulary
    :param embedding_dim: the dimension of the embedding matrix
    :param max_length: the maximum length of a caption
    :return: caption model is returned
    """
    inputs1 = Input(shape=(OUTPUT_DIM,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    cap_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    cap_model.layers[2].set_weights([embedding_matrix])
    cap_model.layers[2].trainable = False
    cap_model.compile(loss='categorical_crossentropy', optimizer='adam')
    return cap_model

def generateCaption(photo,caption_model,max_length,wordtoidx,idxtoword,START,STOP):

    """
    Function to return the caption after the prediction
    :param photo: input image
    :param caption_model: our caption model
    :param max_length: maximum length of a sequence
    :param wordtoidx: word to index lookup table
    :param idxtoword: index to word lookup table
    :param START:starting token
    :param STOP: stopping token
    :return: caption generated from the model
    """
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def create_embedding_matrix(vocab_size,embedding_dim,embeddings_index,wordtoidx):
    """
    Function to create the embedding matrix from Glove
    :param vocab_size: size of the vocabulary
    :param embedding_dim: dimension of the embedding matrix to be created
    :param embeddings_index: index of loaded glove embeddings
    :param wordtoidx:word to index lookup table
    :return: embedding matrix
    """
    #intialize embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoidx.items():
        #if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix