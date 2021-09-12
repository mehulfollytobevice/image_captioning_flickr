#importing essentials
import glob
import logging
import numpy as np
import datetime 
import pytz 
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.applications.mobilenet  
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers
from tqdm import tqdm


# using now() to get current time 
current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')) 
logging.info(f'The time at which this script was run {current_time}')

#some initial settings 
START = "startseq"
STOP = "endseq"
EPOCHS = 10
USE_INCEPTION = True
logging.basicConfig(filename='image_captioning.log', encoding='utf-8', level=logging.INFO)
logging.info("All essential libraries imported")

if __name__=="__main__":
  
  #path where data is stored 
  root_captioning="./data/Image_Captioning"
  logging.info('Root path is set.')

  #cleaning the captions in the Flickr dataset
  from Preprocessing.captions_preprocessing import caption_cleaning
  lookup,lex,max_length= caption_cleaning(root_captioning=root_captioning)
  logging.info('Captions are cleaned.')

  #loading the image files
  img = glob.glob(os.path.join(root_captioning,'Flicker8k_Dataset', '*.jpg'))
  logging.info('The image files are loaded.')

  #read image names for the train and test set
  from Preprocessing.train_test import train_test
  train_img,test_img,train_images,test_images=train_test(root_captioning,img)
  logging.info('Filenames for the train and the test set are read.')

  #building the sequences
  from Preprocessing.captions_preprocessing import train_desc
  train_descriptions=train_desc(lookup,train_images,START,STOP)
  logging.info('Training sequences are built.')

  #settings for model building
  if USE_INCEPTION:
    encode_model = InceptionV3(weights='imagenet')
    encode_model = Model(encode_model.input, encode_model.layers[-2].output)
    WIDTH = 299
    HEIGHT = 299
    OUTPUT_DIM = 2048
    preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
    logging.info('Using Inception pre-trained model.')
  else:
    encode_model = MobileNet(weights='imagenet',include_top=False)
    WIDTH = 224
    HEIGHT = 224
    OUTPUT_DIM = 50176
    preprocess_input = tensorflow.keras.applications.mobilenet.preprocess_input
    logging.info('Using MobileNet pre-trained model.')

  #generate training set and test set 
  from Preprocessing.train_test import train_test_generator
  from Preprocessing.image_preprocessing import encodeImage
  from Utils.capture_time import capture
  encoding_train=train_test_generator(root_captioning=root_captioning
                                      ,img_list=train_img
                                      ,encodeImage=encodeImage
                                      ,OUTPUT_DIM=OUTPUT_DIM
                                      ,HEIGHT=HEIGHT,
                                      WIDTH=WIDTH,
                                      preprocess_input=preprocess_input,
                                      encode_model=encode_model,hms_string=capture)
  logging.info('Training set is generated.')
  encoding_test=train_test_generator(root_captioning=root_captioning
                                      ,img_list=train_img
                                      ,encodeImage=encodeImage
                                      ,OUTPUT_DIM=OUTPUT_DIM
                                      ,HEIGHT=HEIGHT,
                                      WIDTH=WIDTH,preprocess_input=preprocess_input,
                                      encode_model=encode_model,hms_string=capture,training=False)
  logging.info('Testing set is generated.')

  #seperate captions that are going to be used for training
  all_train_captions = []
  for key, val in train_descriptions.items():
      for cap in val:
          all_train_captions.append(cap)

  #removing words that do not occur very often
  word_count_threshold = 10
  word_counts = {}
  nsents = 0
  for sent in all_train_captions:
      nsents += 1
      for w in sent.split(' '):
          word_counts[w] = word_counts.get(w, 0) + 1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

  #creating lookup tables. index to word and word to index
  idxtoword = {}
  wordtoidx = {}
  ix = 1
  for w in vocab:
      wordtoidx[w] = ix
      idxtoword[ix] = w
      ix += 1
  logging.info('Lookup tables are created.')

  #we added START and STOP tokens to all our sequences, this increases the max_length by 2 
  vocab_size = len(idxtoword) + 1 
  print("Size of the vocabulary",vocab_size)
  max_length +=2

  #loading Glove embeddings
  glove_dir = os.path.join(root_captioning,'glove.6B')
  embeddings_index = {} 
  f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
  for line in tqdm(f):
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  print(f'Found {len(embeddings_index)} word vectors.')
  
  #building an embedding matrix from Glove
  from Model.model import create_embedding_matrix
  embedding_dim=200
  embedding_matrix=create_embedding_matrix(vocab_size,embedding_dim,embeddings_index,wordtoidx)
  logging.info('Embedding matrix is created.')

  #creating our neural network 
  from Model.model import create_model
  caption_model=create_model(OUTPUT_DIM,vocab_size,embedding_dim,max_length,embedding_matrix)
  logging.info('Model architecture is compiled.')

  #some settings for training
  number_pics_per_batch = 3
  steps = len(train_descriptions)//number_pics_per_batch
  #path of the pre-trained model
  model_path = os.path.join(root_captioning,"data",f'caption-model.hdf5')
  if not os.path.exists(model_path):
    #training the neural network 
    from Model.train_model import train_nn
    from Model.data_gen import data_generator
    caption_model=train_nn(EPOCHS=20,
                           caption_model=caption_model,
                           data_generator=data_generator,
                           train_descriptions=train_descriptions,
                           encoding_train=encoding_train,
                           wordtoidx=wordtoidx,
                           max_length=max_length,
                           number_pics_per_batch=number_pics_per_batch,steps=steps,
                           model_path=model_path,
                           vocab_size=vocab_size)
    logging.info('Our caption model is trained and ready to use.')
  else:
    #loading the pre-trained model
    caption_model.load_weights(model_path)
    logging.info('Our caption model is loaded from the memory.')