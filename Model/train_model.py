def train_nn(EPOCHS,caption_model,data_generator,train_descriptions,encoding_train,wordtoidx,max_length,number_pics_per_batch,steps,model_path,vocab_size,**kwargs):
    """
    Function to train our neural network
    :param EPOCHS: number of epochs
    :param caption_model: our caption model
    :param data_generator: data generator function
    :param train_descriptions: training set descriptions of the images
    :param encoding_train: encoding training set
    :param wordtoidx: word to index lookup table
    :param max_length: maximum length of a sequence
    :param number_pics_per_batch: number of images in a batch
    :param steps: number of steps per epoch
    :param model_path: path where the model should be saved after training
    :param vocab_size: size of the vocabulary
    :return: trained image captioning model
    """
    for i in tqdm(range(EPOCHS*2)):
        generator = data_generator(train_descriptions, encoding_train, 
                        wordtoidx, max_length, number_pics_per_batch,vocab_size)
        caption_model.fit_generator(generator, epochs=1,steps_per_epoch=steps, verbose=1)

    caption_model.optimizer.lr = 1e-4
    number_pics_per_bath = 6
    steps = len(train_descriptions)//number_pics_per_bath

    for i in range(EPOCHS):
        generator = data_generator(train_descriptions, encoding_train, 
                        wordtoidx, max_length, number_pics_per_bath,vocab_size)
        caption_model.fit_generator(generator, epochs=1, 
                                steps_per_epoch=steps, verbose=1)  
    caption_model.save_weights(model_path)
    if "hms_string" in kwargs.keys():
        print(f"\Training took: {kwargs['hms_string'](time()-kwargs['start'])}")
    else:
        logging.debug('No function is given for measuring time.')
    
    return caption_model