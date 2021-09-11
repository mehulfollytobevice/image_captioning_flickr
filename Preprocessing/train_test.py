def train_test(root_captioning,img):
    """
    Function to return the list of training and test images
    :param root_captioning: path where the captions are placed
    :param img: glove embeddings
    :return: list of training images, list of test images
    """
    #load train images text file
    train_images_path = os.path.join(root_captioning,\
                'Flickr_8k.trainImages.txt') 
    train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
    
    #load test images test file
    test_images_path = os.path.join(root_captioning,
                'Flickr_8k.testImages.txt') 
    test_images = set(open(test_images_path, 'r').read().strip().split('\n'))

    #intialise lists
    train_img=[]
    test_img=[]

    for i in img:
        f = os.path.split(i)[-1]
        if f in train_images: 
            train_img.append(f) 
        elif f in test_images:
            test_img.append(f)

    return train_img,test_img,train_images,test_images

def train_test_generator(root_captioning,img_list,encodeImage,OUTPUT_DIM,HEIGHT,WIDTH,preprocess_input,encode_model,hms_string,training=True):
    """
    This function loops over every image and preprocesses it to generate a training set or a test set
    :param root_captioning: root path where the data is stored
    :param img_list: list of image filenames 
    :param encodeImage: function to encode a given image
    :param OUTPUT_DIM: the output dimension
    :param HEIGHT: height of the image
    :param WIDTH: width of the image
    :param preprocess_input: tensorflow function to preprocess the input
    :param encode_model: the caption model 
    :param training: True for training set and False for test set
    :param hms_string: function to capture the time taken
    :return: the encoded training or test set is returned 
    """
    if training:
        PATH = os.path.join(root_captioning,"data",f'train{OUTPUT_DIM}.pkl')
    else:
        PATH = os.path.join(root_captioning,"data",f'test{OUTPUT_DIM}.pkl')
    if not os.path.exists(PATH):
        start = time()
        encoding_set = {}
        for id in tqdm(img_list):
            image_path = os.path.join(root_captioning,'Flicker8k_Dataset', id)
            img = tensorflow.keras.preprocessing.image.load_img(image_path, \
                    target_size=(HEIGHT, WIDTH))
            encoding_set[id] = encodeImage(img,WIDTH,HEIGHT,preprocess_input,encode_model,OUTPUT_DIM)
        with open(PATH, "wb") as fp:
            pickle.dump(encoding_set, fp)

        if training:
            print(f"\nGenerating training set took: {hms_string(time()-start)}")
        else:
            print(f"\nGenerating testing set took: {hms_string(time()-start)}")
    else:
        with open(PATH, "rb") as fp:
            encoding_set = pickle.load(fp)
    return encoding_set
