#import libraries
from PIL import Image
import numpy as np
import tensorflow.keras.preprocessing.image

def encodeImage(img,WIDTH,HEIGHT,preprocess_input,encode_model,OUTPUT_DIM):
    """
    Function to encode the images
    :param img: input image
    :param WIDTH: width of the image
    :param HEIGHT: height of the image
    :param preprocess_input: tensorflow function to preprocess the image
    :param encode_model: our caption model
    :param OUTPUT_DIM: output dimension
    :return: encoded image
    """
    # Resize all images to a standard size (specified bythe image 
    # encoding network)
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    # Convert a PIL image to a numpy array
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    # Expand to 2D array
    x = np.expand_dims(x, axis=0)
    # Perform any preprocessing needed by InceptionV3 or others
    x = preprocess_input(x)
    # Call InceptionV3 (or other) to extract the smaller feature set for 
    # the image.
    x = encode_model.predict(x) # Get the encoding vector for the image
    # Shape to correct form to be accepted by LSTM captioning network.
    x = np.reshape(x, OUTPUT_DIM )
    return x

