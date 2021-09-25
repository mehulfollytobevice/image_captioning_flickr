## Image Captioning using Flickr8K 

- This repository contains the code for **"Image Captioning System using Flickr8K dataset"**.
- Using this project, we can generate captions for images automatically.
- This project uses deep learning to accomplish the aformentioned task.
  
## 📝 Description
- Image Captioning is the process of generating a textual description for a given image. It has been a very important and fundamental task in the domain of Deep Learning and has a huge number of applications. For instance, image captioning technologies can be used to create an application to help people who have low or no eyesight to gather information about the world around them. 
- Our approach for creating an image captioning system is to use transfer learning and re-purpose the existing knowledge of the pre-trained model to generate captions for the input images. 
- In this project we have used the **Flickr8K dataset** to train our model and the **InceptionV3 model** as our pre-trained model. Furthermore, we have also utilised **GloVe** embeddings to augment the predictive performance of our image captioning model. 

## ⏳ Dataset
- There are different versions of the Flickr Dataset. We use the Flickr8K dataset which has 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events.
- Download from here: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
- Download the text descriptions: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
- Download the dataset and place it in the main directory.


## 🏽‍ Download GloVe Embeddings 
- Download the GloVE embeddings from here : https://nlp.stanford.edu/projects/glove/
- Download the embeddings file and place it in the main directory.

##  🏽‍ For Image Captioning, download our model
- For image captioning, download our model manually: **" caption-model.hdf5 "** from following place in our repository 
- https://github.com/mehulfollytobevice/image_captioning_flickr/tree/main/data
- Download the file and place it into **" ./data/ ".** folder.

## :hammer_and_wrench: Requirements
* Python 3.5+
* voila
* tensorflow
* pillow<7
* packaging
* ipywidgets==7.5.1
* Linux

## Contributors <img src="https://raw.githubusercontent.com/TheDudeThatCode/TheDudeThatCode/master/Assets/Developer.gif" width=35 height=25> 
- Mehul Jain
- Manas Suryabhan Patil
- Ankit Jain

