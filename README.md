# Image Captioning using Flickr8K 

- This repository contains the code for **"Image Captioning System using Flickr8K dataset"**.
- Using this project, we can generate captions for images automatically.
- This project uses deep learning to accomplish the above mentioned task.

## Some Examples

![pic1](https://github.com/mehulfollytobevice/image_captioning_flickr/blob/main/pics/pic1.png)

![pic2](https://github.com/mehulfollytobevice/image_captioning_flickr/blob/main/pics/pic2.png)

![pic3](https://github.com/mehulfollytobevice/image_captioning_flickr/blob/main/pics/pic3.png)

![pic4](https://github.com/mehulfollytobevice/image_captioning_flickr/blob/main/pics/pic4.png)


  
## 📝 Description
- Image Captioning is the process of generating a textual description for a given image. It has been a very important and fundamental task in the domain of Deep Learning and has a huge number of applications. For instance, image captioning technologies can be used to create an application to help people who have low or no eyesight to gather information about the world around them. 
- Our approach for creating an image captioning system is to use transfer learning and re-purpose the existing knowledge of the pre-trained model to generate captions for the input images. 
- In this project we have used the **Flickr8K dataset** to train our model and the **InceptionV3 model** as our pre-trained model. Furthermore, we have also utilised **GloVe** embeddings to augment the predictive performance of our image captioning model. 

## ⏳ Dataset
- There are different versions of the Flickr Dataset. We use the Flickr8K dataset which has 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events.
- Download from here: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
- Download the text descriptions: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
- Download the dataset and place it in the main directory.

## 📝 Description of files:

- <b>main.py -</b> This is the main file of the project. It contains the code to train the image captioning model and save it into the '/data/' directory.
- <b>image_captioner.ipynb -</b> This file contains the code for our image captioning application. The application is built using ipywidgets and Voila. This notebook can be directly deployed onto Binder where it gets converted into an interactive application. For more information, see the next section.
- <b>requirements.txt -</b> This file contains the dependencies required to run the image captioning application (image_captioner.ipynb).

## :monocle_face: Test our project
To test our project using a friendly GUI, follow these steps:
1. Go to https://mybinder.org/
2. Paste the URL of this repo into Binder’s URL field.
3. Change the File drop-down to instead select URL.
4. In the “URL to open” field, enter “/voila/render/image_captioner.ipynb”
5. Click the clipboard button at the bottom right to copy the URL and paste it somewhere safe.
6. Click Launch.
7. Wait a while for the application to launch. Afterwards, you can play around with it and share the link with your friends.

It should look like this:

![binder_pic](https://github.com/mehulfollytobevice/image_captioning_flickr/blob/main/pics/binder.PNG)

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

## Photo Credits:
- Photo 1 by <a href="https://unsplash.com/@imandrewpons?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Andrew Pons</a> on <a href="https://unsplash.com/s/photos/dog-playing-with-ball?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
- Photo 2 by <a href="https://unsplash.com/@chrishcush?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Christian Bowen</a> on <a href="https://unsplash.com/s/photos/kids-beach?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
- Photo 3 by <a href="https://unsplash.com/@kaeptn?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Kay Liedl</a> on <a href="https://unsplash.com/s/photos/biking?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
- Photo 4 by <a href="https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Markus Spiske</a> on <a href="https://unsplash.com/s/photos/basketball-nba?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
