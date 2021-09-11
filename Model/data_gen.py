def data_generator(descriptions, photos, wordtoidx,max_length, num_photos_per_batch,vocab_size):
    """
    Function to duplicate image for each partial caption
    :param descriptions: training set descriptions of the images
    :param photos: input images 
    :param wordtoidx: word to index lookup table
    :param max_length: maximum length of a caption
    :param num_photos_per_batch: number of images in a batch
    :param vocab_size: size of the vocabulary  
    """

    # x1 - Training data for photos
    # x2 - The caption that goes with each photo
    # y - The predicted rest of the caption
    x1, x2, y = [], [], []
    n=0
    while True:
        for key, desc_list in descriptions.items():
            n+=1
            photo = photos[key+'.jpg']
        # Each photo has 5 descriptions
            for desc in desc_list:
                # Convert each word into a list of sequences.
                seq = [wordtoidx[word] for word in desc.split(' ') \
                    if word in wordtoidx]
            # Generate a training case for every possible sequence and outcome
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    x1.append(photo)
                    x2.append(in_seq)
                    y.append(out_seq)
            if n==num_photos_per_batch:
                yield ([np.array(x1), np.array(x2)], np.array(y))
                x1, x2, y = [], [], []
                n=0