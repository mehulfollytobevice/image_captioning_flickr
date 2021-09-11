def caption_cleaning(root_captioning):
    """
    Function to clean captions and remove extra whitespace, punctuation, and other distractions
    :param root_captioning:  path where the captions stored
    :return: number of unique words, the dictionary, max length of a caption 
    """

    null_punct = str.maketrans('', '',string.punctuation)
    lookup = dict()

    #open the captions file
    with open( os.path.join(root_captioning,\
        'Flickr8k.token.txt'), 'r') as fp:
        max_length=0
        for line in fp.read().split('\n'):
            tok=line.split()
            if len(line)>=2:
                id=tok[0].split('.')[0]
                desc=tok[1:]

                # Cleanup description
                desc = [word.lower() for word in desc]
                desc = [w.translate(null_punct) for w in desc]
                desc = [word for word in desc if len(word)>1]
                desc = [word for word in desc if word.isalpha()]
                max_length = max(max_length,len(desc))

                if id not in lookup:
                    lookup[id] = list()
                lookup[id].append(' '.join(desc))
    
    lex=set()
    for key in lookup:
        [lex.update(d.split()) for d in lookup[key]]

    return lookup,lex,max_length

def train_desc(lookup,train_images,START,STOP):
    """
    Function to add START and STOP token to the train, test images
    :param lookup: how many unique words are there
    :param train_images: images training set 
    :param START: sequence starting token
    :param STOP: sequence ending token
    :return: training descriptions
    """
    #list of descriptions for the training set
    train_descriptions = {k:v for k,v in lookup.items() if f'{k}.jpg' \
                        in train_images}

    #looping over the descriptions and adding START and STOP tokens
    for n,v in train_descriptions.items(): 
        for d in range(len(v)):
            v[d] = f'{START} {v[d]} {STOP}'

    return train_descriptions