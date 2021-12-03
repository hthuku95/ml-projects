import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

#num words picks only the first 10000 most occuring || relevant words
(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words=10000)


#returns a list of the integer encoding of the words
print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

'''
for us to shape the data, we must seet a max limit of words for each movie review
since some reviews might contain more words than others, we need to shape the data by
making every review have the same length, we do this by setting a maxlen for our data
keras already has an inbuilt function for this
'''
test_data = keras.preproccesing.sequences.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=256)
train_data = keras.preproccesing.sequences.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=256)
def decode_review(text):
    return "".join([reverse_word_index.get(i,"?") for i in text])

'''
WORD VECTORS
lets say we have two similar sentences 
"Have a great day"
"Have a good day"

We as humans know the two sentences have similar meanings and they also have similar context.
But  computer cant read words, it would read the sntences as list of integers which encode each word.
It woul look something like this
[0,1,2,3]
[0,1,4,3]
The computer will read 2 and 4 as different words of different meanings hence it might treat the two
sentences differently and yet they have the same meanings.
We therefore need a way of letting the computer know the similarity of two words with different encodings

We randomly assign each integer encoding a vector then move the vectors closer to each other if the 
two words are similar. in the case of our two sentences above, the two words share a similar context 
therefore the computer will look at the other words in the sentence when determinnig teir vectors,
and in this case, the computer will assign the two words vectors that are close to each other.The groupings 
done by the algorithms are way more complex than this method.

inshort the embading layer takes the input and spits out the corresponding vectors to 
the next layer in the network.The next layer in the network is the first dense layer and it takes
in data inform of average representation of the vectors from the embading layer.

the vectors are then passed to the next layer in the network hence solving the similarity problem.

n/b, the word vectors are assigned in the embading layer of the neural netwok.
'''

# Building the model
model = keras.Sequential()
# converts the word encodings to their 16 dimension vector representations
model.add(keras.layers.Embedding(10000,16))
# reduces the 16Ds into a 1D average dimension representation and passes the data to the next dense layer
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))


#MODEL SETTINGS
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

x_val =train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train,y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)

results = model.evaluate(test_data,test_labels)
print(results)

'''
#saving the model
model.save("IMDBMovieClassiffier.h5")


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded
#loading saved model
model = keras.models.load_model("IMDBMovieClassiffier.h5")

'''

'''
Using the saved model

with open("test.txt") as f:
    for line in f.readlines():
        #purifying and encoding the data for the model to understand
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","")
        encode = review_encode(nline)

        encode = keras.preproccesing.sequences.pad_sequences([encode],value=word_index["<PAD>"],padding="post",maxlen=256)
        predict = model.predict(encode)
        
        print(nline)
        print(encode)
        print(predict)
'''