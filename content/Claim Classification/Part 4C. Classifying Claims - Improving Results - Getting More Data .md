Title: 4C. Classifying Claims - Improving Results with Data
Tags: improving_results
Authors: Ben Hoyle
Summary: This post follows on from part 4A and looks at how we can improve our initial results.

# 4. Classifying Claims - Improving Results - Getting More Data

## Getting More Data

Let's see what happens when we try 20,000 claims. 

---

At the moment the interface for getting the data is a little slow...


```python
# Now try 20,000 patent publications at random

import os, pickle

from collections import Counter
```


```python
# Get the claim 1 and classificationt text

PIK = "claim_and_class20k.data"

if os.path.isfile(PIK):
    with open(PIK, "rb") as f:
        print("Loading data")
        data = pickle.load(f)
        print("{0} claims and classifications loaded".format(len(data)))
else:
    from patentdata.corpus import USPublications
    from patentdata.models.patentcorpus import LazyPatentCorpus
    
    path = '/patentdata/media/SAMSUNG1/Patent_Downloads'
    ds = USPublications(path)
    
    lzy = LazyPatentCorpus(ds, sample_size=20000)
    
    data = list()
    for i, pd in enumerate(lzy):
        try:
            classifications = [c.as_string() for c in pd.classifications]
        except:
            classifications = ""
        try:
            claim1_text = pd.claimset.get_claim(1).text
        except:
            claim1_text = ""
        current_data = (claim1_text, classifications)
        data.append(current_data)
        if (i % 500) == 0:
            print("Saving a checkpoint at {0} files".format(i))
            print("Current data = ", pd.title)
            with open(PIK, "wb") as f:
                pickle.dump(data, f)
            
    with open(PIK, "wb") as f:
        pickle.dump(data, f)
        
    print("{0} claims saved".format(len(data)))
```

    Loading data
    19001 claims and classifications loaded


We got a memory error after 19000 but that is still more data than we had before. Let's use that. You can find the "claim_and_class20k.data" file in the GitHub directory if you want to load the data yourself.


```python
PIK = "raw_data20k.pkl"

if os.path.isfile(PIK):
    with open(PIK, "rb") as f:
        print("Loading data")
        data = pickle.load(f)
else:
    # Check for and remove 'cancelled' claims
    data = [d for d in data if '(canceled)' not in d[0]]
    cleaner_data = list()
    for d in data:
        if len(d[1]) >= 1:
            if len(d[1][0]) > 3:
                classification = d[1][0][2]
                cleaner_data.append(
                    (d[0], classification)
                )
    data = cleaner_data

    from patentdata.models.lib.utils import clean_characters

    data = [(clean_characters(d[0]), d[1]) for d in data]

    with open("raw_data20k.pkl", "wb") as f:
        pickle.dump(data, f)
```

    Loading data



```python
print("We have {0} data samples left after cleaning.".format(len(data)))
```

    We have 17712 data samples left after cleaning.



```python
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

Y_class = [d[1] for d in data]

# encode class values as integers
label_e = LabelEncoder()
label_e.fit(Y_class)
encoded_Y = label_e.transform(Y_class)
# convert integers to dummy variables (i.e. one hot encoded)
Y = to_categorical(encoded_Y)
print("Our classes are now a matrix of {0}".format(Y.shape))
print("Original label: {0}; Converted label: {1}".format(Y_class[0], Y[0]))

from keras.preprocessing.text import Tokenizer

docs = [d[0] for d in data]

# create the tokenizer
t = Tokenizer(num_words=5000)
# fit the tokenizer on the documents
t.fit_on_texts(docs)

X = t.texts_to_matrix(docs, mode='tfidf')

print("Our data has the following dimensionality: ", X.shape)
print("An example array is: ", X[0][0:100])
```

    Our classes are now a matrix of (17712, 8)
    Original label: G; Converted label: [0. 0. 0. 0. 0. 0. 1. 0.]
    Our data has the following dimensionality:  (17712, 5000)
    An example array is:  [0.         2.19509726 1.66470012 0.         1.21538639 1.66504906
     0.         0.88406116 0.         0.         1.66670622 0.69334485
     0.         0.75089151 0.         0.         1.47841041 1.27422143
     1.15174037 0.         0.         0.         0.         0.
     2.9039738  0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         1.93181596 0.         0.         0.         0.
     0.         0.         0.         0.         0.         2.27206512
     0.         1.95791029 0.         0.         2.3501247  0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         5.8289026  0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     4.79273481 0.         0.         0.         0.         0.
     0.         0.         0.         3.04409238 0.         0.
     0.         0.         2.44064762 0.        ]



```python
with open("encoded_data20k.pkl", "wb") as f:
    pickle.dump((X, Y), f)
```


```python
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

classifiers = [
    MultinomialNB(),
    MLPClassifier()
]

results = list()
# Convert one hot to target integer
Y_integers = numpy.argmax(Y, axis=1)

for clf in classifiers:
    name = clf.__class__.__name__
    scores = cross_val_score(clf, X, Y_integers, cv=5)
    results.append((
        name, 
        scores.mean()*100, 
        scores.std()*100
    ))
        
    print(
        "Classifier {0} has an average classification accuracy of {1:.2f} ({2:.2f})".format(
            name, 
            scores.mean()*100, 
            scores.std()*100
        )    
    )
```

Results:

* Classifier MultinomialNB has an average classification accuracy of 58.22 (0.34)
* Classifier MLPClassifier has an average classification accuracy of 62.38 (0.88)

So possibly a very small improvement but well within the realms of statistical variance. Let us see if more data helps with our keras model.


```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
```


```python
input_dim = X.shape[1]
print("Our input dimension for our claim representation is {0}".format(input_dim))

no_classes = Y.shape[1]
print("Our output dimension is {0}".format(no_classes))
```

    Our input dimension for our claim representation is 5000
    Our output dimension is 8



```python
def keras_model():
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=input_dim, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = keras_model()
model.summary()
history = model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=5, verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

    Our input dimension for our claim representation is 5000
    Our output dimension is 8
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 500)               2500500   
    _________________________________________________________________
    dense_2 (Dense)              (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 14169 samples, validate on 3543 samples
    Epoch 1/10
    14169/14169 [==============================] - 530s - loss: 1.1325 - acc: 0.6120 - val_loss: 1.0894 - val_acc: 0.6077
    Epoch 2/10
    14169/14169 [==============================] - 475s - loss: 0.3948 - acc: 0.8646 - val_loss: 1.4186 - val_acc: 0.6105
    Epoch 3/10
    14169/14169 [==============================] - 566s - loss: 0.1108 - acc: 0.9724 - val_loss: 1.9858 - val_acc: 0.6113
    Epoch 4/10
    14169/14169 [==============================] - 454s - loss: 0.0334 - acc: 0.9939 - val_loss: 2.3294 - val_acc: 0.6065
    Epoch 5/10
    14169/14169 [==============================] - 476s - loss: 0.0283 - acc: 0.9960 - val_loss: 2.5955 - val_acc: 0.6139
    Epoch 6/10
    14169/14169 [==============================] - 513s - loss: 0.0250 - acc: 0.9960 - val_loss: 2.8697 - val_acc: 0.6063
    Epoch 7/10
    14169/14169 [==============================] - 457s - loss: 0.0223 - acc: 0.9968 - val_loss: 3.2368 - val_acc: 0.5981
    Epoch 8/10
    14169/14169 [==============================] - 449s - loss: 0.0178 - acc: 0.9977 - val_loss: 3.3843 - val_acc: 0.5995
    Epoch 9/10
    14169/14169 [==============================] - 430s - loss: 0.0177 - acc: 0.9976 - val_loss: 3.4755 - val_acc: 0.6094
    Epoch 10/10
    14169/14169 [==============================] - 427s - loss: 0.0141 - acc: 0.9984 - val_loss: 3.5041 - val_acc: 0.6125
    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])



![png]({filename}/images/4C_output_13_1.png)



![png]({filename}/images/4C_output_13_2.png)


Our accuracy appears to have improved by 1-2%. This is still a small increase though, and possibly within the variance of our classifier. Let's try with regularisation, which had the best effect to prevent overfitting.


```python
from keras.regularizers import l2 # L2-regularisation
l2_lambda = 0.1
```


```python
def keras_reg_model():
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_reg_model()
model.summary()
history = model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=5, verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 500)               2500500   
    _________________________________________________________________
    dense_4 (Dense)              (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 14169 samples, validate on 3543 samples
    Epoch 1/10
    14169/14169 [==============================] - 1109s - loss: 3.5127 - acc: 0.5249 - val_loss: 2.8284 - val_acc: 0.5174
    Epoch 2/10
    14169/14169 [==============================] - 911s - loss: 2.5778 - acc: 0.5597 - val_loss: 2.4618 - val_acc: 0.5577
    Epoch 3/10
    14169/14169 [==============================] - 905s - loss: 2.2790 - acc: 0.5762 - val_loss: 2.1673 - val_acc: 0.5555
    Epoch 4/10
    14169/14169 [==============================] - 930s - loss: 2.1328 - acc: 0.5876 - val_loss: 2.1349 - val_acc: 0.5642
    Epoch 5/10
    14169/14169 [==============================] - 943s - loss: 2.0365 - acc: 0.5944 - val_loss: 2.0831 - val_acc: 0.5795
    Epoch 6/10
    14169/14169 [==============================] - 943s - loss: 1.9580 - acc: 0.5982 - val_loss: 2.0456 - val_acc: 0.5687
    Epoch 7/10
     1285/14169 [=>............................] - ETA: 866s - loss: 1.8422 - acc: 0.6661

Gvien our results so far - let's try a multilayer model with regularisation and aggressive dropout.


```python
def keras_best_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(input_dim,)))
    model.add(Dense(1000, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(0.25))
    model.add(Dense(250, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(0.25))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_best_model()
model.summary()
history = model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=5, verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dropout_3 (Dropout)          (None, 5000)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1000)              5001000   
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1000)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 500)               500500    
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 250)               125250    
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 250)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 8)                 2008      
    =================================================================
    Total params: 5,628,758
    Trainable params: 5,628,758
    Non-trainable params: 0
    _________________________________________________________________
    Train on 14169 samples, validate on 3543 samples
    Epoch 1/10
    14169/14169 [==============================] - 235s 17ms/step - loss: 4.1414 - acc: 0.3485 - val_loss: 2.2931 - val_acc: 0.3796
    Epoch 2/10
    14169/14169 [==============================] - 235s 17ms/step - loss: 2.3008 - acc: 0.3622 - val_loss: 2.2649 - val_acc: 0.3717
    Epoch 3/10
    14169/14169 [==============================] - 233s 16ms/step - loss: 2.2481 - acc: 0.3641 - val_loss: 2.1900 - val_acc: 0.3853
    Epoch 4/10
    14169/14169 [==============================] - 234s 17ms/step - loss: 2.2471 - acc: 0.3657 - val_loss: 2.1706 - val_acc: 0.3872
    Epoch 5/10
    14169/14169 [==============================] - 233s 16ms/step - loss: 2.2167 - acc: 0.3652 - val_loss: 2.1241 - val_acc: 0.3915
    Epoch 6/10
    14169/14169 [==============================] - 228s 16ms/step - loss: 2.1903 - acc: 0.3681 - val_loss: 2.1248 - val_acc: 0.3884
    Epoch 7/10
    14169/14169 [==============================] - 228s 16ms/step - loss: 2.1529 - acc: 0.3649 - val_loss: 2.0640 - val_acc: 0.3810
    Epoch 8/10
    14169/14169 [==============================] - 233s 16ms/step - loss: 2.1292 - acc: 0.3690 - val_loss: 2.0177 - val_acc: 0.3816
    Epoch 9/10
    14169/14169 [==============================] - 234s 17ms/step - loss: 2.1398 - acc: 0.3700 - val_loss: 2.0846 - val_acc: 0.3929
    Epoch 10/10
    14169/14169 [==============================] - 237s 17ms/step - loss: 2.1281 - acc: 0.3656 - val_loss: 2.0500 - val_acc: 0.3864
    dict_keys(['val_loss', 'val_acc', 'acc', 'loss'])



![png]({filename}/images/4C_output_18_1.png)



![png]({filename}/images/4C_output_18_2.png)


Maybe that's too aggressive on the regularisation...


```python
l2_lambda = 0.05

def keras_best_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.25, input_shape=(input_dim,)))
    model.add(Dense(1000, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(0.25))
    model.add(Dense(500, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_best_model()
model.summary()
history = model.fit(X, Y, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dropout_2 (Dropout)          (None, 5000)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1000)              5001000   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 1000)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 500)               500500    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 250)               125250    
    _________________________________________________________________
    dense_4 (Dense)              (None, 8)                 2008      
    =================================================================
    Total params: 5,628,758
    Trainable params: 5,628,758
    Non-trainable params: 0
    _________________________________________________________________
    Train on 14169 samples, validate on 3543 samples
    Epoch 1/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 7.0981 - acc: 0.4782 - val_loss: 2.0088 - val_acc: 0.4773
    Epoch 2/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.9643 - acc: 0.5087 - val_loss: 2.0833 - val_acc: 0.4423
    Epoch 3/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.9313 - acc: 0.5189 - val_loss: 1.8856 - val_acc: 0.5095
    Epoch 4/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.8617 - acc: 0.5285 - val_loss: 1.7967 - val_acc: 0.5202
    Epoch 5/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.8270 - acc: 0.5317 - val_loss: 1.8067 - val_acc: 0.5264
    Epoch 6/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.8229 - acc: 0.5298 - val_loss: 1.7852 - val_acc: 0.5275
    Epoch 7/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.8020 - acc: 0.5363 - val_loss: 1.7479 - val_acc: 0.5278
    Epoch 8/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7970 - acc: 0.5411 - val_loss: 1.7326 - val_acc: 0.5413
    Epoch 9/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7868 - acc: 0.5490 - val_loss: 1.7972 - val_acc: 0.5405
    Epoch 10/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.8063 - acc: 0.5497 - val_loss: 1.7729 - val_acc: 0.5741
    Epoch 11/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7935 - acc: 0.5521 - val_loss: 1.7277 - val_acc: 0.5543
    Epoch 12/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7793 - acc: 0.5564 - val_loss: 1.7212 - val_acc: 0.5628
    Epoch 13/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7716 - acc: 0.5566 - val_loss: 1.7344 - val_acc: 0.5648
    Epoch 14/100
    14169/14169 [==============================] - 40s 3ms/step - loss: 1.7912 - acc: 0.5646 - val_loss: 1.7609 - val_acc: 0.5459
    Epoch 15/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.8010 - acc: 0.5617 - val_loss: 1.7758 - val_acc: 0.5555
    Epoch 16/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7647 - acc: 0.5708 - val_loss: 1.7577 - val_acc: 0.5715
    Epoch 17/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7735 - acc: 0.5630 - val_loss: 1.7828 - val_acc: 0.5682
    Epoch 18/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7775 - acc: 0.5674 - val_loss: 1.6914 - val_acc: 0.5732
    Epoch 19/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7649 - acc: 0.5616 - val_loss: 1.7659 - val_acc: 0.5656
    Epoch 20/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7653 - acc: 0.5687 - val_loss: 1.7269 - val_acc: 0.5778
    Epoch 21/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7926 - acc: 0.5676 - val_loss: 1.7353 - val_acc: 0.5769
    Epoch 22/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.8137 - acc: 0.5681 - val_loss: 1.7812 - val_acc: 0.5964
    Epoch 23/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.8417 - acc: 0.5700 - val_loss: 1.8175 - val_acc: 0.5543
    Epoch 24/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7910 - acc: 0.5670 - val_loss: 1.7624 - val_acc: 0.5797
    Epoch 25/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7443 - acc: 0.5701 - val_loss: 1.7018 - val_acc: 0.5831
    Epoch 26/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7419 - acc: 0.5736 - val_loss: 1.7574 - val_acc: 0.5684
    Epoch 27/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7701 - acc: 0.5755 - val_loss: 1.7539 - val_acc: 0.5577
    Epoch 28/100
    14169/14169 [==============================] - 40s 3ms/step - loss: 1.7878 - acc: 0.5751 - val_loss: 1.7290 - val_acc: 0.5800
    Epoch 29/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7429 - acc: 0.5757 - val_loss: 1.7327 - val_acc: 0.5580
    Epoch 30/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7342 - acc: 0.5731 - val_loss: 1.6896 - val_acc: 0.5704
    Epoch 31/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7455 - acc: 0.5782 - val_loss: 1.6768 - val_acc: 0.5766
    Epoch 32/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7320 - acc: 0.5803 - val_loss: 1.6411 - val_acc: 0.5857
    Epoch 33/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7088 - acc: 0.5726 - val_loss: 1.6477 - val_acc: 0.5834
    Epoch 34/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7210 - acc: 0.5770 - val_loss: 1.6664 - val_acc: 0.5865
    Epoch 35/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7339 - acc: 0.5729 - val_loss: 1.7065 - val_acc: 0.5676
    Epoch 36/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7247 - acc: 0.5785 - val_loss: 1.7183 - val_acc: 0.5588
    Epoch 37/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7036 - acc: 0.5799 - val_loss: 1.6512 - val_acc: 0.5854
    Epoch 38/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6881 - acc: 0.5803 - val_loss: 1.6607 - val_acc: 0.5845
    Epoch 39/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7112 - acc: 0.5810 - val_loss: 1.6761 - val_acc: 0.5848
    Epoch 40/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6935 - acc: 0.5814 - val_loss: 1.6509 - val_acc: 0.5786
    Epoch 41/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7031 - acc: 0.5799 - val_loss: 1.6444 - val_acc: 0.5899
    Epoch 42/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7041 - acc: 0.5804 - val_loss: 1.6987 - val_acc: 0.5634
    Epoch 43/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7034 - acc: 0.5814 - val_loss: 1.6882 - val_acc: 0.5634
    Epoch 44/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7188 - acc: 0.5779 - val_loss: 1.6513 - val_acc: 0.5834
    Epoch 45/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7077 - acc: 0.5858 - val_loss: 1.6546 - val_acc: 0.5840
    Epoch 46/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6948 - acc: 0.5810 - val_loss: 1.6565 - val_acc: 0.5871
    Epoch 47/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7149 - acc: 0.5813 - val_loss: 1.6405 - val_acc: 0.5871
    Epoch 48/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7212 - acc: 0.5800 - val_loss: 1.6631 - val_acc: 0.5916
    Epoch 49/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7177 - acc: 0.5796 - val_loss: 1.7004 - val_acc: 0.5651
    Epoch 50/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7114 - acc: 0.5935 - val_loss: 1.6883 - val_acc: 0.5851
    Epoch 51/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7091 - acc: 0.5857 - val_loss: 1.6943 - val_acc: 0.5862
    Epoch 52/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7038 - acc: 0.5813 - val_loss: 1.6542 - val_acc: 0.5786
    Epoch 53/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6978 - acc: 0.5801 - val_loss: 1.6885 - val_acc: 0.5879
    Epoch 54/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6972 - acc: 0.5819 - val_loss: 1.6577 - val_acc: 0.5882
    Epoch 55/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7050 - acc: 0.5849 - val_loss: 1.7277 - val_acc: 0.5831
    Epoch 56/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7031 - acc: 0.5805 - val_loss: 1.7176 - val_acc: 0.5721
    Epoch 57/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7122 - acc: 0.5868 - val_loss: 1.6822 - val_acc: 0.5803
    Epoch 58/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7076 - acc: 0.5828 - val_loss: 1.6530 - val_acc: 0.5913
    Epoch 59/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7101 - acc: 0.5854 - val_loss: 1.6495 - val_acc: 0.5958
    Epoch 60/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7016 - acc: 0.5859 - val_loss: 1.6875 - val_acc: 0.5713
    Epoch 61/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6959 - acc: 0.5856 - val_loss: 1.6289 - val_acc: 0.5902
    Epoch 62/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7089 - acc: 0.5776 - val_loss: 1.6507 - val_acc: 0.5865
    Epoch 63/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6848 - acc: 0.5888 - val_loss: 1.6731 - val_acc: 0.5806
    Epoch 64/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7119 - acc: 0.5858 - val_loss: 1.6694 - val_acc: 0.5766
    Epoch 65/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6897 - acc: 0.5854 - val_loss: 1.6580 - val_acc: 0.5865
    Epoch 66/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6992 - acc: 0.5888 - val_loss: 1.6523 - val_acc: 0.5978
    Epoch 67/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6826 - acc: 0.5907 - val_loss: 1.6565 - val_acc: 0.5936
    Epoch 68/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6951 - acc: 0.5906 - val_loss: 1.6434 - val_acc: 0.5955
    Epoch 69/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6809 - acc: 0.5805 - val_loss: 1.6181 - val_acc: 0.5975
    Epoch 70/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6979 - acc: 0.5865 - val_loss: 1.6729 - val_acc: 0.5893
    Epoch 71/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7019 - acc: 0.5798 - val_loss: 1.7150 - val_acc: 0.5828
    Epoch 72/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7150 - acc: 0.5833 - val_loss: 1.6496 - val_acc: 0.5763
    Epoch 73/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6969 - acc: 0.5875 - val_loss: 1.6987 - val_acc: 0.5735
    Epoch 74/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6980 - acc: 0.5814 - val_loss: 1.6555 - val_acc: 0.5933
    Epoch 75/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7002 - acc: 0.5819 - val_loss: 1.6448 - val_acc: 0.5961
    Epoch 76/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6834 - acc: 0.5914 - val_loss: 1.6705 - val_acc: 0.5780
    Epoch 77/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6908 - acc: 0.5857 - val_loss: 1.6920 - val_acc: 0.5651
    Epoch 78/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6909 - acc: 0.5868 - val_loss: 1.6839 - val_acc: 0.5783
    Epoch 79/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6903 - acc: 0.5878 - val_loss: 1.6548 - val_acc: 0.5927
    Epoch 80/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6785 - acc: 0.5889 - val_loss: 1.6671 - val_acc: 0.5848
    Epoch 81/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6960 - acc: 0.5869 - val_loss: 1.6657 - val_acc: 0.5930
    Epoch 82/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6973 - acc: 0.5847 - val_loss: 1.6991 - val_acc: 0.5696
    Epoch 83/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6832 - acc: 0.5844 - val_loss: 1.6917 - val_acc: 0.5874
    Epoch 84/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6950 - acc: 0.5916 - val_loss: 1.6861 - val_acc: 0.5617
    Epoch 85/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.6990 - acc: 0.5835 - val_loss: 1.6539 - val_acc: 0.6060
    Epoch 86/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7008 - acc: 0.5859 - val_loss: 1.6889 - val_acc: 0.5792
    Epoch 87/100
    14169/14169 [==============================] - 38s 3ms/step - loss: 1.7040 - acc: 0.5835 - val_loss: 1.6696 - val_acc: 0.5761
    Epoch 88/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.7172 - acc: 0.5854 - val_loss: 1.6428 - val_acc: 0.5780
    Epoch 89/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6942 - acc: 0.5915 - val_loss: 1.7047 - val_acc: 0.5899
    Epoch 90/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6908 - acc: 0.5909 - val_loss: 1.6569 - val_acc: 0.5792
    Epoch 91/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6987 - acc: 0.5936 - val_loss: 1.6810 - val_acc: 0.5857
    Epoch 92/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6976 - acc: 0.5903 - val_loss: 1.6755 - val_acc: 0.5859
    Epoch 93/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6924 - acc: 0.5928 - val_loss: 1.6425 - val_acc: 0.5938
    Epoch 94/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6905 - acc: 0.5991 - val_loss: 1.6878 - val_acc: 0.5848
    Epoch 95/100
    14169/14169 [==============================] - 39s 3ms/step - loss: 1.6966 - acc: 0.5911 - val_loss: 1.6959 - val_acc: 0.5769
    Epoch 96/100
    14169/14169 [==============================] - 41s 3ms/step - loss: 1.6853 - acc: 0.5943 - val_loss: 1.6480 - val_acc: 0.5927
    Epoch 97/100
    14169/14169 [==============================] - 40s 3ms/step - loss: 1.6791 - acc: 0.5964 - val_loss: 1.6827 - val_acc: 0.5947
    Epoch 98/100
    14169/14169 [==============================] - 40s 3ms/step - loss: 1.6903 - acc: 0.5854 - val_loss: 1.6791 - val_acc: 0.5916
    Epoch 99/100
    14169/14169 [==============================] - 40s 3ms/step - loss: 1.7010 - acc: 0.5883 - val_loss: 1.6691 - val_acc: 0.6023
    Epoch 100/100
    14169/14169 [==============================] - 40s 3ms/step - loss: 1.7092 - acc: 0.5842 - val_loss: 1.6557 - val_acc: 0.5868
    dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])



![png]({filename}/images/4C_output_20_1.png)



![png]({filename}/images/4C_output_20_2.png)

