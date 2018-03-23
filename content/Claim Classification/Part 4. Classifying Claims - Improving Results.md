Title: 4A. Classifying Claims - Improving Results
Tags: improving_results
Authors: Ben Hoyle
Summary: This post looks at how we can improve our initial results.

# 4A. Classifying Claims - Improving Results

At this stage, we will look at taking a few of our best-performing algorithms from our previous post and attempting to increase performance. 

---

As you may remember, the best-performing algorithms were: 

* Naive Bayes - an accuracy of 58.90% (1.12% variance);
* SVM with non-linear kernel - an accuracy of 62.52% (0.80% variance); and
* Multilayer neural networks - an accuracy of 61.20% (0.54% variance).

Ths posts [here](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/) and [here](https://machinelearningmastery.com/improve-deep-learning-performance/) offer some suggestions for how to increase performance: 

1. Improve Performance With Data.
2. Improve Performance With Algorithms.
3. Improve Performance With Algorithm Tuning.
4. Improve Performance With Ensembles.

We should also look at the time required to train our models. If a model is 10x faster to train, it may be preferable to another model with greater accuracy. 

---

## Load the Data


```python
import pickle
with open("encoded_data.pkl", "rb") as f:
    print("Loading data")
    X, Y = pickle.load(f)
    print("{0} claims and {1} classifications loaded".format(len(X), len(Y)))
```

    Loading data
    11238 claims and 11238 classifications loaded



```python
import numpy as np
# Convert one hot to target integer
Y_integers = np.argmax(Y, axis=1)
```

## Naive Bayes to Start

It turns out that there are not too many parameters we can vary for a Naive Bayes classifier. In fact, in scikit-learn there is only one tuneable parameter - alpha - that sets an amount of additive smoothing (see the documentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

The smoothing is set to 1.0 as a default. Let's have a look at performance if we turn this off by setting the alpha parameter to 0.


```python
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

scores = cross_val_score(MultinomialNB(alpha=1.0e-3), X, Y_integers, cv=5)
      
print(
        "NB with alpha = 0 has an average classification accuracy of {0:.2f} ({1:.2f})".format(
            scores.mean()*100, 
            scores.std()*100
        )    
)
```

    NB with alpha = 0 has an average classification accuracy of 58.94 (0.43)


As we can see changing this parameter doesn't seem to have that much effect.

## Improving Performance with Algorithms

Now, scikit-learn provides a rather simple multi-layer neural network implementation. To improve performance we can experiment with more advanced neural network architectures.  

Questions we can ask include:
- Does changing the number of hidden layers increase performance?
- Does using Dropout increase performance?
- Does changing the dimensionality of our hidden layers increase performance?

Keras is an excellent library for experimenting with deep learning models. We will play around using that. This post [here](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/) explains how we can perform k-fold validation with keras models.


```python
input_dim = X.shape[1]
print("Our input dimension for our claim representation is {0}".format(input_dim))

no_classes = Y.shape[1]
print("Our output dimension is {0}".format(no_classes))
```

    Our input dimension for our claim representation is 5000
    Our output dimension is 8


### Keras Model

Before attempting cross-validation, let's build and explore a Keras model.  

This post [here](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/) provides help for obtaining training metrics as Keras callback data and plotting those metrics.


```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt

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

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_9 (Dense)              (None, 500)               2500500   
    _________________________________________________________________
    dense_10 (Dense)             (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/10
    8990/8990 [==============================] - 298s - loss: 1.1831 - acc: 0.5908 - val_loss: 1.1081 - val_acc: 0.6161
    Epoch 2/10
    8990/8990 [==============================] - 297s - loss: 0.3009 - acc: 0.8979 - val_loss: 1.5360 - val_acc: 0.6090
    Epoch 3/10
    8990/8990 [==============================] - 294s - loss: 0.0672 - acc: 0.9850 - val_loss: 1.8990 - val_acc: 0.6174
    Epoch 4/10
    8990/8990 [==============================] - 298s - loss: 0.0364 - acc: 0.9932 - val_loss: 2.1578 - val_acc: 0.6290
    Epoch 5/10
    8990/8990 [==============================] - 307s - loss: 0.0150 - acc: 0.9977 - val_loss: 2.3248 - val_acc: 0.6157
    Epoch 6/10
    8990/8990 [==============================] - 311s - loss: 0.0105 - acc: 0.9986 - val_loss: 2.4560 - val_acc: 0.6277
    Epoch 7/10
    8990/8990 [==============================] - 307s - loss: 0.0106 - acc: 0.9981 - val_loss: 2.6464 - val_acc: 0.6192
    Epoch 8/10
    8990/8990 [==============================] - 308s - loss: 0.0095 - acc: 0.9986 - val_loss: 2.8779 - val_acc: 0.6050
    Epoch 9/10
    8990/8990 [==============================] - 300s - loss: 0.0117 - acc: 0.9976 - val_loss: 3.0982 - val_acc: 0.6085
    Epoch 10/10
    8990/8990 [==============================] - 286s - loss: 0.0386 - acc: 0.9909 - val_loss: 3.3565 - val_acc: 0.5925
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_11_1.png)



![png]({filename}/images/output_11_2.png)


### Overfitting

Looking at this first set of results we can see that our neural network quickly overfits the data, while the test performance stays reasonably constant.  

One way to reduce overfitting is to [apply Dropout](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/). Let's try that now.


```python
from keras.layers import Dropout

def keras_dropout_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(input_dim,)))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_dropout_model()
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
    dropout_1 (Dropout)          (None, 5000)              0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 500)               2500500   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/10
    8990/8990 [==============================] - 297s - loss: 1.2949 - acc: 0.5539 - val_loss: 1.0363 - val_acc: 0.6326
    Epoch 2/10
    8990/8990 [==============================] - 277s - loss: 0.6315 - acc: 0.7833 - val_loss: 1.2275 - val_acc: 0.6174
    Epoch 3/10
    8990/8990 [==============================] - 268s - loss: 0.3351 - acc: 0.8899 - val_loss: 1.5079 - val_acc: 0.6125
    Epoch 4/10
    8990/8990 [==============================] - 285s - loss: 0.1916 - acc: 0.9408 - val_loss: 1.6871 - val_acc: 0.6005
    Epoch 5/10
    8990/8990 [==============================] - 288s - loss: 0.1392 - acc: 0.9590 - val_loss: 1.9611 - val_acc: 0.6143
    Epoch 6/10
    8990/8990 [==============================] - 272s - loss: 0.1226 - acc: 0.9648 - val_loss: 2.1324 - val_acc: 0.6036
    Epoch 7/10
    8990/8990 [==============================] - 272s - loss: 0.1233 - acc: 0.9675 - val_loss: 2.2092 - val_acc: 0.6201
    Epoch 8/10
    8990/8990 [==============================] - 271s - loss: 0.1033 - acc: 0.9730 - val_loss: 2.3560 - val_acc: 0.6170
    Epoch 9/10
    8990/8990 [==============================] - 273s - loss: 0.0860 - acc: 0.9770 - val_loss: 2.5313 - val_acc: 0.6072
    Epoch 10/10
    8990/8990 [==============================] - 273s - loss: 0.0886 - acc: 0.9772 - val_loss: 2.4294 - val_acc: 0.6281
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_13_1.png)



![png]({filename}/images/output_13_2.png)


Now let's try with a more aggressive level of Dropout.


```python
from keras.layers import Dropout

def keras_dropout_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(input_dim,)))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_dropout_model()
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
    dense_13 (Dense)             (None, 500)               2500500   
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_14 (Dense)             (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/10
    8990/8990 [==============================] - 289s - loss: 1.5896 - acc: 0.4839 - val_loss: 1.1905 - val_acc: 0.5778
    Epoch 2/10
    8990/8990 [==============================] - 288s - loss: 1.1707 - acc: 0.5999 - val_loss: 1.1088 - val_acc: 0.6121
    Epoch 3/10
    8990/8990 [==============================] - 290s - loss: 1.0582 - acc: 0.6451 - val_loss: 1.0960 - val_acc: 0.6317
    Epoch 4/10
    8990/8990 [==============================] - 293s - loss: 0.9756 - acc: 0.6762 - val_loss: 1.1284 - val_acc: 0.6094
    Epoch 5/10
    8990/8990 [==============================] - 279s - loss: 0.9106 - acc: 0.6919 - val_loss: 1.1016 - val_acc: 0.6352
    Epoch 6/10
    8990/8990 [==============================] - 288s - loss: 0.8499 - acc: 0.7195 - val_loss: 1.1338 - val_acc: 0.6383
    Epoch 7/10
    8990/8990 [==============================] - 297s - loss: 0.8072 - acc: 0.7384 - val_loss: 1.1584 - val_acc: 0.6299
    Epoch 8/10
    8990/8990 [==============================] - 324s - loss: 0.7602 - acc: 0.7536 - val_loss: 1.1878 - val_acc: 0.6263
    Epoch 9/10
    8990/8990 [==============================] - 289s - loss: 0.7387 - acc: 0.7682 - val_loss: 1.2066 - val_acc: 0.6152
    Epoch 10/10
    8990/8990 [==============================] - 267s - loss: 0.7270 - acc: 0.7704 - val_loss: 1.2316 - val_acc: 0.6317
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_15_1.png)



![png]({filename}/images/output_15_2.png)


This seems to work a little bit better - but there is about 2-3% variance so it is difficult to tell. Maybe we can try to train for a number of additional epochs.


```python
from keras.layers import Dropout

def keras_dropout_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(input_dim,)))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_dropout_model()
model.summary()
history = model.fit(X, Y, validation_split=0.2, epochs=20, batch_size=5, verbose=1)

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
    dropout_5 (Dropout)          (None, 5000)              0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 500)               2500500   
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/20
    8990/8990 [==============================] - 278s - loss: 1.6090 - acc: 0.4763 - val_loss: 1.1685 - val_acc: 0.6005
    Epoch 2/20
    8990/8990 [==============================] - 283s - loss: 1.1541 - acc: 0.6034 - val_loss: 1.1139 - val_acc: 0.6241
    Epoch 3/20
    8990/8990 [==============================] - 285s - loss: 1.0295 - acc: 0.6519 - val_loss: 1.1211 - val_acc: 0.6192
    Epoch 4/20
    8990/8990 [==============================] - 260s - loss: 0.9636 - acc: 0.6799 - val_loss: 1.1416 - val_acc: 0.6219
    Epoch 5/20
    8990/8990 [==============================] - 259s - loss: 0.9235 - acc: 0.6949 - val_loss: 1.1070 - val_acc: 0.6157
    Epoch 6/20
    8990/8990 [==============================] - 259s - loss: 0.8431 - acc: 0.7226 - val_loss: 1.1461 - val_acc: 0.6125
    Epoch 7/20
    8990/8990 [==============================] - 259s - loss: 0.7866 - acc: 0.7433 - val_loss: 1.1609 - val_acc: 0.6237
    Epoch 8/20
    8990/8990 [==============================] - 258s - loss: 0.7834 - acc: 0.7493 - val_loss: 1.1745 - val_acc: 0.6272
    Epoch 9/20
    8990/8990 [==============================] - 258s - loss: 0.7383 - acc: 0.7671 - val_loss: 1.1679 - val_acc: 0.6232
    Epoch 10/20
    8990/8990 [==============================] - 324s - loss: 0.7204 - acc: 0.7705 - val_loss: 1.2209 - val_acc: 0.6192
    Epoch 11/20
    8990/8990 [==============================] - 321s - loss: 0.7308 - acc: 0.7707 - val_loss: 1.2136 - val_acc: 0.6188
    Epoch 12/20
    8990/8990 [==============================] - 312s - loss: 0.6907 - acc: 0.7908 - val_loss: 1.2437 - val_acc: 0.6241
    Epoch 13/20
    8990/8990 [==============================] - 325s - loss: 0.6700 - acc: 0.7973 - val_loss: 1.2450 - val_acc: 0.6272
    Epoch 14/20
    8990/8990 [==============================] - 298s - loss: 0.6431 - acc: 0.8097 - val_loss: 1.2810 - val_acc: 0.6197
    Epoch 15/20
    8990/8990 [==============================] - 324s - loss: 0.6202 - acc: 0.8110 - val_loss: 1.3603 - val_acc: 0.6036
    Epoch 16/20
    8990/8990 [==============================] - 296s - loss: 0.6130 - acc: 0.8127 - val_loss: 1.3270 - val_acc: 0.6174
    Epoch 17/20
    8990/8990 [==============================] - 326s - loss: 0.6338 - acc: 0.8135 - val_loss: 1.3408 - val_acc: 0.6121
    Epoch 18/20
    8990/8990 [==============================] - 314s - loss: 0.6336 - acc: 0.8219 - val_loss: 1.3723 - val_acc: 0.6192
    Epoch 19/20
    8990/8990 [==============================] - 314s - loss: 0.5931 - acc: 0.8304 - val_loss: 1.3888 - val_acc: 0.6094
    Epoch 20/20
    8990/8990 [==============================] - 343s - loss: 0.6129 - acc: 0.8259 - val_loss: 1.3998 - val_acc: 0.6148
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_17_1.png)



![png]({filename}/images/output_17_2.png)


Okay so Dropout doesn't appear to improve our performance. Overfitting still occurs over time and the test accuracy appears stubbornly fixed at around 60-62%.


---

## Improving Performance with Algorithm Tuning

Let's explore a few architecture choices with our keras model to see if we can see any improvement. At this stage we are just looking for low-hanging fruit. If a particular direction looks promising, there is the option to use grid search routines to find optimal parameters. 


```python
def keras_multi_layer_model():
    # create model
    model = Sequential()
    model.add(Dense(1000, input_dim=input_dim, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = keras_multi_layer_model()
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
    dense_17 (Dense)             (None, 1000)              5001000   
    _________________________________________________________________
    dense_18 (Dense)             (None, 500)               500500    
    _________________________________________________________________
    dense_19 (Dense)             (None, 250)               125250    
    _________________________________________________________________
    dense_20 (Dense)             (None, 8)                 2008      
    =================================================================
    Total params: 5,628,758
    Trainable params: 5,628,758
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/10
    8990/8990 [==============================] - 647s - loss: 1.2164 - acc: 0.5645 - val_loss: 1.0894 - val_acc: 0.6277
    Epoch 2/10
    8990/8990 [==============================] - 632s - loss: 0.5716 - acc: 0.7991 - val_loss: 1.2262 - val_acc: 0.5952
    Epoch 3/10
    8990/8990 [==============================] - 631s - loss: 0.2023 - acc: 0.9385 - val_loss: 1.9201 - val_acc: 0.6085
    Epoch 4/10
    8990/8990 [==============================] - 664s - loss: 0.0728 - acc: 0.9806 - val_loss: 2.3951 - val_acc: 0.6214
    Epoch 5/10
    8990/8990 [==============================] - 634s - loss: 0.0720 - acc: 0.9858 - val_loss: 2.7070 - val_acc: 0.6183
    Epoch 6/10
    8990/8990 [==============================] - 636s - loss: 0.0561 - acc: 0.9892 - val_loss: 3.1284 - val_acc: 0.6059
    Epoch 7/10
    8990/8990 [==============================] - 635s - loss: 0.0662 - acc: 0.9899 - val_loss: 3.1941 - val_acc: 0.5988
    Epoch 8/10
    8990/8990 [==============================] - 643s - loss: 0.0495 - acc: 0.9918 - val_loss: 3.4484 - val_acc: 0.6130
    Epoch 9/10
    8990/8990 [==============================] - 635s - loss: 0.0384 - acc: 0.9938 - val_loss: 3.5622 - val_acc: 0.6134
    Epoch 10/10
    8990/8990 [==============================] - 635s - loss: 0.0292 - acc: 0.9961 - val_loss: 3.8888 - val_acc: 0.6117
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_20_1.png)



![png]({filename}/images/output_20_2.png)


## Regularisation

One method of attempting to avoid overfitting is to limit the magnitude of our weights using regularisation. This is explained in this [post](https://cambridgespark.com/content/tutorials/neural-networks-tuning-techniques/index.html).  

Here we will try L2 regularisation, which penalises large weight values.


```python
from keras.regularizers import l2 # L2-regularisation
l2_lambda = 0.01
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
    dense_23 (Dense)             (None, 500)               2500500   
    _________________________________________________________________
    dense_24 (Dense)             (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/10
    8990/8990 [==============================] - 578s - loss: 3.8866 - acc: 0.5388 - val_loss: 2.7980 - val_acc: 0.5867
    Epoch 2/10
    8990/8990 [==============================] - 723s - loss: 2.6364 - acc: 0.6141 - val_loss: 2.6501 - val_acc: 0.5876
    Epoch 3/10
    8990/8990 [==============================] - 667s - loss: 2.1624 - acc: 0.6660 - val_loss: 2.5055 - val_acc: 0.5819
    Epoch 4/10
    8990/8990 [==============================] - 594s - loss: 1.7579 - acc: 0.7053 - val_loss: 2.2376 - val_acc: 0.5792
    Epoch 5/10
    8990/8990 [==============================] - 587s - loss: 1.5254 - acc: 0.7428 - val_loss: 2.1618 - val_acc: 0.5770
    Epoch 6/10
    8990/8990 [==============================] - 598s - loss: 1.3854 - acc: 0.7645 - val_loss: 2.0072 - val_acc: 0.5979
    Epoch 7/10
    8990/8990 [==============================] - 559s - loss: 1.2818 - acc: 0.7796 - val_loss: 2.0247 - val_acc: 0.5890
    Epoch 8/10
    8990/8990 [==============================] - 530s - loss: 1.2330 - acc: 0.7917 - val_loss: 2.0380 - val_acc: 0.5863
    Epoch 9/10
    8990/8990 [==============================] - 530s - loss: 1.1755 - acc: 0.8007 - val_loss: 2.0731 - val_acc: 0.5609
    Epoch 10/10
    8990/8990 [==============================] - 571s - loss: 1.1802 - acc: 0.7996 - val_loss: 2.0386 - val_acc: 0.5796
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_23_1.png)



![png]({filename}/images/output_23_2.png)


Regularisation appears to work here. While it doesn't affect accuracy, we do now have the loss for both the train and test sets decreasing over time.


---

## Increasing Performance with Data

Here are some steps we can try to improve performance by modifying our initial data:
1. Normalise the data - it may be worth looking into whether we can normalise our input X matrix values data to scale between 0 and 1. Our Y vector is already in a one-hot format and so cannot be normalised.
2. Changing our X dimensionality - is there any benefit in increasing or decreasing the dimensionality of our input data. For example, we can change the vocubulary cap on our text tokeniser.
3. Get more data - we have more data available and so we can maybe up our dataset to 50,000 randomly selected claims, or use each claim in each claimset as a data sample. The reason why we have tried to limit our dataset is due to time and memory concerns - if we up our dataset size these may need to be looked at.
4. Changing the form of our input data - what happens if we use (normalised) term frequency instead of TD-IDF? Could we represent our text data as a continuous bag of words (e.g. a sum of word vectors)?

### Normalising the Data

We can first try to zero-center the data by substracting the mean and dividing by the standard deviation. 

Now, we should note that our data is already normalised to a certain extent by the inverse document frequency term. But it may be that zero-centered data provides improvement.



```python
X[0][0:10]
```




    array([ 0.        ,  0.        ,  2.43021996,  2.08331543,  1.71570602,
            2.52741068,  4.87087867,  2.99092954,  2.24937914,  0.        ])




```python
X_zero = X
X_zero -= np.mean(X_zero, axis = 0)
X_zero /= np.std(X, axis = 0)
X_zero[0][0:10]
```

    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:3: RuntimeWarning: invalid value encountered in true_divide
      This is separate from the ipykernel package so we can avoid doing imports until





    array([        nan, -2.22009656,  0.82212546,  0.62343302,  0.41434569,
            1.39141984,  2.02709972,  2.37849159,  1.34165113, -0.63808578])




```python
X_zero = np.nan_to_num(X_zero, copy=False)
X_zero[0][0:10]
```




    array([ 0.        , -2.22009656,  0.82212546,  0.62343302,  0.41434569,
            1.39141984,  2.02709972,  2.37849159,  1.34165113, -0.63808578])




```python
model2 = keras_model()
model2.summary()
history = model2.fit(X_zero, Y, validation_split=0.2, epochs=10, batch_size=5, verbose=1)

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
    dense_21 (Dense)             (None, 500)               2500500   
    _________________________________________________________________
    dense_22 (Dense)             (None, 8)                 4008      
    =================================================================
    Total params: 2,504,508
    Trainable params: 2,504,508
    Non-trainable params: 0
    _________________________________________________________________
    Train on 8990 samples, validate on 2248 samples
    Epoch 1/10
    8990/8990 [==============================] - 333s - loss: 2.4311 - acc: 0.5186 - val_loss: 2.4488 - val_acc: 0.5498
    Epoch 2/10
    8990/8990 [==============================] - 324s - loss: 1.1410 - acc: 0.7861 - val_loss: 2.5771 - val_acc: 0.5885
    Epoch 3/10
    8990/8990 [==============================] - 306s - loss: 0.7450 - acc: 0.8897 - val_loss: 3.1360 - val_acc: 0.5863
    Epoch 4/10
    8990/8990 [==============================] - 306s - loss: 0.5897 - acc: 0.9353 - val_loss: 3.6081 - val_acc: 0.5792
    Epoch 5/10
    8990/8990 [==============================] - 323s - loss: 0.5283 - acc: 0.9505 - val_loss: 3.8894 - val_acc: 0.5721
    Epoch 6/10
    8990/8990 [==============================] - 311s - loss: 0.7029 - acc: 0.9344 - val_loss: 4.2661 - val_acc: 0.5752
    Epoch 7/10
    8990/8990 [==============================] - 295s - loss: 0.5793 - acc: 0.9489 - val_loss: 4.4400 - val_acc: 0.5725
    Epoch 8/10
    8990/8990 [==============================] - 302s - loss: 0.6794 - acc: 0.9425 - val_loss: 4.9078 - val_acc: 0.5707
    Epoch 9/10
    8990/8990 [==============================] - 314s - loss: 0.7816 - acc: 0.9354 - val_loss: 4.6792 - val_acc: 0.5827
    Epoch 10/10
    8990/8990 [==============================] - 328s - loss: 0.7082 - acc: 0.9455 - val_loss: 5.2317 - val_acc: 0.5534
    dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])



![png]({filename}/images/output_32_1.png)



![png]({filename}/images/output_32_2.png)


Normalising our data does not seem to have any significant effect. This is to be expected - our data is already normalised to a certain extent, and we are dealing with sparse language data rather than continuous image data. 

This processing does appear to help slightly to reduce overfitting. This maybe suggests that regularisation may be useful.
