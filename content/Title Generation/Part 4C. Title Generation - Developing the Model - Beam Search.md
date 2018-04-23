Title: 4C. Title Generation - Developing the Model - Beam Search
Tags: improving_results
Authors: Ben Hoyle
Summary: This post looks at developing our initial models to include state of the art features to improve results.

# 4C. Title Generation - Developing the Model - Beam Search

This post looks at developing our initial models to include state of the art features to improve results.

To recap:

* We have two models: the Ludwig model and the Chollet/Brownlee model. 
* Performance so far has been fairly poor but the Ludwig model appears to give better results out of the box.
* Each model had slightly different characteristics - the Ludwig model produced better formed output but seemed to simply memorise and repeat titles, the Chollet/Brownlee model had a lower loss and appeared to memorise less but produced more nonsensical outputs.

In this post we will look at implementing a beam search decoder as an improvement to our previous greedy decoder.

### Beam Search Decoding

This [video from the Udacity Deep Learning Course](https://www.youtube.com/watch?v=UXW6Cs82UKo) provides a good explanation of beam search.

Basically, when you generate a sequence on a token-by-token basis, each token has an associated probability, where the decoder outputs a vector indicative of probabilities for all possible tokens (in our model at least). 

A naive way to select the tokens for the sequence is just to select the largest probability at each step in the sequence. Many decoder structures that are described on the Internet do this. However, this can lead to problems: if you have several tokens with similar probabilities, you might select one token that actually leads to a less likely sequence overall.

To remedy this you might want to keep track of all possible output sequences. However, the number of possible sequences grows exponentially. (Even with a small vocabulary of 2500 tokens you can end up the same number of sequences as there are atoms in the universe!) Beam search makes this manageable by keeping track of only k top sequences. 

Machine Learning Mastery has an article [here](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/) that describes how to implement a beam search in Python. However, in the comments this is criticised for being too simplistic. On review, I think this criticism is warranted: the function does not look at different, branching trees of tokens; instead, the probabilities are "given" in a pre-generated array.

Here is [another example](https://gist.github.com/udibr/67be473cf053d8c38730) of a beam search for keras and here is an [example slide](http://www.cs.cmu.edu/afs/cs/academic/class/46927-f97/slides/Lec3/sld023.htm) showing the branching trees.

To implement, the beam search algorithm, we need a function that performs the following steps:

* Initialise probabilities to 1 for start token;
* Predict the list of probabilities for the start token;
* Update scores as score * -log(list of probs);
* Select the k top scores;
* Build k sequences for each of the k top probabilities  = start token + prediction for 0 to k-1; and
* Iterate by predicting the next token for each sequence and updating the scores and sequences.

We can also amend our methods for printing examples to print the top k sequences and their probabilities.

With reference to our models, we can implement beam search as a custom `_predict_from_seq()` method.

We need a data structure to store, for our k top sequences:

* scores;
* token sequences.

Our input will be:

* claim text (as for our previous decoder); and
* a value for k, the number of sequences to track.

We want to output:

* the k sequences with top scores, where sequences stop when the stop token is predicted or when the maximum number of output tokens is reached.

## Ludwig Model - Custom Decoding Function

Let's start by working on our custom decoding function. This will have the form of `_predict_from_seq()` as previously developed.

We can first try this out and debug as a separate method that takes a pre-existing model object as self. Once we get something working we can then easily create a custom class that inherits from a previous object definition that has the custom beam search decoding method.


```python
from ludwig_model import LudwigModel
```

    /home/ben/anaconda3/envs/tf_gpu_source/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.



```python
# Load our data as before
import pickle
import os

PIK = "claim_and_title.data"

if not os.path.isfile(PIK):
    # Download file
    !wget https://benhoyle.github.io/notebooks/title_generation/claim_and_title.data

with open(PIK, "rb") as f:
    print("Loading data")
    data = pickle.load(f)
    print("{0} samples loaded".format(len(data)))
    
print("\n\nAdding start and stop tokens to output")
data = [(c, "startseq {0} stopseq".format(t)) for c, t in data]
                                      
print("\n\nAn example title:", data[0][1])
print("----")
print("An example claim:", data[0][0])
```

    Loading data
    30000 samples loaded
    
    
    Adding start and stop tokens to output
    
    
    An example title: startseq System and method for session restoration at geo-redundant gateways stopseq
    ----
    An example claim: 
    1. A method for managing a backup service gateway (SGW) associated with a primary SGW, the method comprising:
    periodically receiving from the primary SGW at least a portion of corresponding UE session state information, the received portion of session state information being sufficient to enable the backup SGW to indicate to an inquiring management entity that UEs having an active session supported by the primary SGW are in a live state; and
    in response to a failure of the primary SGW, the backup SGW assuming management of IP addresses and paths associated with said primary SGW and transmitting a Downlink Data Notification (DDN) toward a Mobility Management Entity (MME) for each of said UEs having an active session supported by the failed primary SGW to detach from the network and reattach to the network, wherein each DDN causes the MME to send a detach request with a reattach request code to the respective UE.
    
    



```python
machine = LudwigModel(
    encoder_texts=[d[0] for d in data],
    decoder_texts=[d[1] for d in data],
    encoder_seq_length=300,
    decoder_seq_length=22,
    num_encoder_tokens=2500,
    num_decoder_tokens=2500,
    latent_dim=128,
    weights_file="LW_model.hdf5",
    training_set_size=250
)
```

    Fitting tokenizers
    Our input data has shape (30000, 300) and our output data has shape (30000, 22)
    Generating training and test data
    Building model
    Loading GloVe 100d embeddings from file
    Found 400000 word vectors.
    Building embedding matrix
    Compiling model
    Loaded weights


```python
# Imports 
import math
import random
import numpy as np


def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array. """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return probas

def _predict_from_seq(self, input_data, k, temp=1.0):
    """ Predict output text from input text. """
    input_seq = input_data.reshape(1, self.encoder_seq_length)
    ndt = self.num_decoder_tokens  # Just shorten the variable name 
    output_sequences = []
    
    ans_partial = np.zeros((1, self.decoder_seq_length), dtype=np.int16)
    ans_partial[0, -1] = self.output_dictionary["startseq"]
    
    # Initialise data storage - char sequence, state sequence, scores
    sequences = [[ans_partial, 0.0]]
    
    for _ in range(self.decoder_seq_length):
        # Create an empty array to hold the scores in each pass
        new_scores = np.zeros(shape=(len(sequences)*ndt))
        temp_states = []
        #print([s[0] for s in sequences])
        for i, seq in enumerate(sequences):
            #print(i, seq[0])
            # Get most recent score
            prev_score = seq[1]
            # Get current token sequence
            ans_partial = seq[0]
            # Predict token probabilities using previous state and token for sequence
            yhat = self.infdec.predict([input_seq, ans_partial])
            # Unpack yhat array of probabilities
            #yhat = sample(yhat[0, 0, :])
            yhat = yhat[0,:]
            # print(yhat)
            # As we are taking the log do we not sum the scores?
            new_scores[i*ndt:(i+1)*ndt] = prev_score+-np.log(yhat) 
            # print(new_scores)
            # Actually I only need to look at the top k for each sequence, then further distill these down to the top k
            # across all sequences - maybe one for optimising later
            
        # Outside of loop we want to pick the k highest scores
        # Each sequence has a set of scores equal in size to num_decoder_tokens- i.e. k*n_d_t scores in total
        # We want to select the k highest scores across all scores in total - but then we need to know
        # We just modulo k on the index to find the indice and int(index/num_decoder_tokens) to find k
        
        # Then update the sequences to reflect these k highest scores
            
        # select top k scores -as we are minimising these are at bottom of list
        top_k_indices = np.argsort(new_scores)[:k].tolist()
        #print(new_scores[top_k_indices])
        #print(top_k_indices)
        new_sequences = []
        seqs_to_delete = []
        for index in top_k_indices:
            seq_select = int(index/ndt)
            #print("Selected seq: ", seq_select)
            #print("Length of sequences: ", len(sequences))
            new_token = index % ndt # this is the token index
            # This is equivalent to argmax - but should we actually be sampling?
            # Update the partial answer
            new_ans_partial = np.zeros((1, self.decoder_seq_length), dtype=np.int16)
            new_ans_partial[0, 0:-1] = sequences[seq_select][0][0, 1:]
            # This then adds the newly decoded word onto the end of ans_partial
            new_ans_partial[0, -1] = new_token
            #print(new_scores[index])
            entry = (new_ans_partial, new_scores[index])
            # If predicted token is end token
            if new_token == self.output_dictionary["stopseq"]:
                # Add data for output
                output_sequences.append(entry)
                # Reduce k by 1
                k -= 1
            else:
                # Add to list of new sequences to use
                new_sequences.append(entry)
        sequences = new_sequences
        if k == 0:
            break
    
    # Sort list in reverse "score" order
    output_sequences.sort(key=lambda x: x[1])
    return output_sequences

def print_examples(self, number, k=5):
    num_test_titles = len(self.input_test_data)
    indices = random.sample(range(0, num_test_titles), number)
    for i in indices:
        input_sample = self.input_test_data[i]
        output_sample = self.output_test_data[i]
        seqs = _predict_from_seq(self, input_sample, k)
        output_sample_text = self._seq2text(output_sample)
        claim_text = self._seq2text(input_sample, output=False)
        print("-----------------")
        print("Sample of claim text: {}".format(claim_text[0:200]))
        print("-----------------")
        print("\nActual title is: {} \n---".format(output_sample_text))
        for seq, score in seqs:
            o_string = (
                "Predicted title is: {0} with score {1}"
            )
            
            print(o_string.format(self._seq2text(seq[0].tolist()), score))
        
```


```python
# Let's test without Beam Search decoding
machine.example_output(5)
```

    ------------------------------------------
    Sample of claim text: 1 a computer implemented system for providing users with an interface comprising temporal information the system comprising a computer system the computer system comprising at least one processor and 
    
    Predicted title is: method and system for providing a user interface  
    Actual title is: providing temporal information to users  
    ---
    Sample of claim text: blocks being disposed against the hot key patterns the first electrode blocks being disposed against the key group pattern the second sensing block being disposed against the cursor control pattern th
    
    Predicted title is: integrated circuit with a power supply  
    Actual title is: integrated input apparatus  
    ---
    Sample of claim text: 1 a mobile product inventory management device comprising a display a camera a processor and a recognition module executable on the processor and that the processor to capture image data via the camer
    
    Predicted title is: method and system for providing a product  
    Actual title is: virtual management systems and methods  
    ---
    Sample of claim text: 1 a method comprising receiving a two dimensional 2d image frame receiving three dimensional 3d image data corresponding to the 2d image frame using the 3d image data corresponding to the 2d image fra
    
    Predicted title is: method and apparatus for three dimensional image  
    Actual title is: using a combination of 2d and 3d image data to determine hand features information  
    ---
    Sample of claim text: 1 a non transitory computer readable medium having computer executable instructions for performing a method comprising collecting performance data for different domains in a system by activating a plu
    
    Predicted title is: method and system for managing data  
    Actual title is: system and method to monitor performance of different domains associated with a computer or network  
    ---



```python
self = machine
print_examples(self, 5)
```

    -----------------
    Sample of claim text: 1 a display control apparatus for controlling a display content of a display section of a peripheral that is connected to the display control apparatus the display section being included by the periph
    -----------------
    
    Actual title is: display control apparatus communicating with a peripheral to present operational information to users  
    ---
    Predicted title is: display apparatus  with score 4.325193405151367
    Predicted title is: information processing apparatus information processing method and recording medium  with score 8.561894416809082
    Predicted title is: method and apparatus for portable electronic device  with score 8.68794059753418
    Predicted title is: information processing apparatus information processing method and electronic device  with score 10.179014205932617
    Predicted title is: information processing apparatus information processing method and computer readable medium  with score 10.28657341003418
    -----------------
    Sample of claim text: and memory configured for connection with a plurality of components of the aircraft to retrieve data from one or more modules of the aircraft components a configuration file in said portable computer 
    -----------------
    
    Actual title is: configuration management apparatus and related methods  
    ---
    Predicted title is: system and method for data processing  with score 8.64958381652832
    Predicted title is: system and method for managing data  with score 9.201067924499512
    Predicted title is: system and method for data processing in a virtual environment  with score 12.002704620361328
    Predicted title is: system and method for data processing in a virtual machine  with score 13.135804176330566
    Predicted title is: system and method for data processing in a web page  with score 13.158220291137695
    -----------------
    Sample of claim text: 1 a computer implemented method for processing a selection associated with a media content received in an electronic broadcast transmission the media content presented by a receiving device and the me
    -----------------
    
    Actual title is: broadcast response method and system  
    ---
    Predicted title is: system and method for data  with score 6.181354522705078
    Predicted title is: system and method for data processing  with score 7.070134162902832
    Predicted title is: system and method for data processing in an electronic device  with score 11.689477920532227
    Predicted title is: system and method for data processing in an electronic system  with score 11.725565910339355
    Predicted title is: system and method for data processing in a digital media system  with score 12.572397232055664
    -----------------
    Sample of claim text: 1 a method of detecting human objects in a video comprising determining pixels of a video image are foreground pixels the group of foreground pixels a foreground set of one or more foreground for each
    -----------------
    
    Actual title is: 3d human pose and shape modeling  
    ---
    Predicted title is: method and apparatus for creating images  with score 7.900638103485107
    Predicted title is: method and apparatus for creating and displaying images  with score 11.06941032409668
    Predicted title is: method and apparatus for creating images of images  with score 11.170570373535156
    Predicted title is: method and apparatus for creating and displaying video  with score 12.05427360534668
    Predicted title is: method and apparatus for creating and displaying video images  with score 12.832809448242188
    -----------------
    Sample of claim text: 1 in a computing system environment a method of of files stored on one or more computing devices each file having a plurality of symbols representing an underlying data stream of original bits of data
    -----------------
    
    Actual title is: grouping and volumes of files  
    ---
    Predicted title is: system and method for creating data  with score 9.60139274597168
    Predicted title is: system and method for creating a data stream  with score 11.702018737792969
    Predicted title is: systems and methods for creating a data stream  with score 12.231966018676758
    Predicted title is: system and method for creating data in a data stream  with score 14.472331047058105
    Predicted title is: system and method for creating data from a data stream  with score 14.49394416809082



```python
print_examples(self, 5, k=10)
```

    -----------------
    Sample of claim text: 1 a computer implemented method of reducing risk in a payment based transaction wherein payment is made from an account holder to a using a payment bank system operated by a payment bank the method co
    -----------------
    
    Actual title is: reducing risk in a payment based transaction based upon at least one user supplied risk parameter including a payment limit  
    ---
    Predicted title is: payment system  with score 5.297518253326416
    Predicted title is: payment payment system  with score 5.336144924163818
    Predicted title is: system and method for payment transactions  with score 5.893258571624756
    Predicted title is: systems and methods for payment transactions  with score 6.355759620666504
    Predicted title is: method and system for payment transactions  with score 6.844301223754883
    Predicted title is: systems and methods for payment payment transactions  with score 7.168615818023682
    Predicted title is: system and method for payment payment transactions  with score 7.247190952301025
    Predicted title is: system and method for payment processing  with score 7.810699939727783
    Predicted title is: system and method for payment payment  with score 7.991192817687988
    Predicted title is: methods and systems for payment payment transactions  with score 8.089160919189453
    -----------------
    Sample of claim text: 1 a method performed by a system that supports services between a plurality of and a plurality of in a communication network the method comprising delivering a seller interface via the communication n
    -----------------
    
    Actual title is: system supporting creation  
    ---
    Predicted title is: method and system for providing a product  with score 13.070158004760742
    Predicted title is: methods and systems for providing a product  with score 13.112994194030762
    Predicted title is: system and method for providing a product  with score 13.368507385253906
    Predicted title is: method and apparatus for providing a product  with score 13.528003692626953
    Predicted title is: methods and systems for providing a product item  with score 15.519207954406738
    Predicted title is: method and system for providing a product item  with score 15.596986770629883
    Predicted title is: method and apparatus for providing real time messaging  with score 15.767982482910156
    Predicted title is: method and system for providing real time messaging  with score 15.802291870117188
    Predicted title is: method and system for providing real time party based on the internet  with score 24.388505935668945
    Predicted title is: method and system for providing real time party based on a plurality of resources  with score 27.679676055908203
    -----------------
    Sample of claim text: 1 a method of providing consistent user experience in the method comprising receiving an indication of a target level of user experience for an application computing a target application performance c
    -----------------
    
    Actual title is: cache management in application  
    ---
    Predicted title is: system and method for managing application  with score 11.204858779907227
    Predicted title is: system and method for managing application development  with score 12.491059303283691
    Predicted title is: system and method for providing a document  with score 13.075180053710938
    Predicted title is: system and method for providing a secure document  with score 13.515348434448242
    Predicted title is: system and method for generating a secure document  with score 14.135204315185547
    Predicted title is: system and method for providing a user interface  with score 14.223642349243164
    Predicted title is: system and method for providing a user space  with score 14.36276912689209
    Predicted title is: system and method for providing a cache based on user preferences  with score 19.699052810668945
    Predicted title is: system and method for providing a cache based on user space  with score 20.409120559692383
    Predicted title is: system and method for providing a cache based on user preferences of a user  with score 28.28386688232422
    -----------------
    Sample of claim text: or more programmable processors a plurality of data objects associated with data stored in a database at a persistent storage the providing for execution of a service called by an application storing 
    -----------------
    
    Actual title is: for objects  
    ---
    Predicted title is: system and method for data collection  with score 9.759042739868164
    Predicted title is: system and method for data processing  with score 9.763725280761719
    Predicted title is: system and method for managing data  with score 10.533380508422852
    Predicted title is: system and method for creating a database  with score 11.950915336608887
    Predicted title is: system and method for data processing in a database  with score 13.33000373840332
    Predicted title is: system and method for data processing in a cluster  with score 13.828896522521973
    Predicted title is: system and method for data collection in a database  with score 13.97393798828125
    Predicted title is: system and method for data processing in a database system  with score 14.517136573791504
    Predicted title is: system and method for data processing in a cloud  with score 14.586236000061035
    Predicted title is: system and method for data processing in a cloud computing environment  with score 14.967475891113281
    -----------------
    Sample of claim text: 1 a method for enabling a product offering received by a client from a vendor the method comprising in a computer system at the client providing to the vendor a plurality of strings computing an index
    -----------------
    
    Actual title is: secure product over channels with bandwidth  
    ---
    Predicted title is: method and system for providing financial transactions  with score 10.173721313476562
    Predicted title is: method and system for providing a product  with score 10.914628982543945
    Predicted title is: system and method for providing a product  with score 10.918341636657715
    Predicted title is: methods and systems for providing a product  with score 11.22200870513916
    Predicted title is: system and method for providing a secure product  with score 12.802101135253906
    Predicted title is: method and system for providing a secure product  with score 12.827587127685547
    Predicted title is: system and method for providing a product item  with score 13.25611686706543
    Predicted title is: method and system for providing a product item  with score 13.26268196105957
    Predicted title is: method and system for providing a product development  with score 13.407301902770996
    Predicted title is: system and method for providing a product development  with score 13.513633728027344


---
## Comments

Using the beam search decoder has two advantages:
- it improves the quality of our predicted titles: the highest scoring titles in these examples appear to reflect fairly accurately the most relevant titles; and
- it allows us to visualise how our models are working by outputing the k highest scoring entries.

The downside is that decoding time is increased. But this doesn't matter that much for our toy project.

It seems worthwhile folding beam search into our models. One option is to maybe create a mix-in class to add the beam search functionality.
