import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import math

from abstract_model_wrapper import AbstractModelWrapper

class BaseSeq2Seq(AbstractModelWrapper):
    """ Abstract class for sequence to sequence models for mixing in. """
    
    def __init__(
        self,
        encoder_texts=None,
        decoder_texts=None,
        encoder_seq_length=None,
        decoder_seq_length=None,
        num_encoder_tokens=None,
        num_decoder_tokens=None, 
        latent_dim=300,
        weights_file=None,
        training_set_size=250 # Due to memory issues we need to train in sets
    )
        # If passed with init process texts
        self.load_text_data(encoder_texts, decoder_texts)
        # If encoder / decoder seq_length = none we can set based on data
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # Likewise if the x_tokens are None we can set later based on the data
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.latent_dim = latent_dim
        self._start_checks()
        self._build_model()
        # Build model here?
        if weights_file and self.model:
            self.weights_file = weights_file
            
            self._load_weights(weights_file)
        else:
            self.weights_file = "weights.hdf5"
    
    # These are the exposed methods (to keep things simple)
    def load_text_data(self, encoder_texts, decoder_texts):
        """ Print a representation of the model. """
        self.encoder_texts = encoder_texts
        self.decoder_texts = decoder_texts
        
        # Convert to seqs - this will generate tokenizers
        self.input_data = self._text2seq(self, self.encoder_texts, encoder=True)
        self.output_data = self._text2seq(self, self.decoder_texts, encoder=False)
        
        # Set dictionaries - we need dictionaries in each direction
        # word_index = for word>int then generate as below for reverse
        self.input_dictionary = self.input_tokenizer.word_index
        self.output_dictionary = self.output_tokenizer.word_index
        self.input_dictionary_rev = dict(
            (i, char) for char, i in self.input_dictionary.items()
        )
        self.output_dictionary_rev = dict(
            (i, char) for char, i in self.output_dictionary.items()
        )
    
        self.input_data = _pad_seq(
            self.input_data, 
            self.encoder_seq_length, 'pre'
            )
        self.output_data = _pad_seq(
            self.output_data, 
            self.decoder_seq_length, 'post'
            )
        o_string = "Our input data has shape {0} and our output data has shape {1}"
        print(ostring.format(self.input_data.shape, self.output_data.shape))
        # Split into test and train
        self._split_data()
        
    def predict(self, input_text):
        """ Predict output text from input text. """
        input_seq = self._text2encoderseq(input_text)
        predicted_output_seq = self._predict_from_seq(input_seq)
        predicted_text = self._seq2text(predicted_output_seq)
        return predicted_text
        
    def train(self, epochs=1, reset_metrics=False):
        """ Train in batches on all data with validation. """
        if reset_metrics:
            self._reset_metrics()
        for e in epochs:
            print("Training for epoch {0}".format(e))
            self.example_output(5)
        self.plot_loss()
    
    def print(self):
        """ Print model summary."""
        print("Training Model:\n")
        print(self.model.summary())
    
    def plot_loss(self):
        """ Plot training and validation loss. """
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    def example_output(self, number):
        """ Print a number of example predictions. """
        if not self.input_test_data or not self.output_test_data:
            print("No test data")
            return None
        
        # Select a set of test data at random
        num_test_titles = len(self.input_test_data)
        indices = random.sample(range(0,num_test_titles), number)
        
        for i in indices:
            input_sample = self.input_test_data[i]
            output_sample = self.output_test_data[i]
            predicted_text = self.predict(input_sample)
            output_sample_text = self._seq2text(output_sample)
            claim_text = self._seq2text(input_sample, output=False)
            print("Sample of claim text: {}\n".format(claim_text[0:200]))
            o_string = (
                "Predicted title is: {}"
                " \nActual title is: {} \n---"
            )
            print(o_string.format(predicted_text, output_sample_text))
        
    def _seq2text(self, seq, output=True):
        """ Convert a sequence of integers to text."""
        control_ints = [
            self.output_dictionary["stopseq"], 
            self.output_dictionary["startseq"], 
            0
        ]
        text = ''
        for k in seq:
            w = ''
            k = k.astype(int)
            if output:
                # Adapted to take account of different control integers
                if k not in control_ints and k < self.num_decoder_tokens:
                    w = self.output_dictionary_rev[k]
            else:
                # If input
                if k not 0 and k < self.num_decoder_tokens:
                    w = self.input_dictionary_rev[k]
            if w:
                text = text + w + ' '
        return text
        
    def _load_weights(self, filename):
        """ Load weights from file. """
        try:
            # load weights
            self.model.load_weights(self.weights_file)
            print("Loaded weights")
        except:
            print("No existing weights found")
            
    def _save_weights(self):
        """ Load weights from file. """
        self.model.save_weights(self.weights_file, overwrite=True)
    
    def _split_data(self, seed=9, proportion=0.2):
        # We need to split into train and test data
        np.random.seed(seed)
        # split the data into training (80%) and testing (20%)
        (
            self.input_train_data, 
            self.input_test_data, 
            self.output_train_data, 
            self.output_test_data, 
        ) = train_test_split(
                self.input_data, 
                self.output_data, 
                test_size=proportion, 
                random_state=seed
            )
            
    def _get_dataset(self, i, i_end, test=False):
        """ Get a segment of data from i-i_end. """
        if test:
            # Get training dataset
            dataset = self._generate_dataset(
                self.input_test_data,
                self.output_test_data,
                i, i_end)
        else:
            # Get test dataset
            dataset = self._generate_dataset(
                self.input_train_data,
                self.output_train_data,
                i, i_end)
        return dataset
            
    def _train_set(self, training_data, validation_data):
        """ Perform one training iteration model."""
        model_inputs, model_outputs = training_data
        if validation_data:
            callback = self.model.fit(
                model_inputs, 
                model_outputs, 
                validation_data=(validation_data),
                batch_size = self.batch_size, 
                epochs=1
            )
        else:
            callback = self.model.fit(
                model_inputs, 
                model_outputs, 
                batch_size = self.batch_size, 
                epochs=1
            )
        self.train_loss += callback.history['loss']
        self.val_loss += callback.history['val_loss']
        
    def _train_epoch(self):
        """ Training iterations for one epoch - i.e. one pass through the
        training data. """
        num_train_examples = len(self.input_train_data)
        num_test_examples = len(self.input_test_data)
    
        # Loop here to avoid memory issues with the target one hot vector
        for i in range(0, num__train_examples, self.training_set_size):
            if i + self.training_set_size >= num_train_examples:
                i_end = num_train_examples
            else:
                i_end = i + self.training_set_size
                
            # Generate a range for the test data
            i_test = math.floor(i * (num_test_examples/num_train_examples))
            i_test_end = math.floor(i_end * (num_test_examples/num_train_examples))
        
            # Generate small sets of train and test data
            training_data = self._get_dataset(i, i_end, test=False)
            validation_data = self._get_dataset(i_test, i_test_end, test=True)
            # Run a training iteration
            print(
                "Training on batch {0} to {1} of {2}".format(
                        i, 
                        i_end,
                        num_train_examples
                    )
                )
            self._train_set(training_data, validation_data)
        
    def _reset_metrics(self):
        """ Reset tracked training metrics. """
        self.train_loss = []
        self.val_loss = []
        
    def _text2seq(self, input_text, encoder=True):
        """ Convert texts to sequences. """
        if not (self.input_tokenizer and self.output_tokenizer):
            # Generate tokenizers
            self._generate_tokenizers()
        if encoder:
            # Convert using encoder configuration
            tokenizer = self.input_tokenizer
        else:
            # Convert using decoder configuration
            tokenizer = self.output_tokenizer
        encoded_data = tokenizer.texts_to_sequences(input_text)
        return encoder_data
        
    def _pad_seqs(self, input_seq_data, length, padding):
        """ Pad sequences. """
        return pad_sequences(
            input_seq_data, 
            maxlen=length, 
            padding=padding
            )
    
    # Below are methods that will be customised for particular models
    def _predict_from_seq(self, seq):
        """ Predict output sequence from input seq. """
        predicted_output_seq = ""
        return predicted_output_seq
        
    def _start_checks(self):
        """ Checks to run when initialising. """
        pass
        
    def _build_model(self):
        """ Build the model. """
        pass
        
    def _generate_tokenizers(self):
        """ Generate tokenizers for data. """
        self.input_tokenizer = ""
        self.output_tokenizer = ""
        
    def _generate_dataset(self, X, Y, i, i_end):
        """ Fill this in for models."""
        dataset = ""
        return dataset

    
            
