from base_seq2seq import BaseSeq2Seq
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

class CBModel(BaseSeq2Seq):
    """ Model based on Chollet / Brownlee Blog Posts.""" 
    
    def _predict_from_seq(self, seq):
        """ Predict output sequence from input seq. """
        predicted_output_seq = ""
        return predicted_output_seq
        
    def _start_checks(self):
        """ Checks to run when initialising. """
        pass
        
    def _generate_tokenizers(self):
        """ Generate tokenizers for data. """
        self.input_tokenizer = ""
        self.output_tokenizer = ""
        
    def _sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

    def _target_one_hot(self, input_seqs):
        """ Convert a sequence of integers to a one element shifted sequence of one-hot vectors."""
        one_hot_out = np.zeros((len(input_seqs), self.decoder_seq_len, vocab_len))
        for i, sequence in enumerate(input_seqs):
            for t, word_int in enumerate(sequence):
                if t > 0:
                    # Shift decoder target get so it is one ahead
                    one_hot_out[i, t-1, word_int] = 1
        return one_hot_out
    
    # We need to convert this for our present problem - this is similar to our generate dataset above
    # prepare data for the LSTM
    def _generate_dataset(self, X, Y, i, i_end):
        """Return encoder_input_data, decoder_input_data, and decoder_target_data, latter as one-hot"""
        encoder_input_data = X[i:i_end]
        decoder_input_data = Y[i:i_end]
        decoder_target_data = self.target_one_hot(decoder_input_data)
        return ([encoder_input_data, decoder_input_data], decoder_target_data)
    
    # returns train, inference_encoder and inference_decoder models
    def _build_model(self):
        # define training encoder
        # Define an input sequence and process it.
        self._load_shared_embedding()
        encoder_inputs = Input(shape=(None,))
        Shared_Embedding = Embedding(
            output_dim=self.word_embedding_size, 
            input_dim=self.num_encoder_tokens, 
            weights=[self.embedding_matrix]
        )
        encoder_embedding = Shared_Embedding(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(self.latent_dim, return_state=True)(encoder_embedding)
        encoder_states = [state_h, state_c]
    
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        # Possibly share the embedding below
        decoder_embedding = Shared_Embedding(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        print("Building models for training and inference")
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
        # define inference encoder
        self.encoder_model = Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Need to adjust this line for the embedding
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        print("Compiling model")
        # Can we use sparse_C_C if we are using integers? https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
