"""This module contains a class to model a sequence to sequence
model based on blogposts from Chollet and Brownlee
"""

from base_seq2seq import SharedGlove
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding


def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array. """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class CBModel(SharedGlove):
    """ Model based on Chollet / Brownlee Blog Posts."""

    def _start_checks(self):
        """ Checks to run when initialising. """
        pass

    def _predict_from_seq(self, seq, temp=1.0):
        """ Predict output sequence from input seq. """
        # encode
        state = self.infenc.predict(seq)
        # start of sequence input
        target_seq = np.array([self.output_dictionary["startseq"]])
        # collect predictions
        output = list()
        for _ in range(self.decoder_seq_length):
            # predict next char
            yhat, input_h, input_c = self.infdec.predict([target_seq] + state)
            # update state
            state = [input_h, input_c]
            # update target sequence - this needs to be the argmax
            next_int = sample(yhat[0, 0, :], temp)
            output.append(next_int)
            # It seems like we throw a lot of information away here
            # can we build in the probabilities?
            target_seq = np.array([next_int])
            # Check for stopping character
            if next_int == self.output_dictionary["stopseq"]:
                break
        return output

    def _target_one_hot(self, input_seqs):
        """ Convert a sequence of integers to a one element shifted
        sequence of one-hot vectors."""
        length = len(input_seqs)
        one_hot_out = np.zeros(
            (length, self.decoder_seq_length, self.num_decoder_tokens)
            )
        for i, sequence in enumerate(input_seqs):
            for timestamp, word_int in enumerate(sequence):
                if timestamp > 0:
                    # Shift decoder target get so it is one ahead
                    one_hot_out[i, timestamp-1, word_int] = 1
        return one_hot_out

    def _generate_dataset(self, X, Y, i, i_end):
        """Return encoder_input_data, decoder_input_data, and
        decoder_target_data, latter as one-hot"""
        encoder_input_data = X[i:i_end]
        decoder_input_data = Y[i:i_end]
        decoder_target_data = self._target_one_hot(decoder_input_data)
        return ([encoder_input_data, decoder_input_data], decoder_target_data)

    # returns train, inference_encoder and inference_decoder models
    def _build_model(self):
        # define training encoder
        # Define an input sequence and process it.
        self._load_shared_embedding()
        encoder_inputs = Input(shape=(None,))
        shared_embedding = Embedding(
            output_dim=self.word_embedding_size,
            input_dim=self.num_encoder_tokens,
            weights=[self.embedding_matrix]
        )
        encoder_embedding = shared_embedding(encoder_inputs)
        _, state_h, state_c = \
            LSTM(self.latent_dim, return_state=True)(encoder_embedding)
        encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        # Possibly share the embedding below
        decoder_embedding = shared_embedding(decoder_inputs)
        decoder_lstm = \
            LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = \
            decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        print("Building models for training and inference")
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` > `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # define inference encoder
        self.infenc = Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Need to adjust this line for the embedding
        decoder_outputs, state_h, state_c = \
            decoder_lstm(
                decoder_embedding, initial_state=decoder_states_inputs
                )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.infdec = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        print("Compiling model")
        # Can we use sparse_C_C if we are using integers?
        # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['acc']
        )
