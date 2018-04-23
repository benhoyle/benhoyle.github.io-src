import string
import random
import numpy as np

from base_seq2seq import MemEfficientS2S
from keras.layers import GRU, Input, Dense
from keras.models import Model
from keras.utils import to_categorical


def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array. """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class CharS2S(MemEfficientS2S):
    """ Class to model a character decoder from an embedding."""
    def __init__(
        self,
        encoder_states=None,
        decoder_seqs=None,
        decoder_seq_length=None,
        latent_dim=100,
        weights_file=None,
        training_set_size=50000,  # Due to memory we need to train in sets
        batch_size=128
    ):
        # If encoder / decoder seq_length = none we can set based on data
        self.decoder_seq_length = decoder_seq_length
        # Likewise if the x_tokens are None we can set later based on the data
        # Setup data and dictionaries
        # We can probably build a general 'vocab' object
        self.vocab = ["_PAD_", "_SOW_", "_EOW_"] + list(string.ascii_lowercase)
        self.num_decoder_tokens = len(self.vocab)
        # Build dictionaries
        self.output_dictionary_rev = {i: c for i, c in enumerate(self.vocab)}
        self.output_dictionary = {
            v: k
            for k, v in self.output_dictionary_rev.items()
            }

        self.latent_dim = latent_dim
        self.training_set_size = training_set_size
        self.batch_size = batch_size
        self.input_data = encoder_states
        self.output_data = decoder_seqs
        self.model = None
        self._build_model()
        # Build model here?
        if weights_file and self.model:
            self.weights_file = weights_file
            self._load_weights()
        else:
            self.weights_file = "weights.hdf5"
        self._reset_metrics()
        # Create test and training data
        self._split_data()

    def _build_model(self):
        """ Build the model. """
        print("Building model")
        encoded_state = Input(shape=(self.latent_dim,), name="EncodedState")
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name="DecoderInputs")
        decoder_gru = GRU(self.latent_dim, return_sequences=True, return_state=True, name="Decoder")
        decoder_outputs, decoder_state = decoder_gru(decoder_inputs, initial_state=encoded_state)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name="VocabProjection")
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = Model([encoded_state, decoder_inputs], decoder_outputs)

        # We also need an inference model
        self.infdec = Model(inputs=[encoded_state, decoder_inputs], outputs=[decoder_outputs, decoder_state])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['acc']
        )

    def _text2seq(self, input_text, encoder=True):
        """ Convert texts to sequences. """
        word = input_text[0]
        try:
            encoded_data = (
                [self.output_dictionary["_SOW_"]] +
                [self.output_dictionary[c] for c in word] +
                [self.output_dictionary["_EOW_"]]
            )
        except KeyError:
            print("Word contains foreign characters")
            encoded_data = None
        return encoded_data

    def predict(self, input_data):
        """ Predict output characters from input word embedding. """
        # if input is text convert to sequence
        predicted_output_seq = self._predict_from_seq(input_data)
        predicted_text = self._seq2text(predicted_output_seq)
        return predicted_text

    def _predict_from_seq(self, seq, temp=1.0):
        """ Predict output text from input text. """
        state = seq.reshape(1, self.latent_dim)
        # start of sequence input
        target_seq = to_categorical(self.output_dictionary["_SOW_"], num_classes=self.num_decoder_tokens).reshape(1, 1, self.num_decoder_tokens)
        # collect predictions
        output = list()
        for _ in range(self.decoder_seq_length):
            # predict next char
            yhat, state = self.infdec.predict([state, target_seq])
            # update state
            next_int = sample(yhat[0, 0, :], temp)
            output.append(next_int)
            target_seq = to_categorical(next_int, num_classes=self.num_decoder_tokens).reshape(1, 1, self.num_decoder_tokens)
            # Check for stopping character
            if next_int == self.output_dictionary["_EOW_"]:
                break
        return output

    def _generate_dataset(self, X, Y, i, i_end):
        """ Generate a dataset. """
        encoded_states = np.array(X[i:i_end])
        decoder_input_data = Y[i:i_end]
        length = len(decoder_input_data)
        decoder_in_seqs = np.zeros(
            (length, self.decoder_seq_length, self.num_decoder_tokens)
            )
        decoder_out_seqs = np.zeros(
            (length, self.decoder_seq_length, self.num_decoder_tokens)
            )
        for i, sequence in enumerate(decoder_input_data):
            for timestamp, word_int in enumerate(sequence[0:self.decoder_seq_length]):
                decoder_in_seqs[i, timestamp, word_int] = 1
                if timestamp > 0:
                    # Shift decoder target get so it is one ahead
                    decoder_out_seqs[i, timestamp-1, word_int] = 1
        return ([encoded_states, decoder_in_seqs], decoder_out_seqs)

    def train(self, epochs=1, reset_metrics=False):
        """ Train in batches on all data with validation. """
        if reset_metrics:
            self._reset_metrics()
        for e in range(0, epochs):
            print("Training for epoch {0}".format(e))
            self._train_epoch()
            self._save_weights()
        self.plot_loss()

    def _seq2text(self, seq, output=True):
        """ Convert a sequence of integers to text.
        This can be used for both word and char models - just need to abstract
        the control_ints
        """
        control_ints = [
            self.output_dictionary["_SOW_"],
            self.output_dictionary["_EOW_"],
            self.output_dictionary["_PAD_"]
        ]
        text = ''
        for k in seq:
            w = ''
            if not isinstance(k, int):
                k = k.astype(int)
            if output:
                # Adapted to take account of different control integers
                if k not in control_ints and k < self.num_decoder_tokens:
                    w = self.output_dictionary_rev[k]
            else:
                # If input
                if k != 0 and k < self.num_decoder_tokens:
                    w = self.input_dictionary_rev[k]
            if w:
                text = text + w + ' '
        return text

    def example_output(self, number):
        """ Print a number of example predictions. """
        # Select a set of test data at random
        num_test_titles = len(self.input_test_data)
        indices = random.sample(range(0, num_test_titles), number)
        print("------------------------------------------")
        for i in indices:
            input_sample = self.input_test_data[i]
            output_sample = self.output_test_data[i]
            predicted_text = self.predict(input_sample)
            output_sample_text = self._seq2text(output_sample)
            o_string = (
                "Predicted word is: {}"
                "\nActual word is: {} \n---"
            )
            print(o_string.format(predicted_text, output_sample_text))
