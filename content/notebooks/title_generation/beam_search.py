# Imports
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


class BeamSearchMixin:

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
            for i, seq in enumerate(sequences):
                # Get most recent score
                prev_score = seq[1]
                # Get current token sequence
                ans_partial = seq[0]
                # Predict token probabilities
                yhat = self.infdec.predict([input_seq, ans_partial])
                # Unpack yhat array of probabilities
                yhat = yhat[0, :]
                new_scores[i*ndt:(i+1)*ndt] = prev_score+-np.log(yhat)

            # Select top k scores from bottom of list
            top_k_indices = np.argsort(new_scores)[:k].tolist()
            new_sequences = []
            for index in top_k_indices:
                seq_select = int(index/ndt)
                new_token = index % ndt  # This is the token index
                # Update the partial answer
                new_ans_partial = np.zeros(
                        (1, self.decoder_seq_length),
                        dtype=np.int16
                    )
                new_ans_partial[0, 0:-1] = sequences[seq_select][0][0, 1:]
                # This then adds the newly decoded word onto the end
                new_ans_partial[0, -1] = new_token
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
            seqs = self._predict_from_seq(self, input_sample, k)
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
