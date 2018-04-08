from base_seq2seq import SharedGlove
import numpy as np
from keras.preprocessing import text
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.layers import concatenate
import os

class LudwigModel(SharedGlove):
    
    # I think we can reuse some of this?
    def _predict_from_seq(self, seq):
        """ Predict output sequence from input seq. """
        # reformat input seq
        input_seq = np.zeros((1, self.encoder_seq_length))
        input_seq[0, :] = seq
        flag = 0
        prob = 1
        ans_partial = np.zeros((1, self.decoder_seq_length))
        # Add start token integer to end of ans_partial input - initially [0,0,...BOS]
        ans_partial[0, -1] = self.output_dictionary["startseq"]  #  the index of the symbol BOS (begin of sentence)
        # Minus one as last entry is stopseq? But can be full length
        for k in range(self.decoder_seq_length):
            # Remember to set infdec as model when building model
            ye = self.infdec.predict([input_seq, ans_partial])
            # yel = ye[0,:]
            # p = np.max(yel) - only really useful as a relative measure of confidence
            mp = np.argmax(ye)
            # It is this line that sets how our training data should be arranged - need to change both
            # the line below shifts the existing ans_partial by 1 to the left - [0, 0, ..., BOS, 0]
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            # This then adds the newly decoded word onto the end of ans_partial
            ans_partial[0, -1] = mp
            if mp == self.output_dictionary["stopseq"]:  #  the index of the symbol EOS (end of sentence)
                break
            #if flag == 0:    
            #    prob = prob * p
        predicted_output_seq = ans_partial[0]
        return predicted_output_seq
        
    def _generate_dataset(self, X, Y, i, i_end):
        """ Generate the data for training/validation from X and Y.
        i_end is the end of the set, i is the start."""
        set_size = 0
        limit_list = list()
        for sent in Y[i:i_end]:
            # Edited below to use integer value of EOS symbol
            EOS = self.output_dictionary["stopseq"]
            limit = np.where(sent==EOS)[0][0]  #  the position of the symbol EOS
            set_size += limit + 1
            limit_list.append(limit)
   
        # Generate blank arrays for the set
        I_1 = np.zeros((set_size, self.encoder_seq_length))
        I_2 = np.zeros((set_size, self.decoder_seq_length))
        # This below is a big array
        Y_set = np.zeros((set_size, self.num_decoder_tokens))
        count = 0
        for l in range(0, (i_end - i)):
            limit = limit_list[l]
            # We only need to create examples up to the length of the title 
            for m in range(1, limit+1):
                # Generate our one-hot y out
                one_hot_out = np.zeros((1, self.num_decoder_tokens))
                # This builds our one-hot generation into our training loop
                # The l and m respectively iterate through the samples and the output sequence elements
                one_hot_out[0, Y[l+i][m]] = 1
                # Create a blank row/array for a partial input for our summary model - this is fed into the decoder
                partial_input = np.zeros((1, self.decoder_seq_length))
                partial_input[0, -m:] = Y[l+i][0:m]
                # This fills in each sample of the training data, i.e. count increments up to set size
                I_1[count, :] = X[l+i]
                I_2[count, :] = partial_input
                Y_set[count, :] = one_hot_out
                count += 1
                    
            # Shuffle the I_1, I_2 and Y_set vectors for better training - trick from RL
            # - see here - np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X);
            indices = np.random.permutation(I_1.shape[0])
            np.take(I_1, indices, axis=0, out=I_1)
            np.take(I_2, indices, axis=0, out=I_2)
            np.take(Y_set, indices, axis=0, out=Y_set)
        return ([I_1, I_2], Y_set)
        
    def _start_checks(self):
        """ Checks to run when initialising. """
        pass
        
    def _build_model(self):
        """ Build the model. """
        print("Building model")
        self._load_shared_embedding()
        # source text input model
        inputs1 = Input(shape=(self.encoder_seq_length,))
        #am1 = Embedding(X_vocab_len, 128)(inputs1)
        Shared_Embedding = Embedding(
            output_dim=self.word_embedding_size, 
            input_dim=self.num_encoder_tokens,
            weights=[self.embedding_matrix], 
            input_length=self.encoder_seq_length
        )
        am1 = Shared_Embedding(inputs1)
        am2 = LSTM(self.latent_dim)(am1)
        # summary input model
        inputs2 = Input(shape=(self.decoder_seq_length,))
        sm1 = Shared_Embedding(inputs2)
        sm2 = LSTM(self.latent_dim)(sm1)
        # decoder output model
        decoder1 = concatenate([am2, sm2])
        outputs = Dense(self.num_decoder_tokens, activation='softmax')(decoder1)
        # tie it together [article, summary] [word]
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        self.infdec = self.model
        print("Compiling model")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
    
