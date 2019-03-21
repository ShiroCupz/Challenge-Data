# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense, Layer, GlobalMaxPooling1D, Conv1D, Concatenate
from keras import initializers, regularizers, constraints

# =================================================================== #

def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

# =================================================================== #

def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh',
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)

# =================================================================== #

class AttentionWithContext(Layer):
    """
    initially taken from: https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

# =================================================================== #

def build_model_HAN_simple(sents_shape, embeddings, n_units, n_outputs, drop_rate, is_GPU) :

    sent_ints = Input(shape=(sents_shape,))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=sents_shape,
                        trainable=False,
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr, n_units, is_GPU)
    sent_att_vec,word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)

    preds = Dense(units=n_outputs,
                  activation='softmax')(sent_att_vec_dr)

    model = Model(sent_ints, preds)

    return model

# =================================================================== #

def build_model_HAN(docs_shape, embeddings, n_units, n_outputs, drop_rate, is_GPU) :

    sent_ints = Input(shape=(docs_shape[2],))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_shape[2],
                        trainable=False,
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU)
    sent_att_vec,word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)

    sent_encoder = Model(sent_ints,sent_att_vec_dr)

    doc_ints = Input(shape=(docs_shape[1],docs_shape[2],))

    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU)
    doc_att_vec,sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

    preds = Dense(units=n_outputs,
                  activation='softmax')(doc_att_vec_dr)

    model = Model(doc_ints,preds)

    return model

# =================================================================== #

def cnn_branch(n_filters,k_size,d_rate,my_input):
    return Dropout(d_rate)(GlobalMaxPooling1D()(Conv1D(filters=n_filters,
                                                       kernel_size=k_size,
                                                       activation='relu')(my_input)))

# =================================================================== #

def build_CNN(doc_shape, embeddings, nb_branches, nb_filters, filter_sizes, drop_rate, n_outputs) :

    doc_ints = Input(shape=(None,))

    doc_wv = Embedding(embeddings.shape[0],
                       embeddings.shape[1],
                       weights=[embeddings],
                       input_length=doc_shape,
                       trainable=True)(doc_ints)

    doc_wv_dr = Dropout(drop_rate)(doc_wv)

    branch_outputs = []
    for idx in range(nb_branches):
        branch_outputs.append(cnn_branch(nb_filters, filter_sizes[idx], drop_rate, doc_wv_dr))

    concat = Concatenate()(branch_outputs)

    preds = Dense(units=n_outputs,
                  activation='softmax')(concat)

    model = Model(doc_ints,preds)

    return model

# =================================================================== #
