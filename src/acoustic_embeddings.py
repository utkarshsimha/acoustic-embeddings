import tensorflow as tf
import numpy as np
from sklearn.externals import joblib
import cPickle as pickle
import spacy
import os
nlp = spacy.load("en")
from keras.layers import Input, Conv1D, Dense, Layer, Permute
from keras.layers import Lambda, Reshape, Conv2DTranspose
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import metrics
import pdb
from get_data import getWordSegmentation

def loadData(path="./speech-data"):
    if not os.path.exists("frame_data.npy") or \
       not os.path.exists("word_seg.p"):
        print "Processing data.."
        getWordSegmentation(path)

    print "Loading data.."
    frame_data = joblib.load( "frame_data.npy" )
    word_seg = pickle.load( open( "word_seg.p", "r" ) )
    a_list, w_list = [], []
    for sample in zip( frame_data, word_seg ):
        a_list.extend( sample[0] )
        w_list.extend( sample[1] )

    acoustic = np.asarray( a_list )
    acoustic = pad_sequences( acoustic )
    word = np.asarray( [ nlp(u""+word).vector for word,_ in w_list ] )
    print "Acoustic: {}; Word: {}".format( acoustic.shape, word.shape )
    return acoustic, word

def getModel( inp_seq_len, out_seq_len, latent_dim=128, batch_size=32 ):
    inp = Input( shape=( 1, inp_seq_len ) )

    ''' Convolution layers '''
    encode = Conv1D( 1024, 2, padding='causal' )( inp )
    encode = Conv1D( 512, 2, padding='causal' )( encode )
    encode = Lambda( K.squeeze, arguments={"axis":1} )( encode )

    ''' Mean and variance '''
    z_mean = Dense( latent_dim, activation='linear', name="mean" )( encode )
    z_log_var = Dense( latent_dim, activation='linear', name="var" )( encode )

    ''' Sampling z from the gaussian '''
    def sample_z( args ):
        z_mean, z_log_var = args
        epsilon = K.random_normal( shape=( latent_dim, ),\
                mean = 0., stddev=1 )
        return z_mean + K.exp( z_log_var / 2 ) * epsilon
    z = Lambda( sample_z )( [ z_mean, z_log_var ] )

    ''' Learning a latent representation '''
    decode = Dense( latent_dim )( z )
    decode = Reshape( ( -1, 1, 1 ) )( decode )
    decode = Permute( ( 2, 3, 1 ) )( decode )

    ''' Deconvolution layers'''
    decode = Conv2DTranspose( out_seq_len, 1 )( decode )
    decode = Lambda( K.squeeze, arguments={"axis":1} )( decode )
    out = Lambda( K.squeeze, arguments={"axis":1} )( decode )

    ''' Create model '''
    model = Model( inp, out )
    print model.summary()
    model.compile( loss="kullback_leibler_divergence", optimizer="adam" )

    return model


if __name__ == '__main__':
    acoustics, words = loadData()
    model = getModel( words.shape[1], acoustics.shape[1] )
    words = np.expand_dims( words, 1 )
    model.fit( words, acoustics, epochs=2, batch_size=128 )
    model.save("vae_model.hd5")

