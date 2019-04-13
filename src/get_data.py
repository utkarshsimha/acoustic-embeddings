import numpy as np
import os
from scikits.audiolab import Sndfile
from sklearn.externals import joblib
import cPickle as pickle
from python_speech_features import sigproc
import pdb

def getWordSegmentation( path ):
    """
    Function to get the segmented frame data of the audio wave and its
    corresponding segmented word transcription
    """

    f_names = []
    frame_data = []
    word_seg = []
    # Find all .wav files
    for root, dirs, files in os.walk( path ):
        for f_name in files:
            f_name, ext = f_name.split(".")
            if ext == "WAV":
                f_names.append( os.path.join( root, f_name ) )

    for f_name in f_names:
        audio_file = Sndfile( f_name + ".WAV", "r" )

        # Get audio
        audio = audio_file.read_frames( audio_file.nframes )

        # Get transcription
        word_segmentation = open( f_name + ".WRD", "r" ).read().strip().split("\n")
        temp_seg = []
        temp_frames = []
        for word in word_segmentation:
            # Get corresponding time frame of audio wave
            start_time, end_time, word = word.split(" ")
            temp_seg.append( ( word, ( start_time, end_time ) ) )
            frame = audio[ int( start_time ) : int( end_time ) ]
            temp_frames.append( frame )

        frame_data.append( temp_frames )
        word_seg.append( temp_seg )
            
    # Store all the data
    print "Taking a dump.."
    joblib.dump( np.asarray( frame_data ), "frame_data.npy" )
    pickle.dump( word_seg, open( "word_seg.p", "w" ) )

if __name__ == '__main__':
    getWordSegmentation( "./speech_data" )

