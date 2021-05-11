import numpy as np

import glob
import os
import sys
from shutil import copyfile
from audio_features import extract_melspec
from text_features import extract_bert_features
import scipy.io.wavfile as wav
import joblib as jl


def align(data1, data2):
    """Truncates to the shortest length and concatenates"""
    
    nframes1 = data1.shape[0]
    nframes2 = data2.shape[0]
    if nframes1<nframes2:
        return np.concatenate((data1, data2[:nframes1,:]), axis=1)
    else:
        return np.concatenate((data1[:nframes2,:], data2), axis=1)
        
            
def import_and_pad(files, speech_data, transcript_path):
    """Imports all features and pads them to samples with equal lenth time [samples, timesteps, features]."""
                    
    max_frames = 0
    for file in files:
        print(file)
        
        # compute longest clip
        speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float16)
        transcript_data = np.load(os.path.join(transcript_path, file + '.npy')).astype(np.float16)
        control_data = align(speech_data, transcript_data[:])
        
        if control_data.shape[0]>max_frames:
            max_frames = control_data.shape[0]
            
        n_feats = control_data.shape[1]
            
    out_data = np.zeros((len(files), max_frames, n_feats))
    
    fi=0
    for file in files:        
        # pad to longest clip length
        speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float16)
        transcript_data = np.load(os.path.join(transcript_path, file + '.npy')).astype(np.float16)
        control_data = align(speech_data, transcript_data[:])

        out_data[fi,:control_data.shape[0], :] = control_data
        print("No: " + str(fi) + " File: " + file)
        fi+=1
        
    return out_data

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled
    
if __name__ == "__main__":
    '''
    Converts wav files into features and creates a test dataset.
    '''     
    # Hardcoded preprocessing params and file structure. 
    # Modify these if you want the data in some different format
    test_window_secs = 20
    window_overlap = 0.5
    fps = 20

    data_root = '../data/GENEA/source'
    audiopath = os.path.join(data_root, 'test_audio')
    textpath = os.path.join(data_root, 'test_text')
    processed_dir = '../data/GENEA/with_text/processed_duplicationAsInGesticulator'
    test_dir = '../data/GENEA/with_text/processed_duplicationAsInGesticulator/test'
    
    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(audiopath):
        for file in sorted(f):
            if '.wav' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    print(files)
    if len(files)==0:
        print ("no files found in: " + audiopath)
    
    speech_feat = 'melspec'
    transcript_feat = 'bert'

    # processed data will be organized as following
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    path = os.path.join(test_dir, f'features_{fps}fps')
    speech_path = os.path.join(path, f'{speech_feat}')
    transcript_path = os.path.join(path, f'{transcript_feat}')


    if not os.path.exists(path):
        os.makedirs(path)

        
    # speech features
    if not os.path.exists(speech_path):
        print('Processing speech features...')
        os.makedirs(speech_path)
        extract_melspec(audiopath, files, speech_path, fps)
    else:
        print('Found speech features. skipping processing...')

    # transcript features
    if not os.path.exists(transcript_path):
        print('Processing transcript features...')
        os.makedirs(transcript_path)
        extract_bert_features(textpath, files, transcript_path, fps)
    else:
        print('Found transcript features. skipping processing...')
    
    # Create test dataset
    print("Preparing datasets...")
        
    test_ctrl = import_and_pad(files, speech_path, transcript_path)
    
    ctrl_scaler = jl.load(os.path.join(processed_dir, 'input_scaler.sav'))
    test_ctrl = standardize(test_ctrl, ctrl_scaler)
    
    np.savez(os.path.join(test_dir,f'test_input_{fps}fps.npz'), clips = test_ctrl)
    copyfile(os.path.join(processed_dir, f'data_pipe_{fps}fps.sav'), os.path.join(test_dir,f'data_pipe_{fps}fps.sav'))
    copyfile(os.path.join(processed_dir, 'input_scaler.sav'), os.path.join(test_dir,'input_scaler.sav'))
    copyfile(os.path.join(processed_dir, 'output_scaler.sav'), os.path.join(test_dir,'output_scaler.sav'))
