import glob
import os
import librosa
import numpy as np

def windows(data, window_size):
    start = 0
    while start + window_size <= len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 50, frames = 50):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('/')[2].split('-')[1]
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[int(start):int(end)]) == window_size):
                    signal = sound_clip[int(start):int(end)]
                    melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                    logspec = librosa.amplitude_to_db(melspec)
                    log_specgrams.append(logspec)
                    labels.append(label)
     
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    
    return np.array(log_specgrams), np.array(labels,dtype = np.int)

parent_dir = 'audio'

tr_sub_dirs= ['fold1']
tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
np.save('sound_training_features',tr_features)
np.save('sound_training_labels',tr_labels)

ts_sub_dirs= ['fold10']
ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
np.save('sound_test_features',ts_features)
np.save('sound_test_labels',ts_labels)