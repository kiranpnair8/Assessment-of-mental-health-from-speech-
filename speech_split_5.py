# Importing relevant libraries
import os
from os import walk
import pandas as pd
import shutil
import librosa
import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math



# Splitting audio files in train, dev and test
train_df = pd.DataFrame()
dev_df = pd.DataFrame()
test_df = pd.DataFrame()

for path, directories, files in os.walk('./'):
    for audio in files:
        if audio == "train_split_Depression_AVEC2017.csv":
            train_df = pd.read_csv(os.path.join(path,audio))
            train_df = train_df[['Participant_ID','Gender','PHQ8_Binary']]
            train_df['Participant_ID'] = train_df['Participant_ID'].map(lambda x: str(x) + '_AUDIO_final.wav')
        if audio == "dev_split_Depression_AVEC2017.csv":
            dev_df = pd.read_csv(os.path.join(path,audio))
            dev_df = dev_df[['Participant_ID','Gender','PHQ8_Binary']]
            dev_df['Participant_ID'] = dev_df['Participant_ID'].map(lambda x: str(x) + '_AUDIO_final.wav')
        if audio == "test_split_Depression_AVEC2017.csv":
            test_df = pd.read_csv(os.path.join(path,audio))
            
            test_df['participant_ID'] = test_df['participant_ID'].map(lambda x: str(x) + '_AUDIO_final.wav')


train_path = "C:\\project main\\Speech_Dirization\\train"
dev_path = "C:\\project main\\Speech_Dirization\\dev"
test_path = "C:\\project main\\Speech_Dirization\\test"
train_list = list(train_df['Participant_ID'])
dev_list = list(dev_df['Participant_ID'])
test_list = list(test_df['participant_ID'])
for path, directories, files in os.walk('./'):
    for audio in files:
        try:
            if audio.endswith("_final.wav"):
                if audio in train_list:
                    shutil.move(os.path.join(path,audio), train_path) 
                if audio in dev_list:
                    shutil.move(os.path.join(path,audio), dev_path)
                if audio in test_list:
                    shutil.move(os.path.join(path,audio), test_path) 
        except:
            pass  



train_df.to_csv("train_df.csv", index= False)


# Getting an idea of the data imbalance
print("The old number of non-depressed people is",train_df['PHQ8_Binary'].value_counts()[0])
print("The old number of depressed people is",train_df['PHQ8_Binary'].value_counts()[1])

# Speed tuning for audio augmentation
af = train_df[train_df['PHQ8_Binary'] == 1]
af_list = list(af['Participant_ID'])
i = 493
newauds = []
for path, directories, files in os.walk('./data'):
    for audio in files:
        if audio in af_list:
            i = i + 1
            speed_rate = np.random.uniform(0.7,1.3) 
            aud, fs = librosa.load(os.path.join(path,audio))
            aud_tuned = librosa.effects.time_stretch(aud,rate= speed_rate)
            scipy.io.wavfile.write(os.path.join(path, "{}_AUDIO_final.wav".format(i)), fs, aud_tuned)
            newauds.append("{}_AUDIO_final.wav".format(i))

# Making changes in the dataframes
bf = pd.DataFrame(newauds, columns = ['Participant_ID'])
bf.reset_index(drop=True, inplace=True)
af.reset_index(drop=True, inplace=True)
newf = pd.concat([bf,af[['Gender','PHQ8_Binary']]], ignore_index=True, axis = 1)
newf = newf.dropna()
newf = newf.rename(columns = {0:'Participant_ID', 1:'Gender', 2:'PHQ8_Binary'})
train_df = pd.concat([train_df, newf])

# Getting an idea of the data imbalance
print("The new number of non-depressed people is",train_df['PHQ8_Binary'].value_counts()[0])
print("The new number of depressed people is",train_df['PHQ8_Binary'].value_counts()[1])





# Spectrogram Conversion of train Audios
spec_list = list(train_df['Participant_ID'])
traini_df = pd.DataFrame(columns = train_df.columns)
for path, directories, files in os.walk('./'):
    for audio in files:
        if audio in spec_list:
            gen = train_df.loc[train_df['Participant_ID'] == audio, 'Gender'].iloc[0] 
            phq = train_df.loc[train_df['Participant_ID'] == audio, 'PHQ8_Binary'].iloc[0]
            sample_rate, samples = wavfile.read(os.path.join(path,audio))
            for i in range(1,math.floor(len(samples)/(30*sample_rate))):
                frequencies, times, spectrogram = signal.spectrogram(samples[30*(i - 1)*sample_rate:30*i*sample_rate], sample_rate)
                plt.pcolormesh(times, frequencies, np.log(spectrogram))
                plt.savefig(os.path.join(path,"{}_{}.png".format(audio,i)), bbox_inches='tight')
                traini_df.loc[len(traini_df.index)] = ["{}_{}.png".format(audio,i), gen, phq]

# Spectrogram Conversion of dev Audios
spec_list = list(dev_df['Participant_ID'])
devi_df = pd.DataFrame(columns = dev_df.columns)
for path, directories, files in os.walk('./'):
    for audio in files:
        if audio in spec_list:
            gen = dev_df.loc[dev_df['Participant_ID'] == audio, 'Gender'].iloc[0] 
            phq = dev_df.loc[dev_df['Participant_ID'] == audio, 'PHQ8_Binary'].iloc[0]
            sample_rate, samples = wavfile.read(os.path.join(path,audio))
            for i in range(1,math.floor(len(samples)/(30*sample_rate))):
                frequencies, times, spectrogram = signal.spectrogram(samples[30*(i - 1)*sample_rate:30*i*sample_rate], sample_rate)
                plt.pcolormesh(times, frequencies, np.log(spectrogram))
                plt.savefig(os.path.join(path,"{}_{}.png".format(audio,i)), bbox_inches='tight')
                devi_df.loc[len(devi_df.index)] = ["{}_{}.png".format(audio,i), gen, phq]                

# Spectrogram Conversion of test Audios
spec_list = list(test_df['participant_ID'])
testi_df = pd.DataFrame(columns = test_df.columns)

for path, directories, files in os.walk('./'):
    for audio in files:
        if audio in spec_list:
            gen = test_df.loc[test_df['participant_ID'] == audio, 'Gender'].iloc[0] 
            # phq = test_df.loc[test_df['participant_ID'] == audio, 'PHQ8_Binary'].iloc[0]
            sample_rate, samples = wavfile.read(os.path.join(path,audio))
            for i in range(1,math.floor(len(samples)/(30*sample_rate))):
                frequencies, times, spectrogram = signal.spectrogram(samples[30*(i - 1)*sample_rate:30*i*sample_rate], sample_rate)
                plt.pcolormesh(times, frequencies, np.log(spectrogram))
                plt.savefig(os.path.join(path,"{}_{}.png".format(audio,i)), bbox_inches='tight')
                # testi_df.loc[len(testi_df.index)] = ["{}_{}.png".format(audio,i), gen, phq]
                testi_df.loc[len(testi_df.index)] = ["{}_{}.png".format(audio,i), gen ]
