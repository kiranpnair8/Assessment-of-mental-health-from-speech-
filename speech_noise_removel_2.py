# Importing relevant libraries
import os
from os import walk
import subprocess


# Using ffmpeg for noise removal
def remove_noise(audio,output):
    command = 'ffmpeg -i {} -af "anlmdn=s=7:p=0.002:r=0.002:m=15, highpass=f=200, lowpass=f=3000" {}'.format(audio, output)
    subprocess.Popen(command, shell = True)

# Removing noise from all audio samples in Data Folder
for path, directories, files in os.walk('./data'):
    for audio in files:
        if audio.endswith(".wav"):
            cleaned = audio.replace(".wav", "_cleaned.wav")
            ip = os.path.join(path,audio)
            op = os.path.join(path,cleaned)
            remove_noise(ip,op)   

