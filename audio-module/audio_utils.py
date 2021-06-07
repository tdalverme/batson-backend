from collections import defaultdict
from pydub import AudioSegment
import speech_recognition as sr
from moviepy.editor import *
from pydub.utils import make_chunks
import shutil
import sys
import os

backend_folder = os.getcwd()

if sys.platform == 'darwin':
    AudioSegment.converter = os.path.join(backend_folder, 'ffmpeg')
else:
   AudioSegment.converter = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

r = sr.Recognizer()

base_path = os.path.dirname(__file__)

# returns list of stopwords read from file
def load_stoplists(path):
    with open(path, "r") as f:
        stoplist = f.readlines()

    stoplist = [x.strip() for x in stoplist]

    return stoplist

# removes stopwords from texts
def remove_stopwords(texts, stoplist):
    texts = [[word for word in text.lower().split() if word not in stoplist] for text in texts]

    return texts

# removes words with frequency lower than freq
def remove_low_freq(texts, freq):
    frequency = defaultdict(int)

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > freq] for text in texts]

    return texts

def separate_audio(path_video):
    if(path_video.endswith('.mp4')):
        audio = AudioFileClip(path_video)
        audio.write_audiofile(path_video.replace('.mp4', '.wav'), 44100, 2, 2000,"pcm_s16le")
    
    return path_video.replace('.mp4', '.wav')

# prints the recognized audio to text_audio.txt
def audio_to_text(path, verbose=False):
    if verbose:
        print("[REC-AUD] Reconociendo audio...")

    # open the audio file stored as a wav file
    audio = AudioSegment.from_wav(path)
  
    # open a file where we will concatenate and store the recognized text
    fh = open(os.path.join(base_path, "text_audio.txt"), "w+")
          
    # split track in chunks of 4 seconds
    chunks = make_chunks(audio, 5000)
    if verbose:
        print("[REC-AUD] Audio separado en chunks.")

    # create a directory to store the audio chunks.
    try:
        os.mkdir('audio_chunks')
    except(FileExistsError):
        pass
  
    # move into the directory to
    # store the audio files.
    os.chdir('audio_chunks')
  
    i = 0
    # process each chunk
    for chunk in chunks:
        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration = 10)
  
        # add 0.5 sec silence to beginning and 
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent
  
        # export audio chunk and save it in 
        # the current directory.
        #print("saving chunk{0}.wav".format(i))
        # specify the bitrate to be 192 k
        audio_chunk.export("chunk{0}.wav".format(i), bitrate ='192k', format ="wav")
  
        # the name of the newly created chunk
        filename = 'chunk'+str(i)+'.wav'
  
        #print("Processing chunk "+str(i))
  
        # get the name of the newly created chunk
        # in the AUDIO_FILE variable for later use.
        file = filename
  
        # create a speech recognition object
        r = sr.Recognizer()
  
        # recognize the chunk
        with sr.AudioFile(file) as source:
            # remove this if it is not working
            # correctly.
            r.adjust_for_ambient_noise(source)
            audio_listened = r.record(source)
  
        try:
            # try converting it to text
            rec = r.recognize_google(audio_listened)
            # write the output to the file.
            fh.write(rec+". ")
            if verbose:
                print("[REC-AUD] Chunk {} procesado.".format(i))
  
        # catch any errors.
        except sr.UnknownValueError:
            pass
            if verbose:
                print("[REC-AUD] No se entendio el audio del chunk {}.".format(i))
  
        except sr.RequestError as e:
            if verbose:
                print("[REC-AUD] Could not request results. check your internet connection")
  
        i += 1
    
    os.chdir('..')
    shutil.rmtree('audio_chunks')
    if verbose:
        print("[REC-AUD] Audio reconocido.")

############################################################################################