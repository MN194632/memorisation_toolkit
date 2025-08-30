import os
import re
import json
#from tqdm import tqdm
import random
import datetime
import multiprocessing
import numpy as np
#from librispeech_dataset import LibriSpeech
from collections import Counter
import torch 
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from datasets import load_dataset
import shutil
import librosa
import pandas as pd
from normal_sentences import *
import re
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TTS ENGINE
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

word_counter = Counter()

def get_top_10k_words(dataset):
    json_file = "top_10000_words.json"

    # Check if the file exists to avoid recomputing
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            top_10000_words = json.load(f)
        print("Loaded top 10000 words from JSON file.")
    else:
        # Compute word frequencies
        word_counter = Counter()
        for _, text in dataset:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counter.update(words)

        top_10000_words = word_counter.most_common(10000)

        with open(json_file, "w") as f:
            json.dump(top_10000_words, f)
        print("Computed and saved top 10000 words to JSON file.")

    return top_10000_words

def random_10_words(json_file = "canary_experiments/top_10000_words.json"): 
    # load the top 10000 words from the json file
    with open(json_file, "r") as f:
        top_10000_words = json.load(f)

    words = [word for word, _ in top_10000_words]

    random_words = random.choices(population=words, k=10)

    return random_words

def words2sentence(no_canaries=1,seed=42):
    random.seed(seed)
    word_list = []
    for _ in range(no_canaries):
        word_list.append(random_10_words())
    sentences = [" ".join(words).capitalize() for words in word_list]
    return sentences

def sentence2utterance(sentence,dir,sped_up_rate):
    # make samples folder
    os.makedirs(dir, exist_ok=True)

    # subfolder using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder =  os.path.join(dir,timestamp)
    os.makedirs(subfolder, exist_ok=True)

    inputs = processor(text=sentence, return_tensors="pt").to(DEVICE)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_idx = random.randint(0,7929)

    try:
        sample = embeddings_dataset[speaker_idx]
        speaker_embeddings = torch.tensor(sample["xvector"]).unsqueeze(0).to(DEVICE)
       
        filename = sample["filename"]
        speaker_id = filename.split("_")[2]

        # save the speaker id from CMU arctic database together with sample 
        file_name = "speaker_id.txt"
        file_path = os.path.join(subfolder, file_name)
        with open(file_path, "w") as f:
            f.write(speaker_id)
        
        # # generate the audio
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # # create the canary - sped up version 
        file_name = "canary.wav"
        file_path = os.path.join(subfolder, file_name)
        sample_rate = sped_up_rate * 16000
        sf.write(file_path, speech.cpu().numpy(), samplerate=int(sample_rate))

        # also save the normal version of the audio
        file_name = "speech.wav"
        file_path = os.path.join(subfolder, file_name)
        sf.write(file_path, speech.cpu().numpy(), samplerate=(16000))

        # save sentence as ground truth transcription 
        file_name = "sentence.txt"
        file_path = os.path.join(subfolder, file_name)
        with open(file_path, "w") as f:
            f.write(sentence)
        return 1
    except: 
        print("Error with speaker, skipping sample")
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        return 0 

def holdout_set(sentence,dir,sped_up_rates = [1.5,2,2.5,3,3.5,4]):
    # make samples folder
    os.makedirs(dir, exist_ok=True)

    # subfolder using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder =  os.path.join(dir,timestamp)
    os.makedirs(subfolder, exist_ok=True)

    inputs = processor(text=sentence, return_tensors="pt").to(DEVICE)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_idx = random.randint(0,7929)

    try:
        sample = embeddings_dataset[speaker_idx]
        speaker_embeddings = torch.tensor(sample["xvector"]).unsqueeze(0).to(DEVICE)
        
        filename = sample["filename"]
        speaker_id = filename.split("_")[2]
        
        # save the speaker id from CMU arctic database together with sample 
        file_name = "speaker_id.txt"
        file_path = os.path.join(subfolder, file_name)
        with open(file_path, "w") as f:
            f.write(speaker_id)
        
        # generate the audio
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # create the canary - sped up version
        for sped_up_rate in sped_up_rates: 
            file_name = f"canary_{sped_up_rate}.wav"
            file_path = os.path.join(subfolder, file_name)
            sample_rate = sped_up_rate * 16000
            sf.write(file_path, speech.cpu().numpy(), samplerate=int(sample_rate))

        # also save the normal version of the audio
        file_name = "speech.wav"
        file_path = os.path.join(subfolder, file_name)
        sf.write(file_path, speech.cpu().numpy(), samplerate=(16000))

        # save sentence as ground truth transcription 
        file_name = "sentence.txt"
        file_path = os.path.join(subfolder, file_name)
        with open(file_path, "w") as f:
            f.write(sentence)
        return 1
    except: 
        print("Error with speaker, skipping sample")
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        return 0 

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create canaries with different speed up factors")
    parser.add_argument("--no_canaries", type=int, required=True, help="Number of generated canaries")
    parser.add_argument("--speed", type=int, required=True, help="Speeding factor of canary" )
    parser.add_argument("--samples_dir",type=str,required=True,help="Directory to save the canaries in")
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()

    canaries = words2sentence(no_canaries=args.no_canaries,seed=args.seed) 
    samples_dir = args.samples_dir
    sped_up_rate = args.speed
    succesful_canaries = 0
    random.seed(args.seed)
    for sentence in canaries:
        succesful_canaries += sentence2utterance(sentence, samples_dir,sped_up_rate)
    print(f"Number of generated canaries is: {succesful_canaries} with speed up factor {sped_up_rate}. The are saved in {samples_dir}.")
