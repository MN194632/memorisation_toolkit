import os
import re
import json
import random
import datetime
import multiprocessing
import numpy as np
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TTS ENGINE
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

def load_agentlans_sentences(num_samples=200, seed=86):
    random.seed(seed)
    
    ds = load_dataset("agentlans/high-quality-english-sentences", split=f'train[:{num_samples*3}]')
    
    sentences = []
    for item in ds:
        if len(sentences) != num_samples:
            text = item['text'].strip()
            if 40 < len(text) < 70:  # Filter reasonable length sentences
                if not has_numbers_or_special_chars(text):
                    sentences.append(text)
    
    print(f"Loaded {len(sentences)} sentences from agentlans dataset")
    return sentences

def sentence2utterance(sentence, dir, sped_up_rate):
    os.makedirs(dir, exist_ok=True)

    # subfolder using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(dir, timestamp)
    os.makedirs(subfolder, exist_ok=True)

    inputs = processor(text=sentence, return_tensors="pt").to(DEVICE)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_idx = random.randint(0, 7929)

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
    except Exception as e: 
        print(f"Error with speaker: {e}, skipping sample")
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        return 0 

def holdout_set(sentence, dir, sped_up_rates=[1.5, 2, 2.5, 3, 3.5, 4]):
    os.makedirs(dir, exist_ok=True)

    # subfolder using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(dir, timestamp)
    os.makedirs(subfolder, exist_ok=True)

    inputs = processor(text=sentence, return_tensors="pt").to(DEVICE)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_idx = random.randint(0, 7929)

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
    except Exception as e: 
        print(f"Error with speaker: {e}, skipping sample")
        if os.path.exists(subfolder):
            shutil.rmtree(subfolder)
        return 0 

    
def has_numbers_or_special_chars(sentence):
    # Check if sentence contains anything other than letters, spaces, and periods
    return bool(re.search(r'[^a-zA-Z\s.]', sentence))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create holdout set from all agentlans sentences")
    parser.add_argument("--speed", type=int, required=True, help="Speeding factor of canary" )
    parser.add_argument("--samples_dir",type=str,required=True,help="Directory to save the canaries in")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--no_canaries", type=int, required=True, help="Number of generated canaries")

    args = parser.parse_args()

    canaries = load_agentlans_sentences(num_samples=args.no_canaries, seed=args.seed)
    canaries = [sentence.replace('.', '') for sentence in canaries]
    
    print(f"Processing {len(canaries)} sentences from agentlans dataset")

    # Create holdout set with multiple speed rates
    samples_dir = args.samples_dir
    sped_up_rates = [args.speed]
    successful_canaries = 0
    random.seed(args.seed)
    
    for sentence in canaries:
        successful_canaries += holdout_set(sentence, samples_dir, sped_up_rates)
    
    print(f"Generated {successful_canaries} holdout samples with speed rates {sped_up_rates}. Saved in {samples_dir}.")