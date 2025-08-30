import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T

class CanariesDataset(Dataset):
    def __init__(self, samples_dir, processor, target_sample_rate, transform=None):
        self.samples_dir = samples_dir
        self.transform = transform
        self.processor = processor
        self.target_sample_rate = target_sample_rate

        self.samples = []
       
        # Walk through the samples directory
        for timestamp_folder in os.listdir(samples_dir):
            folder_path = os.path.join(samples_dir, timestamp_folder)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue
                
            # Paths to the required files
            audio_path = os.path.join(folder_path, "canary.wav")
            sentence_path = os.path.join(folder_path, "sentence.txt")
            speaker_id_path = os.path.join(folder_path, "speaker_id.txt")
            
            # Check if all required files exist
            if all(os.path.exists(p) for p in [audio_path, sentence_path, speaker_id_path]):
                # Read the sentence
                with open(sentence_path, 'r') as f:
                    transcript = f.read().strip()
                
                # Read the speaker ID
                with open(speaker_id_path, 'r') as f:
                    speaker_id = f.read().strip()
                
                self.samples.append({
                    'audio_path': audio_path,
                    'transcript': transcript,
                    'speaker_id': speaker_id
                })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.samples[idx]
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(sample['audio_path'])

        # Resample to 16000 in order for processing to work
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        # Calculate duration of audio from shape and sample_rate
        duration = waveform.shape[1] / sample_rate
        
        processed = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )

        transcript = sample['transcript'].rstrip('.,!?;:')

        # CTC models work with uppercase text!
        transcript_upper = transcript.upper()
        
        labels = self.processor.tokenizer(transcript_upper).input_ids
        
        return {
            'input_values': processed.input_values,  # Shape is [1, sequence_length]
            'labels': labels,
            'speaker_id': sample['speaker_id'],
            'duration': duration,
            'transcript': sample['transcript'],
            'audio_path': sample['audio_path']
        }
