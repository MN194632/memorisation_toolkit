import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


    
class LibriSpeechDataset(Dataset):
    def __init__(self, csv_path, base_audio_dir, processor, tokenizer, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_audio_dir = base_audio_dir
        self.transform = transform
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.df.iloc[idx]
        
        # Load audio file using the path from 'wav' column
        wav_path = os.path.join(self.base_audio_dir, row['wav'])
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Get transcript 
        transcript = row['wrd']
        
        # Process audio with the processor
        input_values = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_values
        
        # Tokenize transcript for labels
        labels = self.tokenizer(transcript).input_ids
        
        # Return format expected by DataCollatorCTCWithPadding
        sample = {
            'input_values': input_values,
            'labels': labels,
            # Optional metadata if needed
            'speaker_id': row['spk_id'],
            'duration': row['duration'],
            'transcript': transcript  # Keep original for reference
        }
        
        return sample
