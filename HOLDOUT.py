import os
import csv
import torch
import torchaudio
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from torchmetrics.text import CharErrorRate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torchaudio.transforms as T
from utils import get_config_class
import random
import math
import torch.nn.functional as F 


class MultiSpeedHoldOutCERCalculator:
    """
    Calculate CER for multi-speed hold-out samples using a fine-tuned Wav2Vec2ForCTC model
    """
    
    def __init__(self, config, checkpoint_dir):
        """
        Initialise with config containing model checkpoint and settings
        
        Args:
            config: Config object with checkpoint_dir, processor, tokenizer, etc.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        # Load the fine-tuned model from config
        print(f"Loading fine-tuned model from {self.checkpoint_dir}")
        
        if os.path.exists(checkpoint_dir):
            # Load fine-tuned model
            self.model = Wav2Vec2ForCTC.from_pretrained(
                checkpoint_dir,
                use_safetensors=True,
                device_map="auto"
            )
            self.processor = Wav2Vec2Processor.from_pretrained(self.checkpoint_dir)
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.checkpoint_dir)
            print(f"Using fine-tuned model from {self.checkpoint_dir}")
        else:
            # Fall back to base model from config
            print(f"Checkpoint directory not found: {self.checkpoint_dir}")
            print("Using base model from config...")
            self.model = config.model.to(self.device)
            self.processor = config.processor
            self.tokenizer = config.tokenizer
        
        self.model.eval()
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        
        # Initialise CER metric
        self.cer_metric = CharErrorRate()
    
    def load_audio(self, audio_path):
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed waveform tensor
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.config.target_sample_rate:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.config.target_sample_rate)
                waveform = resampler(waveform)
            
            return waveform.squeeze()
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def transcribe_audio(self, waveform,ground_truth):
        """
        Transcribe audio using the fine-tuned model
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            Transcribed text
        """
        try:
            # Process audio
            processed = self.processor(
                waveform.numpy(), 
                sampling_rate=self.config.target_sample_rate, 
                return_tensors="pt"
            )

            labels = self.processor.tokenizer(ground_truth.upper()).input_ids
            labels = torch.tensor(labels).unsqueeze(0).to(self.device) 
            
            # Move to device
            input_values = processed.input_values.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model(input_values, labels=labels)
                logits = outputs.logits



                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def load_ground_truth(self, sentence_path):
        """
        Load ground truth sentence from text file
        
        Args:
            sentence_path: Path to sentence.txt file
            
        Returns:
            Ground truth text
        """
        try:
            with open(sentence_path, 'r', encoding='utf-8') as f:
                sentence = f.read().strip()
            return sentence
        except Exception as e:
            print(f"Error loading sentence from {sentence_path}: {e}")
            return ""
    
    def load_speaker_id(self, speaker_id_path):
        """
        Load speaker ID from text file
        
        Args:
            speaker_id_path: Path to speaker_id.txt file
            
        Returns:
            Speaker ID string
        """
        try:
            with open(speaker_id_path, 'r', encoding='utf-8') as f:
                speaker_id = f.read().strip()
            return speaker_id
        except Exception as e:
            print(f"Error loading speaker ID from {speaker_id_path}: {e}")
            return "unknown"
    
    def calculate_cer(self, reference, prediction):
        """
        Calculate Character Error Rate
        
        Args:
            reference: Ground truth text
            prediction: Predicted text
            
        Returns:
            CER value
        """
        if not reference or not prediction:
            return 1.0  # Maximum CER if either is empty
        
        # Convert to uppercase for consistency (following existing pattern)
        reference = reference.upper()
        reference = reference.strip(',.-_')
        prediction = prediction.upper()
        
        return float(self.cer_metric(prediction, reference))
    
    def get_audio_filename_for_speed(self, speed):
        """
        Get the appropriate audio filename for the given speed
        
        Args:
            speed: Speed factor (1.5, 2, 2.5, 3, 3.5, 4, or 'normal')
            
        Returns:
            Audio filename
        """
        if speed == 'normal' or speed == 1.0:
            return "speech.wav"
        else:
            if isinstance(speed, int):
                speed_str = str(speed)
            elif isinstance(speed, float):
                if speed.is_integer():
                    speed_str = str(int(speed)) 
                else:
                    speed_str = str(speed)  
            else:
                speed_str = str(speed)
            
            return f"canary_{speed_str}.wav"
    
    def process_holdout_folder(self, holdout_folder_path, speed, max_samples=20_000):
        """
        Process all hold-out samples in a folder for a specific speed
        
        Args:
            holdout_folder_path: Path to folder containing hold-out sample subfolders
            speed: Speed factor to process (1.5, 2, 2.5, 3, 3.5, 4, or 'normal')
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            List of results dictionaries
        """
        holdout_path = Path(holdout_folder_path)
        
        if not holdout_path.exists():
            raise ValueError(f"Hold-out folder not found: {holdout_folder_path}")
        
        results = []
        
        # Get all subdirectories
        subdirs = [d for d in holdout_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"No subdirectories found in {holdout_folder_path}")
            return results
        
        subdirs = random.sample(subdirs, max_samples)
        audio_filename = self.get_audio_filename_for_speed(speed)
        
        print(f"Processing {len(subdirs)} hold-out samples for speed {speed}x ({audio_filename})...")
        
        for subdir in tqdm(subdirs, desc=f"Processing speed {speed}x"):
            try:
                # File paths
                audio_path = subdir / audio_filename
                sentence_path = subdir / "sentence.txt"
                speaker_id_path = subdir / "speaker_id.txt"
                
                # Check if required files exist
                if not audio_path.exists():
                    print(f"Audio file not found: {audio_path}")
                    continue
                
                if not sentence_path.exists():
                    print(f"Sentence file not found: {sentence_path}")
                    continue
                
                # Load ground truth
                ground_truth = self.load_ground_truth(sentence_path)
                if not ground_truth:
                    print(f"Empty or invalid sentence file: {sentence_path}")
                    continue
                
                # Load speaker ID
                speaker_id = self.load_speaker_id(speaker_id_path)
                
                # Load and transcribe audio
                waveform = self.load_audio(audio_path)
                if waveform is None:
                    print(f"Failed to load audio: {audio_path}")
                    continue
                
                transcription = self.transcribe_audio(waveform,ground_truth)
                
                # Calculate CER
                cer = self.calculate_cer(ground_truth, transcription)
                
                # Store results
                results.append({
                    'sentence': ground_truth,
                    'transcription': transcription,
                    'CER': cer,
                    'speaker_id': speaker_id,
                    'sample_folder': subdir.name,
                    'speed': speed,
                    'audio_file': audio_filename
                })
                
            except Exception as e:
                print(f"Error processing {subdir}: {e}")
                continue
        
        return results
    
    def save_results_to_csv(self, results, speed, output_dir=None):
        """
        Save results to CSV file
        
        Args:
            results: List of result dictionaries
            speed: Speed factor used
            output_dir: Directory to save CSV (if None, uses config.output_dir)
        """
        if not results:
            print("No results to save")
            return None
        
        # Use config output directory if no path specified
        if output_dir is None:
            checkpoint_name = os.path.basename(self.checkpoint_dir.rstrip('/'))
            output_dir = os.path.join(self.config.output_dir, "evaluation", checkpoint_name)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on speed
        if speed == 'normal' or speed == 1.0:
            filename = "holdout_cer_results_normal.csv"
        else:
            # Format speed for filename (replace . with _ for filesystem compatibility)
            # Examples: 1.5 -> "1_5", 2.5 -> "2_5", 2 -> "2"
            if isinstance(speed, float):
                if speed.is_integer():
                    speed_str = str(int(speed))  # 2.0 -> "2"
                else:
                    speed_str = str(speed).replace('.', '_')  # 1.5 -> "1_5", 2.5 -> "2_5"
            else:
                speed_str = str(speed)
            filename = f"holdout_cer_results.csv"
        
        csv_path = output_path / filename
        
        # Save to CSV with required columns
        df = pd.DataFrame(results)
        
        # Keep only the required columns in the correct order
        df_output = df[['sentence', 'transcription', 'CER', 'speaker_id']].copy()
        
        df_output.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Print summary statistics
        print(f"\nSummary for speed {speed}x:")
        print(f"Total samples processed: {len(results)}")
        print(f"Average CER: {df_output['CER'].mean():.4f}")
        print(f"Min CER: {df_output['CER'].min():.4f}")
        print(f"Max CER: {df_output['CER'].max():.4f}")
        print(f"Std CER: {df_output['CER'].std():.4f}")
        print(f"Unique speakers: {df_output['speaker_id'].nunique()}")
        
        return csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate holdout CER CSV for specific speed using fine-tuned Wav2Vec2ForCTC")
    parser.add_argument("--config", type=str, required=True, help="Config class name")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory") 
    parser.add_argument("--holdout_folder", type=str, 
                        help="Path to folder containing hold-out sample subfolders")
    parser.add_argument("--speed", type=str, required=True,
                        help="Speed factor to process (1.5, 2, 2.5, 3, 3.5, 4, or 'normal')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for CSV file (if not specified, uses config.output_dir/evaluation/)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (useful for testing)")

    args = parser.parse_args()

    # Parse speed argument
    try:
        if args.speed.lower() == 'normal':
            speed = 'normal'
        else:
            speed = float(args.speed)
            # Validate speed
            valid_speeds = [1.5, 2, 2.5, 3, 3.5, 4]
            if speed not in valid_speeds:
                print(f"Warning: Speed {speed} not in standard speeds {valid_speeds}")
    except ValueError:
        print(f"Error: Invalid speed '{args.speed}'. Use 'normal' or a number like 1.5, 2, 2.5, 3, 3.5, 4")
        return
    
    # Load config
    config = get_config_class(args.config)
    

    
    print("Configuration:")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Holdout folder: {args.holdout_folder}")
    print(f"Speed: {speed}")
    print(f"Output directory: {args.output_dir or config.output_dir + '/evaluation'}")
    print()
    
    # Initialise calculator with config
    calculator = MultiSpeedHoldOutCERCalculator(config,args.checkpoint_dir)
    
    # Process hold-out samples for the specified speed
    results = calculator.process_holdout_folder(
        args.holdout_folder, 
        speed, 
        max_samples=args.max_samples
    )
    
    # Save results
    output_csv = calculator.save_results_to_csv(results, speed, args.output_dir)
    
    if output_csv:
        print(f"\nHoldout CSV generated successfully: {output_csv}")
    else:
        print("\nFailed to generate holdout CSV.")

def debug_holdout_shapes(config, checkpoint_dir, holdout_folder, speed):
    """Debug function to check shapes for a single holdout sample"""
    calculator = MultiSpeedHoldOutCERCalculator(config, checkpoint_dir)
    
    # Get first holdout sample
    holdout_path = Path(holdout_folder)
    subdirs = [d for d in holdout_path.iterdir() if d.is_dir()]
    if not subdirs:
        print("No samples found")
        return
    
    sample_dir = subdirs[0]
    audio_filename = calculator.get_audio_filename_for_speed(speed)
    audio_path = sample_dir / audio_filename
    sentence_path = sample_dir / "sentence.txt"
    
    # Load sample
    waveform = calculator.load_audio(audio_path)
    ground_truth = calculator.load_ground_truth(sentence_path)
    
    print(f"Sample: {sample_dir.name}")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Ground truth: {ground_truth}")
    
    # Modified transcribe with debug prints
    processed = calculator.processor(
        waveform.numpy(), 
        sampling_rate=calculator.config.target_sample_rate, 
        return_tensors="pt"
    )
    
    print(f"Input shape: {processed.input_values.shape}")
    
    with torch.no_grad():
        outputs = calculator.model(processed.input_values.to(calculator.device))
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Logits timesteps: {outputs.logits.shape[1]}")
        print(outputs.logits)
if __name__ == "__main__":
    main()
