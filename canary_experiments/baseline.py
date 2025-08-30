import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import argparse
from datetime import datetime
from dataloader_canaries import CanariesDataset
from dataloader_libri import LibriSpeechDataset
from dataset_utils import DataCollatorCTCWithPadding
from utils import get_config_class
from HOLDOUT import MultiSpeedHoldOutCERCalculator

class BaselineEvaluatorWithExposure:
    """
    Evaluate all canaries (all speeds) and LibriSpeech test on the base Wav2Vec2 model before finetuning for baseline,
    including exposure calculation
    """
    
    def __init__(self, base_canaries_dir, model_name="facebook/wav2vec2-base-960h", target_sample_rate=16000):
        """
        Initialise with base model
        
        Args:
            base_canaries_dir: Base directory containing canaries folders
            model_name: Name of the base pre-trained Wav2Vec2 model
            target_sample_rate: Target sample rate for audio processing
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.base_canaries_dir = base_canaries_dir
        self.target_sample_rate = target_sample_rate
        
        # Define all speed folders
        self.speed_folders = {
            "1x": "1x_samples",
            "1.5x": "15x_samples", 
            "2x": "2x_samples",
            "2.5x": "25x_samples",
            "3x": "3x_samples", 
            "3.5x": "35x_samples",
            "4x": "4x_samples"
        }
        
        # Load base model and processor
        print(f"Loading base model: {model_name}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        
        self.model.eval()
        
        # Setup metrics
        self.cer_metric = CharErrorRate()
        
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
    
    def evaluate_holdout_samples(self, holdout_folder, speed, max_samples=20000):
        """
        Evaluate holdout samples using the base model for exposure calculation
        
        Args:
            holdout_folder: Path to holdout audio samples
            speed: Speed factor for holdout samples
            max_samples: Maximum number of holdout samples to process
            
        Returns:
            List of CER values from holdout samples
        """
        print(f"Evaluating holdout samples for speed {speed}.")
        
        # Dummy config to initalise MultiSpeedHoldOutCERCalculator with default
        config = type('Config', (), {
            'target_sample_rate': self.target_sample_rate,
            'model': self.model,
            'processor': self.processor,
            'tokenizer': self.tokenizer
        })()

        # MultiSpeedHoldOutCERCalculator will default to pre-trained checkpoint facebook-base if it doesn't recognise checkpoint, therefore this non-existent_checkpoint
        calculator = MultiSpeedHoldOutCERCalculator(config, "non-existent_checkpoint: baseline model used") 

        # Process holdout samples 
        holdout_results = calculator.process_holdout_folder(
            holdout_folder, 
            speed, 
            max_samples=max_samples
        )
        
        # Extract CER values
        holdout_cers = [r['CER'] for r in holdout_results]
        print(f"Evaluated {len(holdout_cers)} holdout samples")
        print(f"Holdout CER stats: mean={np.mean(holdout_cers):.3f}, std={np.std(holdout_cers):.3f}")
        
        return holdout_cers
    
    def calculate_exposure(self, canary_cer, holdout_cers):
        """
        Calculate exposure for a canary given holdout CERs
        
        Args:
            canary_cer: CER of the canary
            holdout_cers: List of CER values from holdout set
            
        Returns:
            tuple: (rank, exposure)
        """
        if not holdout_cers:
            print('NO HOLD-OUT SET FOUND. ')
            return None, None
            
        # Calculate rank (1-indexed)
        rank = 1 + sum(1 for h_cer in holdout_cers if h_cer <= canary_cer)
        
        # Calculate exposure
        exposure = np.log2(len(holdout_cers)+1) - np.log2(rank)
        
        return rank, exposure
    
    def evaluate_canaries_for_speed(self, speed, batch_size=1, holdout_cers=None):
        """
        Evaluate canaries for a specific speed, including exposure calculation. To be for each speed.
        
        Args:
            speed: Speed factor (e.g., "1x", "1.5x", etc.)
            batch_size: Batch size for evaluation
            holdout_cers: Holdout CERs (given using function evaluate_holdout_samples) for exposure calculation
            
        Returns:
            List of result dictionaries including exposure
        """
        folder_name = self.speed_folders[speed]
        canaries_dir = os.path.join(self.base_canaries_dir, folder_name)
        
        if not os.path.exists(canaries_dir):
            print(f"Warning: Canaries directory not found: {canaries_dir}")
            return []
        
        # Create dataset
        dataset = CanariesDataset(
            canaries_dir,
            self.processor,
            self.target_sample_rate
        )
        
        if len(dataset) == 0:
            print(f"No canaries found in {canaries_dir}")
            return []    
        print(f"Found {len(dataset)} canaries for {speed} in {canaries_dir}")
        

        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor, 
            padding=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=data_collator
        )
        
        # Evaluation
        results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {speed} canaries")):

                input_values = batch["input_values"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_values)
                logits = outputs.logits
                
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions = self.processor.batch_decode(predicted_ids)
                
                # Get ground truth
                references = batch["transcript"]
                
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    cer = float(self.cer_metric(pred, ref.upper()))                    
                    rank, exposure = self.calculate_exposure(cer, holdout_cers)
                    
                    # Get additional info about this canary
                    sample_idx = batch_idx * batch_size + i
                    if sample_idx < len(dataset):
                        sample = dataset[sample_idx]
                        audio_path = sample.get('audio_path', '')
                        speaker_id = sample.get('speaker_id', '')
                        
                        # Extract canary ID from folder name
                        folder_name = os.path.basename(os.path.dirname(audio_path)) if audio_path else f"canary_{sample_idx}"
                        canary_id = f"canary_{folder_name}"
                        
                        results.append({
                            'dataset_type': 'canary',
                            'speed': speed,
                            'canary_id': canary_id,
                            'reference': ref,
                            'prediction': pred,
                            'CER': cer,
                            'rank': rank,
                            'exposure': exposure,
                            'speaker_id': speaker_id,
                            'audio_path': audio_path
                        })
        
        return results
    
    def evaluate_librispeech_test(self, test_csv, base_audio_dir, batch_size=1):
        """
        Evaluate LibriSpeech test set (no exposure calculation needed)
        
        Args:
            test_csv: Path to test CSV file
            base_audio_dir: Base audio directory
            batch_size: Batch size for evaluation
            
        Returns:
            List of result dictionaries
        """
        print("Evaluating LibriSpeech test set...")
        
        # Create dataset
        dataset = LibriSpeechDataset(
            test_csv,
            base_audio_dir,
            self.processor,
            self.tokenizer
        )
        
        print(f"LibriSpeech test dataset contains {len(dataset)} samples")
        
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=data_collator
        )
        
        # Evaluation
        results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing LibriSpeech test")):
                # Move to device
                input_values = batch["input_values"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_values)
                logits = outputs.logits
                
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions = self.processor.batch_decode(predicted_ids)
                
                # Get reference transcripts
                references = batch["transcript"]
                speaker_ids = batch.get("speaker_id", ["unknown"] * len(references))
                
                # Process each sample in the batch
                for i, (pred, ref, spk_id) in enumerate(zip(predictions, references, speaker_ids)):
                    # Calculate CER
                    cer = float(self.cer_metric(pred, ref.upper()))
                    
                    results.append({
                        'dataset_type': 'librispeech',
                        'speed': 'normal',
                        'canary_id': f'libri_{batch_idx * batch_size + i}',
                        'reference': ref,
                        'prediction': pred,
                        'CER': cer,
                        'rank': None,
                        'exposure': None,
                        'speaker_id': str(spk_id),
                        'audio_path': ''
                    })
        
        return results
    
    def evaluate_all(self, test_csv=None, base_audio_dir=None, batch_size=4, 
                     holdout_folder=None, max_holdout_samples=10000):
        """
        Evaluate all canary speeds and LibriSpeech test separately
        
        Args:
            test_csv: Path to LibriSpeech test CSV
            base_audio_dir: Base audio directory for LibriSpeech
            batch_size: Batch size for evaluation
            holdout_folder: Path to holdout samples for exposure calculation
            max_holdout_samples: Maximum holdout samples to use
            
        Returns:
            Tuple of (canary_results_df, librispeech_results_df)
        """
        canary_results = []
        librispeech_results = []
        
        # Evaluate all canary speeds with exposure
        for speed in self.speed_folders.keys():
            holdout_cers = []
            if holdout_folder and os.path.exists(holdout_folder):
                try:
                    numeric_speed = float(speed.replace('x', ''))
                    holdout_cers = self.evaluate_holdout_samples(
                        holdout_folder, numeric_speed, max_holdout_samples
                    )
                except Exception as e:
                    print(f"Warning: Could not evaluate holdout samples for {speed}: {e}")
            
            speed_results = self.evaluate_canaries_for_speed(
                speed, batch_size, holdout_cers
            )
            canary_results.extend(speed_results)
        
        # Evaluate LibriSpeech test if paths provided
        if test_csv and base_audio_dir:
            librispeech_results = self.evaluate_librispeech_test(
                test_csv, base_audio_dir, batch_size * 2
            )
        
        # Create DataFrames
        canary_df = pd.DataFrame(canary_results) if canary_results else pd.DataFrame()
        librispeech_df = pd.DataFrame(librispeech_results) if librispeech_results else pd.DataFrame()

        
        return canary_df, librispeech_df
    
    def save_results(self, canary_df, librispeech_df, output_filename=None):
        """
        Save canary and LibriSpeech results to separate CSV files
        
        Args:
            canary_df: DataFrame with canary results including exposure
            librispeech_df: DataFrame with LibriSpeech results
            output_filename: Custom filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_filename is None:
            prefix = f"baseline_{timestamp}"
        else:
            prefix = output_filename.replace('.csv', '')
        
        # Save canary results
        if not canary_df.empty:
            canary_path = os.path.join(os.getcwd(), f"{prefix}_canaries.csv")
            canary_df.to_csv(canary_path, index=False)
            print(f"Canary results saved to: {canary_path}")
        
        # Save LibriSpeech results
        if not librispeech_df.empty:
            libri_path = os.path.join(os.getcwd(), f"{prefix}_librispeech.csv")
            librispeech_df.to_csv(libri_path, index=False)
            print(f"LibriSpeech results saved to: {libri_path}")
        
        return canary_path if not canary_df.empty else None, libri_path if not librispeech_df.empty else None


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation with exposure calculation")
    parser.add_argument("--base_canaries_dir", type=str, 
                        help="Base directory containing speed folders")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h",
                        help="Base model to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--test_csv", type=str, 
                        help="Path to LibriSpeech test CSV")
    parser.add_argument("--base_audio_dir", type=str, 
                        default="/work3/s194632/LibriSpeech",
                        help="Base audio directory for LibriSpeech")
    parser.add_argument("--holdout_folder", type=str, 
                        help="Path to holdout samples for exposure calculation (same as tracked canaries)")
    parser.add_argument("--max_holdout_samples", type=int, default=20000,
                        help="Maximum holdout samples to use for exposure calculation")
    parser.add_argument("--output_filename", type=str, default=None,
                        help="Custom output filename")
    parser.add_argument("--no_librispeech", action="store_true",
                        help="Skip LibriSpeech evaluation")
    parser.add_argument("--no_exposure", action="store_true",
                        help="Skip exposure calculation")
    
    args = parser.parse_args()
    
    # Check if base canaries directory exists
    if not os.path.exists(args.base_canaries_dir):
        print(f"Error: Base canaries directory not found: {args.base_canaries_dir}")
        return
    
    # Initialize evaluator
    evaluator = BaselineEvaluatorWithExposure(
        base_canaries_dir=args.base_canaries_dir,
        model_name=args.model_name
    )
    
    # Set holdout folder to None if exposure calculation is disabled
    holdout_folder = None if args.no_exposure else args.holdout_folder
    
    # Run comprehensive evaluation
    if args.no_librispeech:
        test_csv = None
        base_audio_dir = None
    else:
        test_csv = args.test_csv
        base_audio_dir = args.base_audio_dir
    
    results_df, summary_df = evaluator.evaluate_all(
        test_csv=test_csv,
        base_audio_dir=base_audio_dir,
        batch_size=args.batch_size,
        holdout_folder=holdout_folder,
        max_holdout_samples=args.max_holdout_samples
    )
    
    # Save results
    if not results_df.empty:
        evaluator.save_results(results_df, summary_df, args.output_filename)
        print(f"\nBaseline evaluation with exposure complete!")
        print(f"Evaluated {len(results_df)} total samples across all speeds and LibriSpeech.")
    else:
        print("No data to save.")


if __name__ == "__main__":
    main()