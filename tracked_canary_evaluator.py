import os
import json
import torch
import pandas as pd
import numpy as np
import math
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics.text import CharErrorRate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2CTCTokenizer,  Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from dataset_utils import DataCollatorCTCWithPadding
from dataloader_canaries import CanariesDataset
from exposure import calculate_rank_and_exposure
from HOLDOUT import MultiSpeedHoldOutCERCalculator
import torch.nn.functional as F 

class TrackedCanaryEvaluator:
    """
    Evaluator that evaluates canaries tracked during training.
    Uses the tracking information from the canary coordinator saved in the output_dir given in config.
    """
    
    def __init__(self,config,checkpoint_dir):
        """
        Initialise the tracked canary evaluator.
        
        Args:
            config: contains all nessecary path arguments 
        """
        self.model_checkpoint_dir = checkpoint_dir
        self.canaries_dir = config.canaries_dir
        self.canaries_tracking_dir = config.canaries_tracking_dir
        self.target_sample_rate = config.target_sample_rate
        
        # Load the model and processor
        self.model, self.processor, self.tokenizer = self._load_model()
        
        # Load canary tracking information
        self.tracked_canaries = self._load_tracking_info()
        
        # Set up metrics
        self.cer_metric = CharErrorRate()
    
    def _load_model(self):
        print(f"Loading fine-tuned model from {self.model_checkpoint_dir}")
        
        if not os.path.exists(self.model_checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.model_checkpoint_dir}")

        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_checkpoint_dir,
            use_safetensors=True,
            device_map="auto"
        )
        
        processor = Wav2Vec2Processor.from_pretrained(self.model_checkpoint_dir)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_checkpoint_dir)
        

        device = next(model.parameters()).device
        print(f"Model loaded on device: {device}")
        
        return model, processor, tokenizer
    
    def _load_tracking_info(self):
        csv_file = os.path.join(self.canaries_tracking_dir, "canary_assignments.csv")
        print(f"Loading tracking info from CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} tracked canaries with frequency distribution:")
        freq_counts = df['frequency'].value_counts().sort_index()
        for freq, count in freq_counts.items():
            print(f"Frequency {freq}: {count} canaries")
        
        return df
 
    def create_tracked_canary_dataset(self):
        """
        Create a dataset containing only the tracked canaries.
        
        Returns:
            CanariesDataset: Dataset containing only tracked canaries
        """
        # First, load the full canaries dataset
        full_dataset = CanariesDataset(
            self.canaries_dir,
            self.processor,
            self.target_sample_rate
        )
   
        # Only include tracked canaries
        audio_path_to_idx = {}
        for i, sample in enumerate(full_dataset.samples):
            audio_path_to_idx[sample['audio_path']] = i
        
        # Find the indices of tracked canaries in the full dataset
        tracked_indices = [] 
        for _, row in self.tracked_canaries.iterrows():
            audio_path = row['audio_path']
            if audio_path in audio_path_to_idx:
                tracked_indices.append(audio_path_to_idx[audio_path])
   
        tracked_dataset = Subset(full_dataset, tracked_indices)

        self.subset_to_tracking = {}
        for i, idx in enumerate(tracked_indices):
            audio_path = full_dataset.samples[idx]['audio_path']
            
            # Find this canary in the tracking info
            tracking_info = self.tracked_canaries[
                self.tracked_canaries['audio_path'] == audio_path
            ]
            
            if len(tracking_info) == 0:
                # Try looking up by folder name (timestamp)
                folder_name = os.path.basename(os.path.dirname(audio_path))
                canary_id = f"canary_{folder_name}"
                tracking_info = self.tracked_canaries[
                    self.tracked_canaries['canary_id'] == canary_id
                ]
            
            if len(tracking_info) > 0:
                tracking_row = tracking_info.iloc[0]
                self.subset_to_tracking[i] = {
                    'canary_id': tracking_row['canary_id'],
                    'frequency': tracking_row['frequency'],
                    'audio_path': audio_path
                }
        
        return tracked_dataset
    
    def evaluate(self, config, batch_size=1):
        """
        Evaluate the model on tracked canaries.
        
        Args:
            config: Config object containing paths and settings
            batch_size: Batch size for evaluation (default=1 for per-sample losses)
            
        Returns:
            DataFrame with evaluation results
        """
        # Create dataset with tracked canaries
        tracked_dataset = self.create_tracked_canary_dataset()
        
        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        
        dataloader = DataLoader(
            tracked_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=data_collator
        )
        
        # Setup for evaluation
        device = next(self.model.parameters()).device
        self.model.eval()
        results = []
        
        # Evaluation loop 
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating tracked canaries")):

                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with labels to get loss
                outputs = self.model(input_values, labels=labels)
                logits = outputs.logits
        
                predicted_ids = torch.argmax(logits, dim=-1)
                pred = self.processor.batch_decode(predicted_ids)[0]
                
                sample = tracked_dataset[batch_idx]
                ref = sample['transcript'].strip(".,-_")
                speaker_id = sample['speaker_id']

                # Calculate CER
                cer = float(self.cer_metric(pred.upper(), ref.upper()))
                
                # Get tracking info
                tracking_info = self.subset_to_tracking.get(batch_idx, {})
                canary_id = tracking_info.get('canary_id', f"canary_{batch_idx}")
                frequency = tracking_info.get('frequency', 0)
                audio_path = tracking_info.get('audio_path', '')
                
                results.append({
                    'canary_id': canary_id,
                    'frequency': frequency,
                    'speaker_id':speaker_id,
                    'reference': ref,
                    'prediction': pred,
                    'CER': cer,
                    'audio_path': audio_path,
                })
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)

        # Save results 
        checkpoint_name = os.path.basename(self.model_checkpoint_dir.rstrip('/'))
        evaluation_dir = os.path.join(config.output_dir, "evaluation", checkpoint_name)
        os.makedirs(evaluation_dir, exist_ok=True)
        csv_path = os.path.join(evaluation_dir, "tracked_canary_evaluation.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to {csv_path}")
        
        # Save a summary by frequency
        summary = results_df.groupby('frequency').agg({
            'CER': ['mean', 'min', 'max', 'count'],
 
        })
        summary.columns = ['mean_CER', 'min_CER', 'max_CER', 'count']
        summary_path = os.path.join(evaluation_dir, "frequency_evaluation_summary.csv")
        summary.to_csv(summary_path)
        print(f"Frequency summary saved to {summary_path}")
        
        
        return results_df
    
    def analyse_exposure(self, config, results_df, holdout_folder="/dtu/blackhole/04/147012/hold_out_set", 
                        speed=None, max_holdout_samples=None):
        """
        Calculate exposure metrics for the evaluated canaries. 
        
        Args:
            config: Config object
            results_df: DataFrame with evaluation results
            holdout_folder: Path to holdout samples directory
            speed: Speed factor for holdout samples (auto-detected if None)
            max_holdout_samples: Maximum holdout samples to process
            
        Returns:
            DataFrame with exposure analysis results
        """

        calculator = MultiSpeedHoldOutCERCalculator(config, self.model_checkpoint_dir)

        holdout_results = calculator.process_holdout_folder(
            holdout_folder, 
            speed, 
            max_samples=max_holdout_samples
        )
        
        holdout_cers = [r['CER'] for r in holdout_results]
        
        # Calculate exposure for each canary
        exposure_results = []
        for _, row in results_df.iterrows():
            canary_id = row['canary_id']
            canary_cer = row['CER']
            frequency = row['frequency']

            
            # Calculate rank and exposure using in-memory holdout CERs
            
            #sorted_hold_out_cers = holdout_cers.sort()
            #rank = np.searchsorted(sorted_hold_out_cers,canary_cer, side = 'right') + 1
            rank = 1 + sum(1 for h_cer in holdout_cers if h_cer <= canary_cer)
            exposure = np.log2(len(holdout_cers)) - np.log2(rank)
            if exposure < 0:
                exposure = 0

            exposure_results.append({
                'canary_id': canary_id,
                'frequency': frequency,
                'CER': canary_cer,
                'rank': rank,
                'exposure': exposure,

            })
        
        exposure_df = pd.DataFrame(exposure_results)
        
        # Save results
        checkpoint_name = os.path.basename(self.model_checkpoint_dir.rstrip('/'))
        evaluation_dir = os.path.join(config.output_dir, "evaluation", checkpoint_name)
        os.makedirs(evaluation_dir, exist_ok=True)
        
        exposure_csv = os.path.join(evaluation_dir, "tracked_canary_exposure.csv")
        exposure_df.to_csv(exposure_csv, index=False)
        print(f"Exposure analysis saved to {exposure_csv}")
        
        # Save summary by frequency
        summary = exposure_df.groupby('frequency').agg({
            'exposure': ['mean', 'min', 'max'],
            'rank': ['mean'],
            'CER': ['mean']

        })
        summary.columns = ['mean_exposure', 'min_exposure', 'max_exposure', 'mean_rank', 'mean_CER']
        summary_path = os.path.join(evaluation_dir, "frequency_exposure_summary.csv")
        summary.to_csv(summary_path)
        print(f"Frequency exposure summary saved to {summary_path}")
        
        return exposure_df