import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
import argparse
import json
from datetime import datetime
from config import Config
from dataloader_libri import LibriSpeechDataset
from dataset_utils import DataCollatorCTCWithPadding
from utils import get_config_class


class LibriSpeechModelTester:
    """
    Test the fine-tuned model on the LibriSpeech test set to check if catastrophic forgetting is occuring
    """
    
    def __init__(self, config, checkpoint_dir, evaluation_dir):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.evaluation_dir = evaluation_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model, processor, and tokenizer
        self.model, self.processor, self.tokenizer = self._load_model()
        
        # Setup metrics
        self.cer_metric = CharErrorRate()
        
    def _load_model(self):
        """Load the fine-tuned model from checkpoint directory"""
        checkpoint_dir = self.checkpoint_dir
        print(f"Loading fine-tuned model from {checkpoint_dir}")
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        try:
            # Try to load the fine-tuned model
            model = Wav2Vec2ForCTC.from_pretrained(
                checkpoint_dir,
                use_safetensors=True,
                device_map="auto"
            )
            
            processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(checkpoint_dir)
            
            print(f"Successfully loaded fine-tuned model from {checkpoint_dir}")
            
        except Exception as e:
            print(f"✗ Failed to load fine-tuned model: {e}")
            print("Falling back to base model for comparison...")
            
            # Fallback to base model
            model = self.config.model.to(self.device)
            processor = self.config.processor
            tokenizer = self.config.tokenizer
        
        model.eval()
        device = next(model.parameters()).device
        print(f"Model loaded on device: {device}")
        
        return model, processor, tokenizer
    
    def create_test_dataloader(self, batch_size=8, num_workers=4):
        """Create dataloader for LibriSpeech test set"""
        print("Creating LibriSpeech test dataset...")
        
        test_dataset = LibriSpeechDataset(
            self.config.test_csv,
            self.config.base_audio_dir,
            self.processor,
            self.tokenizer
        )
        
        print(f"Test dataset contains {len(test_dataset)} samples")
        
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator
        )
        
        return test_loader
    
    def evaluate_model(self, max_samples=None):
        """
        Evaluate the model on LibriSpeech test set
        
        Args:
            test_loader: DataLoader for test set
            max_samples: Optional limit on number of samples to evaluate
            
        Returns:
            dict: Evaluation results including CER, sample predictions
        """

        test_loader = self.create_test_dataloader()
        print("Starting evaluation on LibriSpeech test set...")
        
        all_predictions = []
        all_references = []
        all_losses = []
        sample_results = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move inputs to device
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device) if "labels" in batch else None
                
                # Forward pass
                if labels is not None:
                    outputs = self.model(input_values, labels=labels)
                    loss = outputs.loss.item()
                    all_losses.append(loss)
                else:
                    outputs = self.model(input_values)
                    loss = None
                
                # Get predictions
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions = self.processor.batch_decode(predicted_ids)
                
                # Get references
                references = batch["transcript"]
                
                # Store results
                all_predictions.extend(predictions)
                all_references.extend(references)
                
                # Store some sample results for inspection
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    cer = float(self.cer_metric([pred], [ref]))
                    sample_results.append({
                        'prediction': pred,
                        'reference': ref,
                        'cer': cer
                    })
                
                total_samples += len(predictions)
                
                # Check if we've reached the maximum number of samples
                if max_samples and total_samples >= max_samples:
                    print(f"Reached maximum samples limit: {max_samples}")
                    break
        
        # Calculate overall metrics
        overall_cer = float(self.cer_metric(all_predictions, all_references))
        avg_loss = np.mean(all_losses) if all_losses else None
        
        results = {
            'total_samples': total_samples,
            'overall_cer': overall_cer,
            'average_loss': avg_loss,
            'sample_results': sample_results,
            'all_predictions': all_predictions,  
            'all_references': all_references
        }
        
        self.save_results(results)
        return results
    
    def print_results(self, results):
        """Print evaluation results in a readable format"""
        print("\n" + "="*60)
        print("LIBRISPEECH TEST SET EVALUATION RESULTS")
        print("="*60)
        
        print(f"Total samples evaluated: {results['total_samples']}")
        print(f"Overall Character Error Rate (CER): {results['overall_cer']:.4f}")
        
        if results['average_loss'] is not None:
            print(f"Average Loss: {results['average_loss']:.4f}")
        
        print("\n" + "="*60)
        
        # Performance assessment
        if results['overall_cer'] < 0.1:
            print("EXCELLENT: Model performs very well on normal speech")
        elif results['overall_cer'] < 0.2:
            print("GOOD: Model performs reasonably well on normal speech")
        elif results['overall_cer'] < 0.5:
            print("⚠ MODERATE: Model has some degradation but still functional")
        else:
            print("✗ POOR: Model has significant degradation on normal speech")
    
    def save_results(self, results, output_file=None):
        """Save evaluation results to files"""
        if output_file is None:
            output_file = f"librispeech_evaluation"
        
        # Use the checkpoint-specific evaluation directory
        eval_dir = self.evaluation_dir
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save sample results as CSV for easy inspection
        if results['sample_results']:
            csv_file = os.path.join(eval_dir, f"{output_file}_samples.csv")
            df = pd.DataFrame(results['sample_results'])
            df.to_csv(csv_file, index=False)
            print(f"Sample results saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model on LibriSpeech test set")
    parser.add_argument('--config', type=str, required=True, help='Config class name')
    parser.add_argument("--checkpoint_dir", required=True, help="Batch size for evaluation")
    parser.add_argument('--evaluation_dir', type=str, default=None,
                        help='Path to directory containing evaluation CSV files (overrides default)')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_name", type=str, default=None, help="Custom output file name")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config_class(args.config)
    
    print("Configuration:")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Test CSV: {config.test_csv}")
    if args.evaluation_dir:
        evaluation_dir = args.evaluation_dir
    else:
        # Create checkpoint-specific evaluation directory path
        checkpoint_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
        evaluation_dir = os.path.join(config.output_dir, "evaluation", checkpoint_name)
    
    if not os.path.exists(evaluation_dir):
        print(f"Error: Evaluation directory {evaluation_dir} does not exist")
        print("Make sure to run evaluation scripts first to generate the required CSV files")
        return
    
    print(f"Using evaluation directory: {evaluation_dir}")
    
    # Initialize tester with the checkpoint-specific evaluation directory
    tester = LibriSpeechModelTester(config, args.checkpoint_dir, evaluation_dir)
    

    # Run evaluation
    results = tester.evaluate_model( max_samples=args.max_samples)
    
    # Print and save results
    tester.print_results(results)
    tester.save_results(results, args.output_name)
    
    print(f"\nEvaluation complete! Check {evaluation_dir}/ for detailed results.")


if __name__ == "__main__":
    main()