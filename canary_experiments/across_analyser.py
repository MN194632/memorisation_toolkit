import pandas as pd
import numpy as np
import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm

class AcrossSpeedAnalyser:
    """
    Analyser that combines CER and exposure results across all speeds and checkpoints
    """
    
    def __init__(self, base_dir):
        """
        Initialise the analyser
        
        Args:
            base_dir: Base directory for given experiment containing all speed result folders
        """
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "across_speed_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_speed_from_folder(self, folder_name):
        """Extract speed factor from folder name like '15x_results' -> 1.5"""
        if "15x" in folder_name:
            return 1.5
        elif "25x" in folder_name:
            return 2.5
        elif "35x" in folder_name:
            return 3.5
        elif "2x" in folder_name:
            return 2.0
        elif "3x" in folder_name:
            return 3.0
        elif "4x" in folder_name:
            return 4.0
        elif "1" or "1x" in folder_name:
            return 1.0
        elif "A" in folder_name:
            return 2.5
        elif "B" in folder_name:
            return 1
        elif "AB" in folder_name:
            return 2.5
        elif "C" in folder_name:
            return 1.0
        elif "AC" in folder_name:
            return 2.5
        elif "BC" in folder_name:
            return 1.0
        elif "ABC" in folder_name:
            return 2.5
        else:
            return None
    
    def extract_checkpoint_step(self, checkpoint_dir_name):
        """Extract training step from checkpoint directory name"""
        match = re.search(r'checkpoint-(\d+)', checkpoint_dir_name)
        if match:
            return int(match.group(1))
        return None
    
    def find_speed_result_folders(self):
        """Find all speed result folders in the base directory"""
        speed_folders = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.endswith('_results'):
                speed = self.extract_speed_from_folder(item.name)
                if speed is not None:
                    speed_folders.append((speed, item))
        
        speed_folders.sort(key=lambda x: x[0])  # Sort by speed
        return speed_folders
    
    def find_checkpoint_evaluation_folders(self, speed_results_dir):
        """Find all checkpoint evaluation folders for a given speed results folder"""
        evaluation_dir = speed_results_dir / "evaluation"
        
        if not evaluation_dir.exists():
            print(f"Found no evaluation directory in {speed_results_dir}")
            return []
        
        checkpoint_dirs = []
        for item in evaluation_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                step = self.extract_checkpoint_step(item.name)
                if step is not None:
                    checkpoint_dirs.append((step, item))
        
        checkpoint_dirs.sort(key=lambda x: x[0])  # Sort by step
        return checkpoint_dirs
    
    def load_evaluation_data(self, checkpoint_eval_dir):
        """Load CER and exposure data from a checkpoint evaluation directory"""
        cer_file = checkpoint_eval_dir / "tracked_canary_evaluation.csv"
        exposure_file = checkpoint_eval_dir / "tracked_canary_exposure.csv"
        
        data = {}
        # CER
        cer_df = pd.read_csv(cer_file)
        
        # Extract speaker_id from audio_path if not present
        if 'speaker_id' not in cer_df.columns and 'audio_path' in cer_df.columns: 
            cer_df['speaker_id'] = cer_df['audio_path'].apply(self._extract_speaker_id_from_path)  
        data['cer_data'] = cer_df

        # Exposure
        exposure_df = pd.read_csv(exposure_file)
        data['exposure_data'] = exposure_df

        return data
    
    def _extract_speaker_id_from_path(self, audio_path):
        """Extract speaker_id from the canary folder since it was not saved in the tracking info when building pipeline """
        try:
            if pd.isna(audio_path) or not audio_path:
                return "unknown"
            
            # Get the folder containing the audio file
            folder_path = Path(audio_path).parent
            speaker_id_file = folder_path / "speaker_id.txt" 
            with open(speaker_id_file, 'r') as f:
                return f.read().strip()

        except Exception as e:
            print(f"Error extracting speaker_id from {audio_path}: {e}")
            return "unknown"
    
    def combine_all_results(self):
        """
        Combine all CER and exposure results across speeds and checkpoints to use in MASTER file
        
        Returns:
            DataFrame with all combined results
        """
        speed_folders = self.find_speed_result_folders()
        
        if not speed_folders:
            print("No speed result folders found")
            return pd.DataFrame()
        
        print(f"Found {len(speed_folders)} speed folders: {[f'{s}x' for s, _ in speed_folders]}")
        
        all_results = []
        
        for speed, speed_dir in tqdm(speed_folders, desc="Processing speeds"):
            checkpoint_dirs = self.find_checkpoint_evaluation_folders(speed_dir)
            
            if not checkpoint_dirs:
                print(f"No checkpoint evaluations found for speed {speed}x")
                continue
            
            print(f"Speed {speed}x: {len(checkpoint_dirs)} checkpoints")
            
            for step, checkpoint_dir in checkpoint_dirs:
                data = self.load_evaluation_data(checkpoint_dir)
                
                # Require both CER and exposure data
                if data['cer_data'] is None or data['exposure_data'] is None:
                    raise ValueError(f"Missing evaluation data for {checkpoint_dir.name}. Both CER and exposure data required.")
                
                cer_df = data['cer_data']
                exposure_df = data['exposure_data']
                cer_columns = ['canary_id', 'frequency', 'CER', 'speaker_id', 'reference', 'prediction', 'audio_path']
                
                # Merge CER and exposure data on canary_id
                merged_df = pd.merge(
                    cer_df[cer_columns],
                    exposure_df[['canary_id', 'exposure', 'rank']],
                    on='canary_id',
                    how='inner'
                )
                
                # Add experiment details for identification
                merged_df['speed'] = speed
                merged_df['checkpoint_step'] = step  
                merged_df['speed_folder'] = speed_dir.name
                merged_df['checkpoint_folder'] = checkpoint_dir.name
                
                all_results.append(merged_df)
                    
        combined_df = pd.concat(all_results, ignore_index=True)

        # Reorder columns for better readability
        column_order = [
            'speed', 'checkpoint_step', 'frequency', 'canary_id', 'speaker_id',
            'CER', 'exposure', 'rank', 'reference', 'prediction', 'audio_path',
            'speed_folder', 'checkpoint_folder'
        ]

        combined_df = combined_df[column_order]
        return combined_df
    
    def save_results(self, master_df):
        """Save all results to MASTER CSV files"""
        if master_df.empty:
            print("No data to save")
            return
        
        # Save master CSV
        master_file = self.output_dir / "MASTER.csv"
        master_df.to_csv(master_file, index=False)
        print(f"Master results saved to: {master_file}")
        
        # Print "sanity check" of analysis of debugging
        print(f"\nCombined Results Summary:")
        print(f"Total rows: {len(master_df):,}")
        print(f"Speeds: {sorted(master_df['speed'].unique())}")
        print(f"Checkpoint steps: {sorted(master_df['checkpoint_step'].unique())}")
        print(f"Frequencies: {sorted(master_df['frequency'].unique())}")
        
        if 'speaker_id' in master_df.columns:
            print(f"Unique speakers: {master_df['speaker_id'].nunique()}")
        
        if 'CER' in master_df.columns:
            print(f"CER range: {master_df['CER'].min():.4f} - {master_df['CER'].max():.4f}")
        if 'exposure' in master_df.columns and not master_df['exposure'].isna().all():
            print(f"Exposure range: {master_df['exposure'].min():.4f} - {master_df['exposure'].max():.4f}")
    
    def run_analysis(self):
        """Run the complete across-speed analysis"""
        print(f"Starting across-speed analysis for: {self.base_dir}")
        
        master_df = self.combine_all_results()
        self.save_results(master_df)
        
        return master_df


def main():
    parser = argparse.ArgumentParser(description='Combine CER and exposure results across all speeds')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing speed result folders')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory {args.base_dir} does not exist")
        return
    
    # Initialise analyser
    analyser = AcrossSpeedAnalyser(args.base_dir)
    
    # Run analysis
    master_df = analyser.run_analysis()
    
    if master_df is not None:
        print(f"\nAnalysis complete. Results saved to: {analyser.output_dir}")
    else:
        print("Analysis failed - no data found")


if __name__ == "__main__":
    main()