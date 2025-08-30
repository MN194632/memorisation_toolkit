import os
import torch
import json
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset
from dataloader_libri import LibriSpeechDataset
from dataloader_canaries import CanariesDataset
from dataset_utils import DataCollatorCTCWithPadding

class CanaryFrequencyCoordinator:
    """
    Coordinates the mixing of canaries into training data at different frequencies.
    Tracks which canaries were mixed in at what frequency.
    """
    
    def __init__(self, config, frequencies=[1, 2, 4, 8, 16], seed=42):
        """
        Initialise the coordinator with configuration and frequencies.
        
        Args:
            config: Configuration object with dataset paths
            frequencies: List of frequencies to mix canaries (default: [1, 2, 4, 8, 16])
            seed: Random seed for reproducibility
        """
        self.config = config
        self.frequencies = frequencies
        self.seed = seed
        self.output_dir = os.path.join(config.output_dir, "canary_tracking")
        os.makedirs(self.output_dir, exist_ok=True)
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
    def _load_datasets(self):
        self.train_dataset = LibriSpeechDataset(
            self.config.train_csv, 
            self.config.base_audio_dir,
            self.config.processor,
            self.config.tokenizer
        )
        
        self.canaries_dataset = CanariesDataset(
            self.config.canaries_dir, 
            self.config.processor,
            self.config.target_sample_rate
        )
        
        print(f"Loaded {len(self.train_dataset)} training samples and {len(self.canaries_dataset)} canaries")
        
        
    def _assign_frequencies_to_canaries(self):
        """
        Assign frequencies to canaries, dividing them equally among frequency groups.
        Returns a list of canary indices to include in the training set, with repetitions.
        """
        num_canaries = len(self.canaries_dataset)
        
        # Calculate how many canaries each frequency group should get
        base_per_freq = num_canaries // len(self.frequencies)
        remainder = num_canaries % len(self.frequencies)
        
        # Create list of group sizes - distribute remainder to first groups
        group_sizes = []
        for i in range(len(self.frequencies)):
            if i < remainder:
                group_sizes.append(base_per_freq + 1)
            else:
                group_sizes.append(base_per_freq)
        
        if min(group_sizes) == 0:
            print(f"Warning: Only {num_canaries} canaries for {len(self.frequencies)} frequency groups")
        
        # Divide canaries among frequency groups
        all_indices = list(range(num_canaries))

        #! REMOVE THIS IS NOT RANDOMISED ASSIGNMENT
        random.shuffle(all_indices)  # Randomise assignments 
        
        expanded_indices = []
        assignments = {}
        
        # Track how many canaries are assigned to each frequency
        freq_distribution = {freq: 0 for freq in self.frequencies}
        
        idx_pointer = 0
        for i, freq in enumerate(self.frequencies):
            
            # Get the predetermined number of canaries for this frequency group
            current_batch_size = group_sizes[i]
            these_indices = all_indices[idx_pointer:idx_pointer + current_batch_size]
            
            # Create assignments and expand indices
            for idx in these_indices:
                # Get canary file path directly from the original samples list
                audio_path = self.canaries_dataset.samples[idx]['audio_path']
                
                # Create a canary ID based on the folder name 
                folder_name = os.path.basename(os.path.dirname(audio_path))
                canary_id = f"canary_{folder_name}"
                

                sample = self.canaries_dataset[idx]
                
                # Add assignment info
                assignments[str(idx)] = { 
                    'id': canary_id,
                    'dataset_index': idx,
                    'frequency': freq,
                    'transcript': sample['transcript'],
                    'speaker_id': sample['speaker_id'],
                    'audio_path': audio_path,
                    'duration': sample['duration']
                }
                
                # Add this index to expanded_indices multiple times
                expanded_indices.extend([idx] * freq)
                
                # Update frequency distribution
                freq_distribution[freq] += 1
            
            idx_pointer += current_batch_size
        
        # Log frequency distribution
        print("Canary assignment distribution:")
        for freq, count in freq_distribution.items():
            print(f"Frequency {freq}: {count} unique canaries ({count * freq} total instances)")
        
        # Save assignments for later analysis
        self.canary_assignments = assignments
        self._save_canary_assignments()
        
        return expanded_indices
    
    def _save_canary_assignments(self):
        csv_file = os.path.join(self.output_dir, "canary_assignments.csv")
        with open(csv_file, "w", newline='') as f:
            f.write("canary_idx,canary_id,dataset_index,frequency,speaker_id,audio_path,duration,transcript\n")
            for idx, info in self.canary_assignments.items():
                f.write(f"{idx},{info['id']},{info['dataset_index']},{info['frequency']},{info['speaker_id']},{info['audio_path']},{info['duration']},\"{info['transcript']}\"\n")
            
        print(f"Saved canary frequency assignments to {csv_file}")

    
    def create_mixed_dataset(self,only_canaries):
        """
        Create a mixed dataset with canaries at different frequencies.
        
        Returns:
            ConcatDataset: A dataset combining the training data with canaries at different frequencies
        """
        self._load_datasets()
        expanded_indices = self._assign_frequencies_to_canaries()
        
        # Create a subset dataset with the expanded indices
        augmented_canaries = Subset(self.canaries_dataset, expanded_indices)

        # If we only want to test finetuning with the canaries 
        if only_canaries == True:
            dataset = augmented_canaries
            print(f"Created dataset with {len(augmented_canaries)} canary samples " 
                f"({len(self.canary_assignments)} unique canaries at varying frequencies)")
        else:
            mixed_dataset = ConcatDataset([self.train_dataset, augmented_canaries])
            dataset = mixed_dataset
        
        # Log the dataset sizes for debugging
            print(f"Created mixed dataset with {len(self.train_dataset)} training samples " 
                f"and {len(augmented_canaries)} canary samples " 
                f"({len(self.canary_assignments)} unique canaries at varying frequencies)")
        
        return dataset
    
    def create_dataloaders(self):
        """
        Create dataloaders with mixed-in canaries.
        
        Returns:
            tuple: (train_loader, dev_loader, test_loader)
        """
        data_collator = DataCollatorCTCWithPadding(
            processor=self.config.processor, 
            padding=True
        )
        
        if self.config.only_canaries:
            # Create dataset and dataloader only with canaries for training
            mixed_train_dataset = self.create_mixed_dataset(only_canaries=True)
            train_loader = DataLoader(
                mixed_train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=self.config.num_workers,
                collate_fn=data_collator
            )
        else:
           # Create mixed dataset and dataloaders for training
            mixed_train_dataset = self.create_mixed_dataset(only_canaries=False)
            train_loader = DataLoader(
                mixed_train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=self.config.num_workers,
                collate_fn=data_collator
            )
        
        # Create dev and test datasets (no canaries)
        dev_dataset = LibriSpeechDataset(
            self.config.dev_csv, 
            self.config.base_audio_dir,
            self.config.processor,
            self.config.tokenizer
        )
        
        test_dataset = LibriSpeechDataset(
            self.config.test_csv, 
            self.config.base_audio_dir,
            self.config.processor,
            self.config.tokenizer
        )
        
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers,
            collate_fn=data_collator
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers,
            collate_fn=data_collator
        )
        
        return train_loader, dev_loader, test_loader

