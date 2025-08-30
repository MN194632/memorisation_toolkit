import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import random
from typing import List, Dict, Tuple, Union
from torchmetrics.text import CharErrorRate 
from torch.utils.data import DataLoader, Subset

# Import from existing codebase
import sys
sys.path.append('/zhome/76/b/147012/memorisation')
from canary_experiments.dataloader_libri import LibriSpeechDataset
from canary_experiments.config import Config

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3


class MultiSpeedEmbeddingAnalyser:
    """
    Analyses embeddings from Wav2Vec2 for:
    - LibriSpeech samples (real speech)
    - RANDOM canaries at multiple speeds (1x, 1.5x, 2x, 2.5x, 3x, 3.5x, 4x)
    - NORMAL canaries at multiple speeds (1x, 1.5x, 2x, 2.5x, 3x, 3.5x, 4x)
    """
    
    def __init__(self,random_canaries_path, normal_canaries_path,model_name="facebook/wav2vec2-base-960h", use_finetuned=False, ):
        """
        Initialise the analyser with Wav2Vec2 model
        
        Args:
            model_name: Name of the Wav2Vec2 checkpoint to use
            use_finetuned: Whether to use the fine-tuned model from config
        """
        # Load config for dataloader paths
        self.config = Config()
        
        if use_finetuned and os.path.exists(self.config.checkpoint_dir):
            print(f"Loading fine-tuned Wav2Vec2 model from {self.config.checkpoint_dir}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.config.checkpoint_dir)
            self.model = Wav2Vec2Model.from_pretrained(self.config.checkpoint_dir).to(DEVICE)
        else:
            print(f"Loading Wav2Vec2 model: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
        
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path("embedding_experiments/multi_speed_analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define speed mapping
        self.speed_mapping = {
            "1x_samples": 1.0,
            "15x_samples": 1.5,
            "2x_samples": 2.0,
            "25x_samples": 2.5,
            "3x_samples": 3.0,
            "35x_samples": 3.5,
            "4x_samples": 4.0
        }
        
        # Define canary paths
        self.canary_paths = {
            "RANDOM": random_canaries_path,
            "NORMAL": normal_canaries_path
        }
        
    def resample_if_needed(self, waveform, sample_rate, target_rate=16000):
        """Resample audio to target sample rate if needed"""
        if sample_rate != target_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
            waveform = resampler(waveform)
        return waveform
    
    def load_audio_file(self, file_path):
        """Load audio from file"""
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample to 16kHz if needed
            waveform = self.resample_if_needed(waveform, sample_rate)
            
            return waveform.squeeze(), 16000
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def extract_embeddings(self, waveform, layer=-1):
        """Extract embeddings from Wav2Vec2 for the given waveform"""
        waveform = waveform.to(DEVICE)
        
        with torch.no_grad():
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract the specified layer's hidden states
            hidden_states = outputs.hidden_states[layer]
            
            # Average over time dimension (1,T,768) --> (1,768)
            embedding = hidden_states.mean(dim=1).cpu().numpy()
            
            return embedding
    
    def process_speed_folder(self, speed_folder_path, speed_name, canary_type, layer=-1):
        """
        Process all samples in a specific speed folder
        
        Args:
            speed_folder_path: Path to speed folder (e.g., "1x_samples")
            speed_name: Name of the speed for labeling
            canary_type: Type of canary ("RANDOM" or "NORMAL")
            layer: Which layer to extract embeddings from
            
        Returns:
            Dictionary with embeddings and metadata
        """
        speed_folder = Path(speed_folder_path)
        if not speed_folder.exists():
            print(f"Warning: Speed folder not found: {speed_folder}")
            return {"embeddings": [], "labels": [], "sentences": [], "speed": [], "type": []}
        
        embeddings = []
        labels = []
        sentences = []
        speeds = []
        types = []
        
        # Get all sample subdirectories
        sample_folders = [folder for folder in speed_folder.iterdir() if folder.is_dir()]
        
        print(f"Processing {len(sample_folders)} {canary_type} samples from {speed_name}...")
        for sample_folder in tqdm(sample_folders, desc=f"Processing {canary_type} {speed_name}"):
            try:
                # Load sentence
                sentence_path = sample_folder / "sentence.txt"
                sentence = ""
                if sentence_path.exists():
                    with open(sentence_path, "r") as f:
                        sentence = f.read().strip()
                
                # Load audio - use canary.wav for the sped-up version
                audio_path = sample_folder / "canary.wav"
                if audio_path.exists():
                    waveform, sample_rate = self.load_audio_file(audio_path)
                    if waveform is not None:
                        # Extract embeddings
                        embedding = self.extract_embeddings(waveform, layer)
                        
                        # Store results
                        embeddings.append(embedding)
                        labels.append(f"{canary_type} {speed_name}: {sample_folder.name}")
                        sentences.append(sentence)
                        speeds.append(speed_name)
                        types.append(canary_type)
                
            except Exception as e:
                print(f"Error processing {sample_folder}: {e}")
                continue
        
        return {
            "embeddings": embeddings,
            "labels": labels,
            "sentences": sentences,
            "speed": speeds,
            "type": types
        }
    
    def process_all_speed_folders_both_types(self, layer=-1):
        """
        Process all speed folders for both RANDOM and NORMAL canary types
        
        Args:
            layer: Which layer to extract embeddings from
            
        Returns:
            Dictionary with all speed data for both types
        """
        all_data = {
            "embeddings": [],
            "labels": [],
            "sentences": [],
            "speeds": [],
            "numeric_speeds": [],
            "types": []
        }
        
        for canary_type, base_path in self.canary_paths.items():
            print(f"\n{'='*40}")
            print(f"Processing {canary_type} canaries from {base_path}")
            print(f"{'='*40}")
            
            canary_base = Path(base_path)
            if not canary_base.exists():
                print(f"Warning: Base directory not found: {canary_base}")
                continue
            
            for folder_name, speed_value in self.speed_mapping.items():
                speed_folder = canary_base / folder_name
                
                if speed_folder.exists():
                    print(f"\nProcessing {canary_type} {folder_name} (speed: {speed_value}x)...")
                    speed_data = self.process_speed_folder(
                        speed_folder, f"{speed_value}x", canary_type, layer
                    )
                    
                    # Combine data
                    all_data["embeddings"].extend(speed_data["embeddings"])
                    all_data["labels"].extend(speed_data["labels"])
                    all_data["sentences"].extend(speed_data["sentences"])
                    all_data["speeds"].extend([f"{speed_value}x"] * len(speed_data["embeddings"]))
                    all_data["numeric_speeds"].extend([speed_value] * len(speed_data["embeddings"]))
                    all_data["types"].extend([canary_type] * len(speed_data["embeddings"]))
                else:
                    print(f"Warning: {canary_type} speed folder not found: {speed_folder}")
        
        print(f"\nTotal canary samples processed: {len(all_data['embeddings'])}")
        print(f"RANDOM samples: {sum(1 for t in all_data['types'] if t == 'RANDOM')}")
        print(f"NORMAL samples: {sum(1 for t in all_data['types'] if t == 'NORMAL')}")
        return all_data
    
    def load_librispeech_samples_from_dataloader(self, num_samples=10, split='test'):
        """Load LibriSpeech samples using the existing dataloader"""
        # Get the appropriate CSV path based on split
        if split == 'train':
            csv_path = self.config.train_csv
        elif split == 'dev':
            csv_path = self.config.dev_csv
        else:  # test
            csv_path = self.config.test_csv
        
        # Create dataset
        dataset = LibriSpeechDataset(
            csv_path,
            self.config.base_audio_dir,
            self.config.processor,
            self.config.tokenizer
        )
        
        print(f"Loaded {split} dataset with {len(dataset)} total samples")
        
        # Create subset if needed
        if num_samples < len(dataset):
            indices = random.sample(range(len(dataset)), num_samples)
            subset = Subset(dataset, indices)
        else:
            subset = dataset
            
        samples = []
        print(f"Processing {len(subset)} LibriSpeech samples from {split} set...")
        
        for i in tqdm(range(len(subset))):
            try:
                sample = subset[i]
                input_values = sample['input_values'].squeeze()
                
                samples.append({
                    'waveform': input_values,
                    'sample_rate': 16000,
                    'transcript': sample['transcript'],
                    'speaker_id': sample['speaker_id'],
                    'duration': sample['duration'],
                    'label': f"LibriSpeech: {sample['speaker_id']}"
                })
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"Successfully loaded {len(samples)} LibriSpeech samples")
        return samples
    
    def process_librispeech_samples_from_dataloader(self, num_samples=10, split='test', layer=-1):
        """Process LibriSpeech samples and extract embeddings"""
        samples = self.load_librispeech_samples_from_dataloader(num_samples, split)
        
        embeddings = []
        labels = []
        transcriptions = []
        
        print("Extracting embeddings from LibriSpeech samples...")
        for sample in tqdm(samples):
            waveform = sample['waveform']
            embedding = self.extract_embeddings(waveform, layer)
            
            embeddings.append(embedding)
            labels.append(sample['label'])
            transcriptions.append(sample['transcript'])
        
        return {
            "embeddings": embeddings,
            "labels": labels,
            "transcriptions": transcriptions
        }
    
    def visualise_multi_speed_tsne(self, data_dict, perplexity=30, random_state=42):
        """
        Visualise embeddings using t-SNE with different colors for each speed and type
        """
        # Collect all embeddings and metadata
        all_embeddings = []
        all_labels = []
        all_types = []
        all_speeds = []
        all_canary_types = []
        
        # Add LibriSpeech data
        if data_dict["librispeech"]["embeddings"]:
            for emb, label in zip(data_dict["librispeech"]["embeddings"], data_dict["librispeech"]["labels"]):
                all_embeddings.append(emb)
                all_labels.append(label)
                all_types.append("LibriSpeech")
                all_speeds.append("LibriSpeech")
                all_canary_types.append("LibriSpeech")
        
        # Add canary data
        if data_dict["canaries"]["embeddings"]:
            for emb, label, speed, canary_type in zip(
                data_dict["canaries"]["embeddings"], 
                data_dict["canaries"]["labels"], 
                data_dict["canaries"]["speeds"],
                data_dict["canaries"]["types"]
            ):
                all_embeddings.append(emb)
                all_labels.append(label)
                all_types.append("Canary")
                all_speeds.append(speed)
                all_canary_types.append(canary_type)
        
        if len(all_embeddings) < 3:
            print("Not enough embeddings for t-SNE visualisation")
            return
        
        stacked_embeddings = np.vstack(all_embeddings)
        
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(all_embeddings)-1),
            random_state=random_state,
            learning_rate='auto',
            init='pca'
        )
        
        print(f"Running t-SNE with perplexity {min(perplexity, len(all_embeddings)-1)}...")
        tsne_result = tsne.fit_transform(stacked_embeddings)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Get unique speeds and sort them
        unique_speeds = list(set(all_speeds))
        
        def sort_key(speed):
            if speed == "LibriSpeech":
                return -1  
            else:
                return float(speed.replace('x', ''))  
        
        unique_speeds.sort(key=sort_key)
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(unique_speeds)]
        speed_colors = dict(zip(unique_speeds, colors))
        
        # Define markers for different types
        markers = {'LibriSpeech': 'o', 'RANDOM': '^', 'NORMAL': 's'}
        
        # Plot 1: Colored by speed
        for i, (x, y) in enumerate(tsne_result):
            speed = all_speeds[i]
            canary_type = all_canary_types[i]
            marker = 'o' if canary_type == "LibriSpeech" else ('^' if canary_type == "RANDOM" else 's')
            
            ax1.scatter(
                x, y,
                color=speed_colors[speed],
                marker=marker,
                s=80,
                alpha=0.8,
                edgecolors='black',
                linewidth=0.8
            )
        
        # Create legend for plot 1
        legend_elements = []
        for speed in unique_speeds:
            if speed == "LibriSpeech":
                legend_elements.append(
                    plt.scatter([], [], color=speed_colors[speed], marker='o', s=80, 
                              alpha=0.8, edgecolors='black', linewidth=0.8, label=speed)
                )
            else:
                # Show both RANDOM and NORMAL for each speed
                legend_elements.append(
                    plt.scatter([], [], color=speed_colors[speed], marker='^', s=80, 
                              alpha=0.8, edgecolors='black', linewidth=0.8, label=f"'Speed' {speed}")
                )

        ax1.set_title("t-SNE: Colored by Speed\n(Triangle=RANDOM, Square=NORMAL, Circle=LibriSpeech)", 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel("t-SNE Dimension 1")
        ax1.set_ylabel("t-SNE Dimension 2")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Colored by canary type
        type_colors = {'LibriSpeech': 'blue', 'RANDOM': 'red', 'NORMAL': 'green'}
        
        for i, (x, y) in enumerate(tsne_result):
            canary_type = all_canary_types[i]
            marker = markers[canary_type]
            
            ax2.scatter(
                x, y,
                color=type_colors[canary_type],
                marker=marker,
                s=80,
                alpha=0.8,
                edgecolors='black',
                linewidth=0.8
            )
        
        # Create legend for plot 2
        type_legend_elements = []
        for type_name in ['LibriSpeech', 'RANDOM', 'NORMAL']:
            type_legend_elements.append(
                plt.scatter([], [], color=type_colors[type_name], marker=markers[type_name], 
                          s=80, alpha=0.8, edgecolors='black', linewidth=0.8, label=type_name)
            )
        
        ax2.legend(handles=type_legend_elements, loc='upper right', fontsize=12)
        ax2.set_title("t-SNE: Colored by Type", fontsize=12, fontweight='bold')
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RANDOM canaries only
        random_indices = [i for i, ct in enumerate(all_canary_types) if ct in ['LibriSpeech', 'RANDOM']]
        if random_indices:
            for i in random_indices:
                x, y = tsne_result[i]
                speed = all_speeds[i]
                canary_type = all_canary_types[i]
                marker = 'o' if canary_type == "LibriSpeech" else '^'
                
                ax3.scatter(
                    x, y,
                    color=speed_colors[speed],
                    marker=marker,
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.8
                )
            ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax3.set_title("t-SNE: RANDOM Canaries + LibriSpeech", fontsize=12, fontweight='bold')
            ax3.set_xlabel("t-SNE Dimension 1")
            ax3.set_ylabel("t-SNE Dimension 2")
            ax3.grid(True, alpha=0.3)
        
        legend_elements = []
        for speed in unique_speeds:
            if speed == "LibriSpeech":
                legend_elements.append(
                    plt.scatter([], [], color=speed_colors[speed], marker='o', s=80, 
                              alpha=0.8, edgecolors='black', linewidth=0.8, label=speed)
                )
            else:
                # Show both RANDOM and NORMAL for each speed
                legend_elements.append(
                    plt.scatter([], [], color=speed_colors[speed], marker='s', s=80, 
                              alpha=0.8, edgecolors='black', linewidth=0.8, label=f"'Speed' {speed}")
                )
        # Plot 4: NORMAL canaries only
        normal_indices = [i for i, ct in enumerate(all_canary_types) if ct in ['LibriSpeech', 'NORMAL']]
        if normal_indices:
            for i in normal_indices:
                x, y = tsne_result[i]
                speed = all_speeds[i]
                canary_type = all_canary_types[i]
                marker = 'o' if canary_type == "LibriSpeech" else 's'
                
                ax4.scatter(
                    x, y,
                    color=speed_colors[speed],
                    marker=marker,
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.8
                )
            ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax4.set_title("t-SNE: NORMAL Canaries + LibriSpeech", fontsize=12, fontweight='bold')
            ax4.set_xlabel("t-SNE Dimension 1")
            ax4.set_ylabel("t-SNE Dimension 2")
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle("Multi-Speed Canary Embedding Analysis (RANDOM vs NORMAL)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "multi_speed_tsne_visualisation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multi-speed t-SNE visualisation saved to {self.output_dir}/multi_speed_tsne_visualisation.png")
    
    def visualise_speed_progression(self, data_dict):
        """
        Create a visualisation showing how embeddings change with speed for both RANDOM and NORMAL
        Includes lower triangle heatmaps and line plots for both types
        """
        if not data_dict["canaries"]["embeddings"]:
            print("No canary data available for speed progression visualisation")
            return
        
        # Separate data by canary type
        canary_embeddings = np.vstack(data_dict["canaries"]["embeddings"])
        speeds = data_dict["canaries"]["numeric_speeds"]
        types = data_dict["canaries"]["types"]
        
        # Separate RANDOM and NORMAL data
        random_mask = [t == "RANDOM" for t in types]
        normal_mask = [t == "NORMAL" for t in types]
        
        random_embeddings = canary_embeddings[random_mask] if any(random_mask) else None
        normal_embeddings = canary_embeddings[normal_mask] if any(normal_mask) else None
        random_speeds = [s for s, m in zip(speeds, random_mask) if m]
        normal_speeds = [s for s, m in zip(speeds, normal_mask) if m]
        
        # Get unique speeds
        unique_speeds = sorted(list(set(speeds)))
        
        # Calculate LibriSpeech mean embedding if available
        librispeech_mean = None
        if data_dict["librispeech"]["embeddings"]:
            librispeech_embeddings = np.vstack(data_dict["librispeech"]["embeddings"])
            librispeech_mean = np.mean(librispeech_embeddings, axis=0)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15)
        
        # Function to calculate distance matrix and mean embeddings
        def calculate_distance_matrix(embeddings_data, speeds_data, canary_type):
            if embeddings_data is None or len(embeddings_data) == 0:
                return None, None
            
            # Calculate mean embedding for each speed
            mean_embeddings = []
            for speed in unique_speeds:
                speed_mask = [s == speed for s in speeds_data]
                if any(speed_mask):
                    speed_embeddings = embeddings_data[speed_mask]
                    mean_embedding = np.mean(speed_embeddings, axis=0)
                    mean_embeddings.append(mean_embedding)
                else:
                    mean_embeddings.append(None)
            
            # Calculate pairwise distances
            distances = np.zeros((len(unique_speeds), len(unique_speeds)))
            for i in range(len(unique_speeds)):
                for j in range(len(unique_speeds)):
                    if mean_embeddings[i] is not None and mean_embeddings[j] is not None:
                        #dist = np.linalg.norm(mean_embeddings[i] - mean_embeddings[j])
                        dist = 1 - cosine_similarity([mean_embeddings[i]], [mean_embeddings[j]])[0,0] #try cosine
                        distances[i, j] = dist
                    else:
                        distances[i, j] = np.nan
            
            return distances, mean_embeddings
        
        # Calculate distance matrices for both types
        random_distances, random_mean_embeddings = calculate_distance_matrix(
            random_embeddings, random_speeds, "RANDOM"
        )
        normal_distances, normal_mean_embeddings = calculate_distance_matrix(
            normal_embeddings, normal_speeds, "NORMAL"
        )
        
        # Plot 1: RANDOM distance heatmap (upper triangle)
        ax1 = fig.add_subplot(gs[0, 0])
        if random_distances is not None:
            # Create upper triangle mask
            mask_upper = np.tril(np.ones_like(random_distances, dtype=bool))
            random_distances_masked = np.copy(random_distances)
            random_distances_masked[mask_upper] = np.nan
            
            im1 = ax1.imshow(random_distances_masked, cmap='Reds', alpha=0.8)
            ax1.set_xticks(range(len(unique_speeds)))
            ax1.set_yticks(range(len(unique_speeds)))
            ax1.set_xticklabels([f"{s}x" for s in unique_speeds])
            ax1.set_yticklabels([f"{s}x" for s in unique_speeds])
            ax1.set_title("RANDOM Canaries\nEmbedding Distance Matrix", fontweight='bold')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Add distance values as text
            for i in range(len(unique_speeds)):
                for j in range(len(unique_speeds)):
                    if i < j and not np.isnan(random_distances[i][j]):
                        text_color = "white" if random_distances[i][j] > np.nanmax(random_distances_masked) * 0.5 else "black"
                        ax1.text(j, i, f'{random_distances[i][j]:.2f}',
                               ha="center", va="center", color=text_color, fontweight='bold', fontsize=9)
        
        # Plot 2: NORMAL distance heatmap (lower triangle)
        ax2 = fig.add_subplot(gs[0, 1])
        if normal_distances is not None:
            # Create lower triangle mask
            mask_lower = np.triu(np.ones_like(normal_distances, dtype=bool))
            normal_distances_masked = np.copy(normal_distances)
            normal_distances_masked[mask_lower] = np.nan
            
            im2 = ax2.imshow(normal_distances_masked, cmap='Blues', alpha=0.8)
            ax2.set_xticks(range(len(unique_speeds)))
            ax2.set_yticks(range(len(unique_speeds)))
            ax2.set_xticklabels([f"{s}x" for s in unique_speeds])
            ax2.set_yticklabels([f"{s}x" for s in unique_speeds])
            ax2.set_title("NORMAL Canaries\nEmbedding Distance Matrix", fontweight='bold')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # Add distance values as text
            for i in range(len(unique_speeds)):
                for j in range(len(unique_speeds)):
                    if i > j and not np.isnan(normal_distances[i][j]):
                        text_color = "white" if normal_distances[i][j] > np.nanmax(normal_distances_masked) * 0.5 else "black"
                        ax2.text(j, i, f'{normal_distances[i][j]:.2f}',
                               ha="center", va="center", color=text_color, fontweight='bold', fontsize=9)
        
        # Plot 3: Combined distance plot from LibriSpeech
        ax3 = fig.add_subplot(gs[1, 0])
        if librispeech_mean is not None:
            # Calculate distances from LibriSpeech for both types
            random_libri_distances = []
            normal_libri_distances = []
            
            if random_mean_embeddings is not None:
                for speed_embedding in random_mean_embeddings:
                    if speed_embedding is not None:
                        #dist = np.linalg.norm(librispeech_mean - speed_embedding)
                        dist = 1 - cosine_similarity([librispeech_mean], [speed_embedding])[0,0] #try cosine
                        random_libri_distances.append(dist)
                    else:
                        random_libri_distances.append(np.nan)
            
            if normal_mean_embeddings is not None:
                for speed_embedding in normal_mean_embeddings:
                    if speed_embedding is not None:
                        #dist = np.linalg.norm(librispeech_mean - speed_embedding)
                        dist = 1 - cosine_similarity([librispeech_mean], [speed_embedding])[0,0] #try cosine    
                        normal_libri_distances.append(dist)
                    else:
                        normal_libri_distances.append(np.nan)
            
            # Plot lines for both types
            if random_libri_distances:
                valid_random = [(s, d) for s, d in zip(unique_speeds, random_libri_distances) if not np.isnan(d)]
                if valid_random:
                    speeds_r, dists_r = zip(*valid_random)
                    ax3.plot(speeds_r, dists_r, 'o-', linewidth=3, markersize=10, 
                           color='#d62728', markerfacecolor='#d62728', markeredgewidth=2, 
                           markeredgecolor='#a41e22', label='RANDOM Canaries')
                    
                    # Add value labels
                    for speed, dist in valid_random:
                        ax3.annotate(f'{dist:.2f}', (speed, dist), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
            
            if normal_libri_distances:
                valid_normal = [(s, d) for s, d in zip(unique_speeds, normal_libri_distances) if not np.isnan(d)]
                if valid_normal:
                    speeds_n, dists_n = zip(*valid_normal)
                    ax3.plot(speeds_n, dists_n, 's-', linewidth=3, markersize=10, 
                           color='#1f77b4', markerfacecolor='#1f77b4', markeredgewidth=2, 
                           markeredgecolor='#0d47a1', label='NORMAL Canaries')
                    
                    # Add value labels
                    for speed, dist in valid_normal:
                        ax3.annotate(f'{dist:.2f}', (speed, dist), textcoords="offset points", 
                                   xytext=(0,-15), ha='center', fontweight='bold', fontsize=9)
            
            ax3.set_xlabel("Speed Factor", fontweight='bold')
            ax3.set_ylabel("Distance from LibriSpeech", fontweight='bold')
            ax3.set_title("Embedding Distance from LibriSpeech", fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: Comparison of variances
        ax4 = fig.add_subplot(gs[1, 1])
        x_pos = np.arange(len(unique_speeds))
        width = 0.35
        
        if random_embeddings is not None and normal_embeddings is not None:
            # Get variances for comparison
            random_vars = []
            normal_vars = []
            
            for speed in unique_speeds:
                # RANDOM variance
                speed_mask_r = [s == speed for s in random_speeds]
                if any(speed_mask_r):
                    speed_embeddings_r = random_embeddings[speed_mask_r]
                    if len(speed_embeddings_r) > 1:
                        centroid_r = np.mean(speed_embeddings_r, axis=0)
                        #distances_r = [np.linalg.norm(emb - centroid_r) for emb in speed_embeddings_r]
                        distances_r = [1 - cosine_similarity([emb], [centroid_r])[0,0] for emb in speed_embeddings_r]
                        random_vars.append(np.mean(distances_r))
                    else:
                        random_vars.append(0)
                else:
                    random_vars.append(0)
                
                # NORMAL variance
                speed_mask_n = [s == speed for s in normal_speeds]
                if any(speed_mask_n):
                    speed_embeddings_n = normal_embeddings[speed_mask_n]
                    if len(speed_embeddings_n) > 1:
                        centroid_n = np.mean(speed_embeddings_n, axis=0)
                        #distances_n = [np.linalg.norm(emb - centroid_n) for emb in speed_embeddings_n]
                        distances_n = [1 - cosine_similarity([emb], [centroid_n])[0,0] for emb in speed_embeddings_n]

                        normal_vars.append(np.mean(distances_n))
                    else:
                        normal_vars.append(0)
                else:
                    normal_vars.append(0)
            
            ax4.bar(x_pos - width/2, random_vars, width, label='RANDOM', 
                   color='#d62728', alpha=0.7, edgecolor='#a41e22')
            ax4.bar(x_pos + width/2, normal_vars, width, label='NORMAL', 
                   color='#1f77b4', alpha=0.7, edgecolor='#0d47a1')
            
            ax4.set_xlabel("Speed Factor", fontweight='bold')
            ax4.set_ylabel("Mean Distance to Centroid", fontweight='bold')
            ax4.set_title("Embedding Dispersion Comparison:\nRANDOM vs NORMAL", fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f"{s}x" for s in unique_speeds])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle("Speed Progression Analysis: RANDOM vs NORMAL Canaries", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "speed_progression_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Speed progression analysis saved to {self.output_dir}/speed_progression_analysis.png")
    
    def analyse_grouped_effects(self, data_dict):
        """
        Analyse cosine distance between groups to isolate content type vs speed effects
        """
        if not data_dict["canaries"]["embeddings"]:
            print("No canary data available")
            return
        
        # Organise data by groups
        canary_embeddings = np.vstack(data_dict["canaries"]["embeddings"])
        speeds = data_dict["canaries"]["numeric_speeds"]
        types = data_dict["canaries"]["types"]
        
        # Calculate centroids for each group
        groups = {}
        unique_speeds = sorted(list(set(speeds)))
        
        for canary_type in ["RANDOM", "NORMAL"]:
            groups[canary_type] = {}
            for speed in unique_speeds:
                mask = [(t == canary_type and s == speed) for t, s in zip(types, speeds)]
                if any(mask):
                    group_embeddings = canary_embeddings[mask]
                    centroid = np.mean(group_embeddings, axis=0)
                    groups[canary_type][speed] = centroid
        
        # Add LibriSpeech reference
        if data_dict["librispeech"]["embeddings"]:
            libri_embeddings = np.vstack(data_dict["librispeech"]["embeddings"])
            groups["LibriSpeech"] = {1.0: np.mean(libri_embeddings, axis=0)}
        
        # Calculate effect matrices
        def cosine_distance(emb1, emb2):
            return 1 - cosine_similarity([emb1], [emb2])[0,0]
        
        # Content type effects (same speed, different type)
        content_effects = []
        speed_labels = []
        
        for speed in unique_speeds:
            if speed in groups.get("RANDOM", {}) and speed in groups.get("NORMAL", {}):
                dist = cosine_distance(groups["RANDOM"][speed], groups["NORMAL"][speed])
                content_effects.append(dist)
                speed_labels.append(f"{speed}x")
        
        # Speed effects within each type
        random_speed_effects = []
        normal_speed_effects = []
        speed_comparisons = []
        
        base_speed = min(unique_speeds)  # Use lowest speed as reference
        
        for speed in unique_speeds:
            if speed != base_speed:
                speed_comparisons.append(f"{base_speed}x→{speed}x")
                
                # RANDOM speed effect
                if (base_speed in groups.get("RANDOM", {}) and 
                    speed in groups.get("RANDOM", {})):
                    dist = cosine_distance(groups["RANDOM"][base_speed], groups["RANDOM"][speed])
                    random_speed_effects.append(dist)
                else:
                    random_speed_effects.append(np.nan)
                
                # NORMAL speed effect  
                if (base_speed in groups.get("NORMAL", {}) and 
                    speed in groups.get("NORMAL", {})):
                    dist = cosine_distance(groups["NORMAL"][base_speed], groups["NORMAL"][speed])
                    normal_speed_effects.append(dist)
                else:
                    normal_speed_effects.append(np.nan)
        
        # LibriSpeech distances
        libri_random_dists = []
        libri_normal_dists = []
        
        if "LibriSpeech" in groups:
            libri_centroid = groups["LibriSpeech"][1.0]
            
            for speed in unique_speeds:
                if speed in groups.get("RANDOM", {}):
                    dist = cosine_distance(libri_centroid, groups["RANDOM"][speed])
                    libri_random_dists.append(dist)
                else:
                    libri_random_dists.append(np.nan)
                    
                if speed in groups.get("NORMAL", {}):
                    dist = cosine_distance(libri_centroid, groups["NORMAL"][speed])
                    libri_normal_dists.append(dist)
                else:
                    libri_normal_dists.append(np.nan)
        
        # Create visualisation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Content type effects by speed
        if content_effects:
            ax1.bar(range(len(speed_labels)), content_effects, color=['purple'], alpha=0.7, edgecolor=['darkpurple'])
            ax1.set_xlabel("Speed")
            ax1.set_ylabel("Cosine Distance")
            ax1.set_title("Content Type Effect\n(RANDOM vs NORMAL at same speed)", fontweight='bold')
            ax1.set_xticks(range(len(speed_labels)))
            ax1.set_xticklabels(speed_labels)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(content_effects):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Speed effects comparison
        if speed_comparisons:
            x_pos = np.arange(len(speed_comparisons))
            width = 0.35
            
            valid_random = [v for v in random_speed_effects if not np.isnan(v)]
            valid_normal = [v for v in normal_speed_effects if not np.isnan(v)]
            
            if valid_random:
                ax2.bar(x_pos - width/2, [v if not np.isnan(v) else 0 for v in random_speed_effects], 
                    width, label='RANDOM', color='red', alpha=0.7, edgecolor='darkred')
            if valid_normal:
                ax2.bar(x_pos + width/2, [v if not np.isnan(v) else 0 for v in normal_speed_effects], 
                    width, label='NORMAL', color='blue', alpha=0.7, edgecolor='darkblue')
            
            ax2.set_xlabel("Speed Change")
            ax2.set_ylabel("Cosine Distance")
            ax2.set_title("Speed Effects within Type\n(Distance from 1x speed)", fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(speed_comparisons, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: LibriSpeech distances
        if libri_random_dists or libri_normal_dists:
            valid_speeds = [i for i, s in enumerate(unique_speeds) 
                        if not (np.isnan(libri_random_dists[i]) and np.isnan(libri_normal_dists[i]))]
            
            if valid_speeds:
                speeds_plot = [unique_speeds[i] for i in valid_speeds]
                random_plot = [libri_random_dists[i] if not np.isnan(libri_random_dists[i]) else None 
                            for i in valid_speeds]
                normal_plot = [libri_normal_dists[i] if not np.isnan(libri_normal_dists[i]) else None 
                            for i in valid_speeds]
                
                if any(x is not None for x in random_plot):
                    ax3.plot(speeds_plot, [x if x is not None else 0 for x in random_plot], 
                            'o-', color='red', linewidth=3, markersize=8, label='RANDOM')
                if any(x is not None for x in normal_plot):
                    ax3.plot(speeds_plot, [x if x is not None else 0 for x in normal_plot], 
                            's-', color='blue', linewidth=3, markersize=8, label='NORMAL')
                
                ax3.set_xlabel("Speed Factor")
                ax3.set_ylabel("Cosine Distance from LibriSpeech")
                ax3.set_title("Distance from LibriSpeech\nby Canary Type and Speed", fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Effect size summary
        effects_data = []
        effect_labels = []
        
        if content_effects:
            effects_data.append(np.mean(content_effects))
            effect_labels.append("Content Type\n(avg across speeds)")
        
        if valid_random:
            effects_data.append(np.mean(valid_random))
            effect_labels.append("RANDOM Speed\n(avg effect)")
            
        if valid_normal:
            effects_data.append(np.mean(valid_normal))
            effect_labels.append("NORMAL Speed\n(avg effect)")
        
        if effects_data:
            colors = ['purple', 'red', 'blue'][:len(effects_data)]
            bars = ax4.bar(effect_labels, effects_data, color=colors, alpha=0.7, 
                        edgecolor=['darkpurple', 'darkred', 'darkblue'][:len(effects_data)])
            
            ax4.set_ylabel("Mean Cosine Distance")
            ax4.set_title("Effect Size Comparison", fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, effects_data):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "grouped_effects_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"\nGrouped Effects Analysis Results:")
        print(f"{'='*40}")
        
        if content_effects:
            print(f"Content Type Effect (RANDOM vs NORMAL):")
            for label, effect in zip(speed_labels, content_effects):
                print(f"{label}: {effect:.3f}")
            print(f"Average: {np.mean(content_effects):.3f}")
        
        if valid_random:
            print(f"\nRANDOM Speed Effects:")
            for label, effect in zip(speed_comparisons, random_speed_effects):
                if not np.isnan(effect):
                    print(f"{label}: {effect:.3f}")
            print(f"Average: {np.mean(valid_random):.3f}")
        
        if valid_normal:
            print(f"\nNORMAL Speed Effects:")
            for label, effect in zip(speed_comparisons, normal_speed_effects):
                if not np.isnan(effect):
                    print(f"{label}: {effect:.3f}")
            print(f"Average: {np.mean(valid_normal):.3f}")
        
        print(f"Grouped effects analysis saved to {self.output_dir}/grouped_effects_analysis.png")
        
        return {
            'content_effects': content_effects,
            'random_speed_effects': valid_random,
            'normal_speed_effects': valid_normal,
            'libri_distances': {'random': libri_random_dists, 'normal': libri_normal_dists}
        }
    def run_complete_multi_speed_analysis(self, num_librispeech=50, librispeech_split='test', layers=[-1]):
        """
        Run complete analysis on multi-speed canary samples for both RANDOM and NORMAL types
        
        Args:
            num_librispeech: Number of LibriSpeech samples to use
            librispeech_split: Which LibriSpeech split to use
            layers: List of layers to analyse
        """
        for layer in layers:
            print(f"\n{'='*60}")
            print(f"Processing layer {layer}...")
            print(f"{'='*60}")
            
            # Process LibriSpeech samples
            libri_data = self.process_librispeech_samples_from_dataloader(
                num_samples=num_librispeech, 
                split=librispeech_split,
                layer=layer
            )
            
            # Process all speed canary folders for both types
            canary_data = self.process_all_speed_folders_both_types(layer)
            
            # Combine data
            data_dict = {
                "librispeech": libri_data,
                "canaries": canary_data
            }
            #grouped_effects = self.analyse_grouped_effects(data_dict)
            # Create visualisations
            self.visualise_multi_speed_tsne(data_dict, perplexity=200, random_state=42)
            self.visualise_speed_progression(data_dict)
            
            # Move files to layer-specific directory if analysing multiple layers
            if len(layers) > 1:
                layer_dir = self.output_dir / f"layer_{layer}"
                layer_dir.mkdir(exist_ok=True)
                
                for file in self.output_dir.glob("*.png"):
                    if not file.name.startswith(f"layer_"):
                        file.rename(layer_dir / file.name)
                
                for file in self.output_dir.glob("*.pdf"):
                    if not file.name.startswith(f"layer_"):
                        file.rename(layer_dir / file.name)
        
        print(f"\n{'='*60}")
        print("Multi-speed analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")

        
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse embeddings of LibriSpeech samples and multi-speed canaries (RANDOM and NORMAL)")

    parser.add_argument("--random_canaries_path", type=int, default="/zhome/76/b/147012/memorisation/canary_experiments",
                        help="Path to random canaries")
    parser.add_argument("--normal_canaries_path", type=int, default="/zhome/76/b/147012/memorisation/canary_experiments/NORMAL/normal_samples",
                        help="Path to normal canaries")
    parser.add_argument("--num_librispeech", type=int, default=200,
                        help="Number of LibriSpeech samples to use")
    parser.add_argument("--librispeech_split", type=str, default="train", choices=["train", "dev", "test"],
                        help="Which LibriSpeech split to use")
    parser.add_argument("--layers", type=str, default="-1",
                        help="Comma-separated list of layers to analyse (e.g., '-1,-2,-5')")
    parser.add_argument("--use_finetuned", action="store_true",
                        help="Use fine-tuned model if available")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(',')]
    
    # Initialise analyser
    analyser = MultiSpeedEmbeddingAnalyser(random_canaries_path=args.random_canaries_path,normal_canaries_path=args.normal_canaries_path,use_finetuned=args.use_finetuned)
    
    # Check if canary base directories exist
    for canary_type, path in analyser.canary_paths.items():
        canary_base = Path(path)
        if not canary_base.exists():
            print(f"Warning: {canary_type} canary directory not found: {canary_base}")
        else:
            # Check which speed folders exist
            available_speeds = []
            for folder_name in analyser.speed_mapping.keys():
                speed_folder = canary_base / folder_name
                if speed_folder.exists():
                    available_speeds.append(folder_name)
            print(f"Found {canary_type} speed folders: {available_speeds}")
    
    # Run analysis
    analyser.run_complete_multi_speed_analysis(
        num_librispeech=args.num_librispeech,
        librispeech_split=args.librispeech_split,
        layers=layers
    )


if __name__ == "__main__":
    main()