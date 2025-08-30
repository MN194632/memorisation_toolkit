# Memorisation Risk Assessment System
## Investigating Memorisation in AI: A Benchmarking Approach


A comprehensive implementation for evaluating AI model memorisation through synthetic "canary" data. This system provides end-to-end functionality for creating, tracking, training with, and evaluating canaries across different experimental conditions.

## Introduction
This repo is the code base for the thesis "Investigating Memorisation in AI: A Benchmarking Approach" at the Technical University of Denmark (DTU). The thesis is written by Maria Christine Neiiendam and supervised by Sneha Das and Line Clemmensen. 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Canary Creation System](#canary-creation-system)
- [File Structure Requirements](#file-structure-requirements)
- [Configuration Management](#configuration-management)
- [Training and Tracking](#training-and-tracking)
- [Evaluation Framework](#evaluation-framework)
- [Usage Workflow](#usage-workflow)
- [Output Structure](#output-structure)

## Overview

This system enables researchers to systematically study memorisation in AI models by:
- Generating synthetic "canary" data with controllable properties
- Integrating canaries into training datasets at specified frequencies
- Tracking which canaries are memorised during training
- Calculating exposure metrics to quantify memorisation risk
- Providing comprehensive analysis across multiple experimental conditions

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd memorisation_toolkit

# Create virtual environment
python -m venv memory
source memory/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

## Canary Creation System

The system provides two primary methods for generating synthetic canary data:

### Synthetic Canary Generation

Generate synthetic speech samples using Microsoft's SpeechT5 text-to-speech model:

```bash
# Generate 100 synthetic canaries at 1.5x speed
python canary_experiments/create_random_sentence_canaries.py \
    --no_canaries 100 \
    --speed 1.5 \
    --samples_dir /path/to/15x_samples \
    --seed 42
```

**Features:**
- **Text Generation**: Creates sentences by randomly sampling 10 words from the top 10,000 most common English words
- **Speech Synthesis**: Converts text to speech using SpeechT5 with randomised speaker embeddings from the CMU Arctic dataset
- **Multi-speed Generation**: Produces audio at various speed factors (1x, 1.5x, 2x, 2.5x, 3x, 3.5x, 4x)
- **Batch Processing**: Supports generating large numbers of canaries with specified parameters

### Natural Sentence Canaries

Create canaries from natural language for more realistic evaluation:

```bash
# Generate canaries from natural sentences at 2.5x speed
python canary_experiments/create_normal_sentence_canaries.py \
    --no_canaries 1000 \
    --speed 2.5 \
    --samples_dir /path/to/25x_samples \
    --seed 42
```

**Features:**
- **Source Data**: Utilises the C4 dataset
- **Filtering**: Applies length constraints (40-70 characters) and removes sentences with numbers or special characters
- **Multi-speed Support**: Generates the same sentence at multiple speed factors for comparative analysis

## File Structure Requirements

Each generated canary follows a standardised directory structure:

```
canary_directory/
├── timestamp_folder_1/
│   ├── speaker_id.txt      # CMU Arctic speaker identifier
│   ├── sentence.txt        # Ground truth transcription
│   ├── speech.wav         # Normal speed audio (16kHz)
│   └── canary.wav         # Speed-modified audio
├── timestamp_folder_2/
│   ├── speaker_id.txt
│   ├── sentence.txt
│   ├── speech.wav
│   └── canary.wav
└── ...
```

**Required Files:**
- **`speaker_id.txt`**: Contains the CMU Arctic speaker identifier
- **`sentence.txt`**: Contains the ground truth transcription text
- **`speech.wav`**: Normal speed audio file at 16kHz sample rate
- **`canary.wav`**: Speed-modified audio file for the specific experiment

## Configuration Management

The system uses a hierarchical configuration system supporting multiple experimental conditions. **Important**: The `output_dir` in your configuration must include the speed indicator in the directory name (e.g., `1x`, `15x`, `2x`, `25x`, `3x`, `35x`, `4x`) to ensure proper organisation and analysis of results.

```python
# Example configuration in CONFIG.py
class Config15(BaseConfig):
    # Output and model paths - NOTE: Include speed indicator (15x) in directory name
    output_dir = "/path/to/CANARY_RESULTS/low_freq/15x_results"
    
    # Canary paths and settings
    canaries_dir = "/path/to/15x_samples"
    canaries_tracking_dir = os.path.join(output_dir, "canary_tracking")
    
    # Training configuration
    max_steps = 100_000
    save_steps = 10_000
    eval_steps = 10_000
    only_canaries = False  
    freeze = True           # Freeze all except CTC head
    speed = 1.5
    frequencies = [1, 2, 4, 8, 16]  # Low frequency experiment
```

**Configuration Variants:**
- **Speed Experiments**: Different audio speed factors (1x to 4x)
- **Semantics Experiments**: Random and normal canaries
- **Training Modes**: Frozen (CTC head only) vs unfrozen (full model) training


## Training and Tracking

### Canary Frequency Coordinator

The core tracking system manages the integration of canaries into training datasets:

```python
# Automatic frequency assignment and dataset mixing
coordinator = CanaryFrequencyCoordinator(config, frequencies=[1, 2, 4, 8, 16])
mixed_dataset = coordinator.create_mixed_dataset(only_canaries=False)
```

**Features:**
- **Frequency Assignment**: Automatically distributes canaries across predefined frequency groups
- **Dataset Mixing**: Creates mixed datasets combining LibriSpeech training data with replicated canaries
- **Metadata Tracking**: Maintains detailed records of canary usage, frequencies, and speaker information
- **Reproducibility**: Uses seeded random assignment for consistent experimental conditions

**Generated Tracking Files:**
- `canary_assignments.csv`: Complete mapping of canaries to frequencies
- `frequency_summary.csv`: Statistical overview of frequency distributions

### Training with Canaries

```bash
# Train model with integrated canaries
python canary_experiments/FINETUNING.py --config Config15
```

## Evaluation Framework

### Core Evaluation Pipeline

Run comprehensive assessment of trained models:

```bash
# Evaluate a specific checkpoint
python canary_experiments/EVALUATION.py \
    --config Config15 \
    --checkpoint_dir /path/to/checkpoint-100000 \
    --speed 1.5 \
    --hold_out_folder /path/to/holdout \
    --max_samples 20000
```

**Pipeline Components:**
1. **Canary Evaluation**: Tests model performance on tracked canaries
2. **Exposure Analysis**: Calculates memorisation metrics using hold-out set comparisons  
3. **Generalisation Testing**: Evaluates model performance on LibriSpeech test set
4. **Visualisation**: Generates comprehensive plots showing exposure trends

### Hold-out Set Generation and Evaluation

```bash
# Generate hold-out CER data for exposure calculation
python canary_experiments/HOLDOUT.py \
    --config Config15 \
    --checkpoint_dir /path/to/checkpoint-100000 \
    --holdout_folder /path/to/holdout_samples \
    --speed 1.5 \
    --max_samples 20000
```

### Cross-Experiment Analysis

For comparative studies across multiple experimental conditions:

```bash
# Combine results across all speed experiments
python canary_experiments/across_analyser.py \
    --base_dir /path/to/experiment_results

# Generate trend analysis plots
python canary_experiments/trend_analyser.py \
    --output_dir /path/to/experiment_results \
    --baseline_csv baseline_canaries.csv
```

**Analysis Features:**
- **Aggregates Results**: Combines evaluation data across all conditions and checkpoints
- **Creates Master Dataset**: Generates `MASTER.csv` with unified results
- **Statistical Summaries**: Produces summary statistics by checkpoint and frequency
- **Trend Analysis**: Enables longitudinal analysis of memorisation patterns

## Usage Workflow

### 1. Experiment Setup

Define your experimental configuration with proper naming:

```python
# In CONFIG.py
class MyExperiment(BaseConfig):
    # IMPORTANT: Include speed indicator in directory name
    output_dir = "/path/to/results/2x_results"  # Note: 2x for 2.0 speed
    canaries_dir = "/path/to/2x_samples"
    speed = 2.0
    frequencies = [1, 2, 4, 8, 16]
    freeze = True
```

### 2. Canary Generation

```bash
# Generate synthetic canaries
python canary_experiments/create_random_sentence_canaries.py \
    --no_canaries 100 \
    --speed 2 \
    --samples_dir /path/to/2x_samples

# Or use natural sentences
python canary_experiments/create_normal_sentence_canaries.py \
    --no_canaries 1000 \
    --speed 2 \
    --samples_dir /path/to/2x_samples
```

### 3. Model Training

```bash
# Train with canary integration
python canary_experiments/FINETUNING.py --config MyExperiment
```

### 4. Comprehensive Evaluation

```bash
# Evaluate all checkpoints
for checkpoint in /path/to/results/2x_results/checkpoint-*; do
    python canary_experiments/EVALUATION.py \
        --config MyExperiment \
        --checkpoint_dir "$checkpoint" \
        --speed 2.0 \
        --hold_out_folder /path/to/holdout \
        --max_samples 20000
done
```

### 5. Cross-Condition Analysis

```bash
# Analyse across all conditions
python canary_experiments/across_analyser.py \
    --base_dir /path/to/all_results

python canary_experiments/trend_analyser.py \
    --output_dir /path/to/all_results \
    --baseline_csv baseline_canaries.csv
```

## Output Structure

The system generates organised output directories:

```
output_dir/                    # e.g., "15x_results"
├── canary_tracking/
│   ├── canary_assignments.csv
│   └── frequency_summary.csv
├── evaluation/
│   └── checkpoint-xxxxx/
│       ├── tracked_canary_evaluation.csv
│       ├── tracked_canary_exposure.csv
│       ├── librispeech_evaluation_samples.csv
│       └── plots/
├── forgetting/                # If running forgetting experiments
│   └── checkpoint-xxxxx/
└── across_speed_analysis/     # If running cross-speed analysis
    ├── MASTER.csv
    ├── summary_by_checkpoint.csv
    └── trend_plots/
```

This implementation provides a complete toolkit for studying memorisation in ASR using canaries, from controlled data generation through comprehensive analysis of memorisation patterns across different experimental conditions. The modular design allows for easy extension to new experimental paradigms whilst maintaining rigorous tracking and reproducibility standards.
