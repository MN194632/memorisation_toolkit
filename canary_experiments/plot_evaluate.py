import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
from utils import get_config_class

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

class Plotter:
    """
    Class for creating various plots showing exposure over different frequencies
    """
    
    def __init__(self, evaluation_dir):
        """
        Initialize the plotter with evaluation directory containing CSV files
        
        Args:
            evaluation_dir (str): Path to directory containing evaluation CSV files
        """
        self.evaluation_dir = Path(evaluation_dir)
        self.output_dir = self.evaluation_dir / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Define colors for consistent plotting
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load exposure data from CSV files"""
        try:
            # Load summary data (aggregated by frequency)
            summary_file = self.evaluation_dir / "frequency_exposure_summary.csv"
            if summary_file.exists():
                self.summary_df = pd.read_csv(summary_file, index_col=0)
                print(f"Loaded summary data: {len(self.summary_df)} frequency groups")
            else:
                self.summary_df = None
                print("Warning: frequency_exposure_summary.csv not found")
            
            # Load individual canary data
            individual_file = self.evaluation_dir / "tracked_canary_exposure.csv"
            if individual_file.exists():
                self.individual_df = pd.read_csv(individual_file)
                print(f"Loaded individual data: {len(self.individual_df)} canaries")
            else:
                self.individual_df = None
                print("Warning: tracked_canary_exposure.csv not found")
            
            # Load LibriSpeech evaluation results
            librispeech_file = self.evaluation_dir / "librispeech_evaluation_samples.csv"
            if librispeech_file.exists():
                self.librispeech_df = pd.read_csv(librispeech_file)
                print(f"Loaded LibriSpeech data: {len(self.librispeech_df)} samples")
            else:
                self.librispeech_df = None
                print("Warning: librispeech_evaluation_samples.csv not found")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def plot_mean_exposure_by_frequency(self):
        """Create a line plot showing mean exposure by frequency with standard deviation error bars"""
        if self.individual_df is None:
            print("No individual data available for mean exposure plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std for each frequency group from individual data
        freq_stats = self.individual_df.groupby('frequency')['exposure'].agg(['mean', 'std']).reset_index()
        freq_stats = freq_stats.sort_values('frequency')
        
        frequencies = freq_stats['frequency'].values
        mean_exposures = freq_stats['mean'].values
        std_exposures = freq_stats['std'].values
        
        # Plot line with markers
        ax.plot(frequencies, mean_exposures, 'o-', linewidth=2, markersize=8, 
                color=self.colors[0], label='Mean Exposure')
        
        # Add error bars using standard deviation
        ax.errorbar(frequencies, mean_exposures, 
                   yerr=std_exposures, 
                   fmt='none', capsize=5, capthick=2, color=self.colors[0], alpha=0.7)
        
        ax.set_xlabel('Training Frequency', fontweight='bold')
        ax.set_ylabel('Mean Exposure', fontweight='bold')
        ax.set_title('Mean Exposure by Training Frequency', fontweight='bold', pad=20)
        ax.grid(True)
        ax.set_xticks(frequencies)
        
        # Add value labels on points
        for freq, exp in zip(frequencies, mean_exposures):
            ax.annotate(f'{exp:.2f}', (freq, exp), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mean_exposure_by_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_summary(self):
        """Create a comprehensive multi-panel figure"""
        if self.individual_df is None:
            print("Need individual data for comprehensive plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate frequency statistics from individual data
        freq_stats = self.individual_df.groupby('frequency')['exposure'].agg(['mean', 'std']).reset_index()
        freq_stats = freq_stats.sort_values('frequency')
        frequencies = freq_stats['frequency'].values
        
        # Panel 1: Mean exposure with error bars
        mean_exposures = freq_stats['mean'].values
        std_exposures = freq_stats['std'].values
        ax1.plot(frequencies, mean_exposures, 'o-', linewidth=2, markersize=6, color=self.colors[0])
        ax1.errorbar(frequencies, mean_exposures, 
                    yerr=std_exposures, 
                    fmt='none', capsize=3, color=self.colors[0], alpha=0.7)
        ax1.set_title('A) Mean Exposure by Frequency', fontweight='bold')
        ax1.set_xlabel('Training Frequency')
        ax1.set_ylabel('Mean Exposure')
        ax1.grid(True)
        ax1.set_xticks(frequencies)
        
        # Panel 2: Mean CER by frequency
        freq_cer_stats = self.individual_df.groupby('frequency')['CER'].agg(['mean', 'min', 'max']).reset_index()
        freq_cer_stats = freq_cer_stats.sort_values('frequency')
        mean_cers = freq_cer_stats['mean'].values
        min_cers = freq_cer_stats['min'].values
        max_cers = freq_cer_stats['max'].values

        # Calculate error bar values (distance from mean to min/max)
        lower_errors = mean_cers - min_cers
        upper_errors = max_cers - mean_cers

        ax2.plot(frequencies, mean_cers, 'o-', linewidth=2, markersize=6, color=self.colors[1])
        ax2.errorbar(frequencies, mean_cers,
                    yerr=[lower_errors, upper_errors],
                    fmt='none', capsize=3, color=self.colors[1], alpha=0.7)
        ax2.set_title('B) Mean CER by Frequency', fontweight='bold')
        ax2.set_xlabel('Training Frequency')
        ax2.set_ylabel('Mean CER')
        ax2.grid(True)
        ax2.set_xticks(frequencies)
        
        # Panel 3: Exposure distribution (box plot)
        freq_list = sorted(self.individual_df['frequency'].unique())
        freq_groups = [self.individual_df[self.individual_df['frequency'] == freq]['exposure'].values 
                      for freq in freq_list]
        box_plot = ax3.boxplot(freq_groups, positions=range(1, len(frequencies)+1), 
                              patch_artist=True)
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            patch.set_alpha(0.7)
        ax3.set_title('C) Exposure Distribution by Frequency', fontweight='bold')
        ax3.set_xlabel('Training Frequency')
        ax3.set_ylabel('Exposure')
        ax3.set_xticks(range(1, len(frequencies)+1))
        ax3.set_xticklabels(frequencies)
        ax3.grid(True)
        
        # Panel 4: Count of canaries by frequency
        freq_counts = self.individual_df['frequency'].value_counts().sort_index()
        bars = ax4.bar(freq_counts.index, freq_counts.values, color=self.colors[2], alpha=0.7)
        ax4.set_title('D) Number of Canaries by Frequency', fontweight='bold')
        ax4.set_xlabel('Training Frequency')
        ax4.set_ylabel('Number of Canaries')
        ax4.grid(True)
        
        # Add value labels on bars
        for bar, count in zip(bars, freq_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Exposure Analysis by Training Frequency', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / 'comprehensive_exposure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self):
        """Create a plot comparing canary CER by frequency vs LibriSpeech generalization CER"""
        if self.individual_df is None:
            print("No individual canary data available for performance comparison")
            return
        
        if self.librispeech_df is None:
            print("No LibriSpeech data available for performance comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean CER for each frequency group from canary data
        freq_cer_stats = self.individual_df.groupby('frequency')['CER'].agg(['mean', 'std']).reset_index()
        freq_cer_stats = freq_cer_stats.sort_values('frequency')
        
        frequencies = freq_cer_stats['frequency'].values
        canary_mean_cer = freq_cer_stats['mean'].values
        canary_std_cer = freq_cer_stats['std'].values
        
        # Calculate LibriSpeech generalization performance
        librispeech_mean_cer = self.librispeech_df['cer'].mean()
        librispeech_std_cer = self.librispeech_df['cer'].std()
        
        # Plot canary performance by frequency
        ax.plot(frequencies, canary_mean_cer, 'o-', linewidth=2, markersize=8, 
                color=self.colors[0], label='Canaries by Frequency')
        
        # Add error bars for canary performance
        ax.errorbar(frequencies, canary_mean_cer, 
                   yerr=canary_std_cer, 
                   fmt='none', capsize=5, capthick=2, color=self.colors[0], alpha=0.7)
        
        # Plot LibriSpeech baseline as horizontal line
        ax.axhline(y=librispeech_mean_cer, color=self.colors[1], linestyle='--', 
                  linewidth=2, label=f'LibriSpeech Baseline (CER: {librispeech_mean_cer:.3f})')
        
        # Add shaded region for LibriSpeech std
        ax.axhspan(librispeech_mean_cer - librispeech_std_cer, 
                  librispeech_mean_cer + librispeech_std_cer, 
                  color=self.colors[1], alpha=0.2, label='LibriSpeech ±1 SD')
        
        ax.set_xlabel('Training Frequency', fontweight='bold')
        ax.set_ylabel('Character Error Rate (CER)', fontweight='bold')
        ax.set_title('Canary Performance vs LibriSpeech Generalization', fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True)
        ax.set_xticks(frequencies)
        
        # Add value labels on canary points
        for freq, cer in zip(frequencies, canary_mean_cer):
            ax.annotate(f'{cer:.3f}', (freq, cer), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_exposure_distribution_by_frequency(self):
        """Create box plots showing exposure distribution by frequency"""
        if self.individual_df is None:
            print("No individual data available for distribution plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        frequencies = sorted(self.individual_df['frequency'].unique())
        data_by_freq = [self.individual_df[self.individual_df['frequency'] == freq]['exposure'].values 
                       for freq in frequencies]
        
        # Create box plot
        box_plot = ax.boxplot(data_by_freq, labels=frequencies, patch_artist=True, notch=True)
        
        # Color the boxes
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Training Frequency', fontweight='bold')
        ax.set_ylabel('Exposure', fontweight='bold')
        ax.set_title('Exposure Distribution by Training Frequency', fontweight='bold', pad=20)
        ax.grid(True)
        
        #plt.tight_layout()
        plt.savefig(self.output_dir / 'exposure_distribution_by_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_exposure_vs_cer_by_frequency(self):
        """Create scatter plot showing exposure vs CER, colored by frequency"""
        if self.individual_df is None:
            print("No individual data available for exposure vs CER plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        frequencies = sorted(self.individual_df['frequency'].unique())
        
        for i, freq in enumerate(frequencies):
            freq_data = self.individual_df[self.individual_df['frequency'] == freq]
            color = self.colors[i % len(self.colors)]
            ax.scatter(freq_data['CER'], freq_data['exposure'], 
                      c=color, label=f'Frequency {freq}', alpha=0.7, s=60)
        
        ax.set_xlabel('Character Error Rate (CER)', fontweight='bold')
        ax.set_ylabel('Exposure', fontweight='bold')
        ax.set_title('Exposure vs CER by Training Frequency', fontweight='bold', pad=20)
        ax.legend(title='Training Frequency', title_fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exposure_vs_cer_by_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self):
        """Generate all exposure plots"""
        print("Generating exposure plots...")

        self.plot_mean_exposure_by_frequency()
        self.plot_exposure_distribution_by_frequency()
        self.plot_exposure_vs_cer_by_frequency()
        self.plot_performance_comparison()
        self.plot_comprehensive_summary()

        print(f"All plots saved to: {self.output_dir}")
        

def main():
    parser = argparse.ArgumentParser(description='Generate exposure plots from canary evaluation results')
    parser.add_argument('--config', type=str, required=True, help='Config class name')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Checkpoint directory name')
    parser.add_argument('--evaluation_dir', type=str, default=None,
                        help='Path to directory containing evaluation CSV files (overrides default)')
    parser.add_argument('--plot_type', type=str, choices=['all', 'mean', 'distribution', 'scatter', 'comparison', 'comprehensive'],
                        default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Load config using new system
    config = get_config_class(args.config)
    
    # Set evaluation directory - either custom or checkpoint-specific default
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
    
    plotter = Plotter(evaluation_dir)
    
    if args.plot_type == 'all':
        plotter.generate_all_plots()
    elif args.plot_type == 'mean':
        plotter.plot_mean_exposure_by_frequency()
    elif args.plot_type == 'distribution':
        plotter.plot_exposure_distribution_by_frequency()
    elif args.plot_type == 'scatter':
        plotter.plot_exposure_vs_cer_by_frequency()
    elif args.plot_type == 'comparison':
        plotter.plot_performance_comparison()
    elif args.plot_type == 'comprehensive':
        plotter.plot_comprehensive_summary()


if __name__ == "__main__":
    main()