import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

class MasterCSVTrendAnalyser:
    """
    Trend analyser that works with MASTER.csv from across-speed analysis
    """
    
    def __init__(self, output_dir, baseline_csv=None):
        """
        Initialise analyser with output directory containing MASTER.csv
        
        Args:
            output_dir: Directory containing MASTER.csv file
            baseline_csv: Optional path to baseline CSV file
        """
        self.output_dir = Path(output_dir)
        self.master_file = self.output_dir / "across_speed_analysis/MASTER.csv"
        self.plots_dir = self.output_dir / "across_speed_analysis/trend_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load master data
        self.master_df = self.load_master_data()
        if 'exposure' in self.master_df.columns:
            self.master_df['exposure'] = self.master_df['exposure'].clip(lower=0, upper=np.log2(20001))
        self.libri_baseline_csv = pd.read_csv('/zhome/76/b/147012/memorisation/baseline_librispeech.csv') 
        # Load and integrate baseline data if provided
        self.baseline_csv = baseline_csv
        if baseline_csv:
            self.baseline_df = self.add_baseline_to_master(baseline_csv)

    def _normalise_speed(self, speed):
        """Convert speed from string (1x, 1.5x) to numeric (1.0, 1.5)"""
        if isinstance(speed, str) and speed.endswith('x'):
            return float(speed[:-1])
        return float(speed)
    def add_baseline_to_master(self, baseline_csv_path):
        """
        Add baseline data (step 0) to the master DataFrame
        """
        try:
            # Load baseline CSV
            baseline_df = pd.read_csv(baseline_csv_path)
            print(f"Loaded baseline CSV: {len(baseline_df)} rows")
            baseline_df['speed'] = baseline_df['speed'].astype(str).str.replace('x', '', regex=False).astype(float)
            freq_lookup = self.master_df.groupby(['canary_id', 'speed'])['frequency'].first().reset_index()

            # Merge the baseline data with the frequency lookup
            baseline_merged = pd.merge(
                baseline_df,
                freq_lookup,
                on=['canary_id', 'speed'],
                how='left'
            )

            # Add fixed columns to baseline rows
            baseline_merged['checkpoint_step'] = 0
            baseline_merged['speed_folder'] = baseline_merged['speed'].astype(str)
            baseline_merged['checkpoint_folder'] = 'baseline'
            baseline_merged = baseline_merged[baseline_merged['frequency'].notna()]
            # Ensure the baseline DataFrame has the same column order as master_df
            baseline_final = baseline_merged[self.master_df.columns]

            # Combine and return sorted by relevant columns
            combined = pd.concat([self.master_df, baseline_final], ignore_index=True)
            combined = combined.sort_values(by=['canary_id', 'speed', 'checkpoint_step']).reset_index(drop=True)

            return combined

            
        except Exception as e:
            print(f"Error processing baseline CSV: {e}")
            return self.master_df
        
    def load_master_data(self):
        """Load the MASTER.csv file"""
        if not self.master_file.exists():
            raise FileNotFoundError(f"MASTER.csv not found at {self.master_file}")
        
        df = pd.read_csv(self.master_file)
        print(f"Loaded MASTER.csv with {len(df)} rows")
        print(f"Speeds: {sorted(df['speed'].unique())}")
        print(f"Checkpoints: {sorted(df['checkpoint_step'].unique())}")
        return df
    
    def plot_cer_trends_by_speed(self):
        """Plot CER trends for each speed across checkpoints"""
        if 'CER' not in self.master_df.columns:
            print("No CER data available")
            return
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        speeds = sorted(self.master_df['speed'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(speeds)))
        
        for i, speed in enumerate(speeds):
            speed_data = self.master_df[self.master_df['speed'] == speed]
            
            # Calculate mean CER per checkpoint
            cer_by_checkpoint = speed_data.groupby('checkpoint_step')['CER'].mean().reset_index()
            
            # Convert to numpy arrays to avoid pandas indexing issues
            x_vals = cer_by_checkpoint['checkpoint_step'].values
            y_vals = cer_by_checkpoint['CER'].values
            
            ax.plot(x_vals, y_vals,
                   'o-', label=f'Speed {speed}x', linewidth=2.5, markersize=7,
                   color=colors[i])
        
        ax.set_xlabel('Training Steps', fontweight='bold', fontsize=14)
        ax.set_ylabel('Mean Character Error Rate (CER)', fontweight='bold', fontsize=14)
        ax.set_title('CER Trends Across All Speeds', fontweight='bold', fontsize=16, pad=20)
        ax.legend(title='Speed Factor', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cer_trends_all_speeds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: cer_trends_all_speeds.png")
    
    def plot_exposure_trends_by_speed(self):
        """Plot final exposure by frequency for each speed"""
        if 'exposure' not in self.master_df.columns or self.master_df['exposure'].isna().all():
            print("No exposure data available")
            return
            
        # Filter to final checkpoint (100000)
        final_checkpoint = 100000
        final_data = self.master_df[self.master_df['checkpoint_step'] == final_checkpoint]
        
        if final_data.empty:
            print(f"No data found for checkpoint {final_checkpoint}")
            return
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        speeds = sorted(final_data['speed'].unique())
        colors = plt.cm.magma(np.linspace(0.8, 0.2, len(speeds)))
        
        for i, speed in enumerate(speeds):
            speed_data = final_data[final_data['speed'] == speed]
            
            # Calculate mean and std exposure per frequency
            exp_stats = speed_data.groupby('frequency')['exposure'].agg(['mean', 'std']).reset_index()
            
            # Convert to numpy arrays
            x_vals = exp_stats['frequency'].values
            y_vals = exp_stats['mean'].values
            y_err = exp_stats['std'].values
            
            ax.errorbar(x_vals, y_vals, yerr=y_err,
                    fmt='o-', label=f'Speed {speed}x', linewidth=2.5, markersize=7,
                    color=colors[i], capsize=5, capthick=2)
        
        ax.set_xlabel('Training Frequency', fontweight='bold', fontsize=14)
        ax.set_ylabel('Mean Exposure', fontweight='bold', fontsize=14)
        ax.set_title('Final Exposure by Frequency (Checkpoint 100000)', fontweight='bold', fontsize=16, pad=20)
        ax.legend(title='Speed Factor', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(final_data['frequency'].unique()))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'exposure_trends_all_speeds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: exposure_trends_all_speeds.png")
     
    def plot_cer_by_frequency_and_speed(self):
        """Plot CER trends by frequency for each speed"""
        if 'CER' not in self.baseline_df.columns or 'frequency' not in self.baseline_df.columns:
            print("Missing CER or frequency data")
            return
            
        speeds = sorted(self.baseline_df['speed'].unique())
        n_speeds = len(speeds)
        
        # Load LibriSpeech baseline data per speed
        libri_data_by_speed = self._load_librispeech_baseline_by_speed()
        
        # Create subplots for each speed
        fig, axes = plt.subplots(2, (n_speeds + 1) // 2, figsize=(5 * ((n_speeds + 1) // 2), 10))
        if n_speeds == 1:
            axes = [axes]
        elif n_speeds <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        frequencies = sorted(self.baseline_df['frequency'].unique())
        colors = plt.cm.magma(np.linspace(0.8, 0.2, len(frequencies)))
        
        # Calculate global y-axis limits for CER
        all_cer_values = []
        plot_data = {}  # Store data for plotting
        
        for speed in speeds:
            speed_data = self.baseline_df[self.baseline_df['speed'] == speed]
            plot_data[speed] = {}
            
            for freq in frequencies:
                freq_data = speed_data[speed_data['frequency'] == freq]
                if freq_data.empty:
                    continue
                    
                cer_by_checkpoint = freq_data.groupby('checkpoint_step')['CER'].agg(['mean', 'std']).reset_index()
                cer_by_checkpoint.columns = ['checkpoint_step', 'mean_cer', 'std_cer']
                cer_by_checkpoint['std_cer'] = cer_by_checkpoint['std_cer'].fillna(0)  # Handle single data points
                
                plot_data[speed][freq] = {
                    'x': cer_by_checkpoint['checkpoint_step'].values,
                    'y': cer_by_checkpoint['mean_cer'].values,
                    'yerr': cer_by_checkpoint['std_cer'].values
                }
                all_cer_values.extend(cer_by_checkpoint['mean_cer'].values)
        
        # Include LibriSpeech baseline values in y-axis calculation
        for speed in speeds:
            if speed in libri_data_by_speed and not libri_data_by_speed[speed].empty:
                libri_data = libri_data_by_speed[speed]
                all_cer_values.extend(libri_data['mean_cer'].values)
        
        # Set global y-axis limits with some padding
        if all_cer_values:
            y_min = max(0, min(all_cer_values) * 0.95)  # Don't go below 0
            y_max = max(all_cer_values) * 1.05
        else:
            y_min, y_max = 0, 1
        
        # Check if we have baseline data (step 0)
        has_baseline = 0 in self.baseline_df['checkpoint_step'].values
        
        # Plot data with unified y-axis
        for idx, speed in enumerate(speeds):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                continue
            
            # Plot canary frequency data (baseline points look like normal points)
            for i, freq in enumerate(frequencies):
                if freq in plot_data[speed]:
                    data = plot_data[speed][freq]
                    
                    ax.errorbar(data['x'], data['y'], yerr=data['yerr'], 
                        fmt='o-', label=f'Freq {freq}', linewidth=2, 
                        markersize=5, color=colors[i], capsize=3)
            

            if speed in libri_data_by_speed and not libri_data_by_speed[speed].empty:
                libri_data = libri_data_by_speed[speed]
                x_vals = libri_data['checkpoint_step'].values
                mean_vals = libri_data['mean_cer'].values
                std_vals = libri_data['std_cer'].values
                
                baseline_cer = self.libri_baseline_csv['CER'].mean() 
                baseline_std = self.libri_baseline_csv['CER'].std() 
                
                x_extended = np.concatenate([[0], x_vals])
                mean_extended = np.concatenate([[baseline_cer], mean_vals])
                std_extended = np.concatenate([[baseline_std], std_vals])
                
                # Plot single continuous LibriSpeech performance line
                ax.plot(x_extended, mean_extended, color='red', linestyle='--', 
                        linewidth=2, label='LibriSpeech Performance' if idx == 0 else "", alpha=0.8)
                ax.fill_between(x_extended, mean_extended - std_extended, 
                            mean_extended + std_extended,
                            color='red', alpha=0.2)
            

            ax.set_xlim(0,100000)            
            ax.set_title(f'Speed {speed}x', fontweight='bold')
            ax.set_xlabel('Training Steps (x10³)')
            current_ticks = ax.get_xticks()
            ax.set_xticks(current_ticks)
            ax.set_xticklabels([f'{int(tick/1000)}' for tick in current_ticks])
            ax.set_ylabel('CER')
            
            ax.grid(True, alpha=0.3)

            ax.set_ylim(y_min, y_max)
        
        # Hide unused subplots
        for idx in range(n_speeds, len(axes) - 1):
            axes[idx].set_visible(False)

        # Create big legend in the last unused subplot
        if len(axes) > n_speeds:
            legend_ax = axes[-1]  # Use last subplot for legend
            legend_ax.set_visible(True)
            legend_ax.axis('off') 
            
            legend_elements = []
            
            # Add frequency legend entries
            for i, freq in enumerate(frequencies):
                legend_elements.append(
                    plt.Line2D([0], [0], color=colors[i], 
                              marker='o', linestyle='-', 
                              label=f'Frequency {freq}', 
                              markersize=8, linewidth=3))
            
            # Add LibriSpeech performance to legend (single red dashed line with shaded area)
            legend_elements.append(
                plt.Line2D([0], [0], color='red', linestyle='--', 
                          linewidth=3, label='LibriSpeech Performance'))
            
            legend_ax.legend(handles=legend_elements,
                loc='center',
                fontsize=14,
                title='Legend',
                title_fontsize=16,
                frameon=True,
                fancybox=True,
                shadow=True)
            
            legend_ax.set_title('Legend', fontsize=16, fontweight='bold', pad=20)
                
        # Update title based on whether baseline is included
        title = 'CER by Frequency for Each Speed'
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cer_by_frequency_and_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: cer_by_frequency_and_speed.png")
    
    def plot_exposure_heatmap_all_speeds(self):
        """Create heatmap showing exposure across speeds, frequencies, and final checkpoints"""
        if 'exposure' not in self.master_df.columns or self.master_df['exposure'].isna().all():
            print("No exposure data available")
            return
            
        # Get final checkpoint for each speed (latest checkpoint step)
        final_checkpoints = self.master_df.groupby('speed')['checkpoint_step'].max().reset_index()
        final_data = []
        
        for _, row in final_checkpoints.iterrows():
            speed = row['speed']
            final_step = row['checkpoint_step']
            speed_final = self.master_df[
                (self.master_df['speed'] == speed) & 
                (self.master_df['checkpoint_step'] == final_step)
            ]
            final_data.append(speed_final)
        
        if final_data:
            final_df = pd.concat(final_data, ignore_index=True)
            
            # Create pivot table for heatmap
            heatmap_data = final_df.groupby(['speed', 'frequency'])['exposure'].mean().reset_index()
            pivot_data = heatmap_data.pivot(index='frequency', columns='speed', values='exposure')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       cbar_kws={'label': 'Mean Exposure'}, linewidths=0.5,vmin=0,vmax=np.log2(20000))
            
            plt.title('Final Exposure by Speed and Frequency', fontweight='bold', fontsize=16, pad=20)
            plt.xlabel('Speed Factor', fontweight='bold', fontsize=14)
            plt.ylabel('Training Frequency', fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'exposure_heatmap_all_speeds.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: exposure_heatmap_all_speeds.png")
    
    def plot_exposure_by_frequency_and_speed(self):
        """Plot exposure violin plots by frequency for each speed for final checkpoint"""


        if 'exposure' not in self.master_df.columns or self.master_df['exposure'].isna().all():
            print("No exposure data available")
            return
            
        speeds = sorted(self.master_df['speed'].unique())
        n_speeds = len(speeds)
        
        # Get final checkpoint for each speed (latest checkpoint step)
        final_checkpoints = self.master_df.groupby('speed')['checkpoint_step'].max().reset_index()
        
        # Create final dataset with only final checkpoints for each speed
        final_data = []
        for _, row in final_checkpoints.iterrows():
            speed = row['speed']
            final_step = row['checkpoint_step']
            speed_final = self.master_df[
                (self.master_df['speed'] == speed) & 
                (self.master_df['checkpoint_step'] == final_step)
            ]
            final_data.append(speed_final)
        
        if not final_data:
            print("No final checkpoint data found")
            return
            
        final_df = pd.concat(final_data, ignore_index=True)
        print(f"Using final checkpoint data: {len(final_df)} canaries total")
        
        # Create subplots for each speed
        fig, axes = plt.subplots(2, (n_speeds + 1) // 2, figsize=(5 * ((n_speeds + 1) // 2), 10))
        if n_speeds == 1:
            axes = [axes]
        elif n_speeds <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, speed in enumerate(speeds):
            if idx >= len(axes):
                continue
                
            ax = axes[idx]
            speed_data = final_df[final_df['speed'] == speed]
            
            # Group by frequency and get exposure values
            freq_groups = speed_data.groupby('frequency')['exposure'].apply(list)
            
            if not freq_groups.empty:
                exposure_data = freq_groups.values
                parts = ax.violinplot(exposure_data, showmeans=True)
                # Set x-tick labels to frequency values
                freq_labels = freq_groups.index.tolist()
                ax.set_xticks(range(1, len(freq_labels) + 1))
                ax.set_xticklabels(freq_labels)
                for partname in ('cbars','cmins','cmaxes','cmeans'):
                    vp = parts[partname]
                    vp.set_edgecolor(plt.cm.magma(0.8))
                    vp.set_linewidth(1.5 )
                # Color the boxes
                color = plt.cm.magma(0.5)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('white')
                    pc.set_alpha(0.7)

            upper_bound = np.log2(20001)
            # Set global y-axis limits with some padding
            y_min = 0
            y_max = upper_bound + 0.5

            ax.set_title(f'Speed {speed}x', fontweight='bold')
            ax.set_xlabel('Training Frequency')
            ax.set_ylabel('Exposure')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_min, y_max) 

            # # Add exposure upper bound            
            ax.axhline(y=upper_bound, color='darkred', linestyle='--', label='Upper Bound Exposure')
            ax.legend(loc='upper right')
        
        # Hide unused subplots
        for idx in range(n_speeds, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Exposure Distribution by Frequency for Each Speed (Last Checkpoint)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'exposure_by_frequency_and_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: exposure_by_frequency_and_speed.png")

    def plot_cer_with_librispeech_baseline(self):
        """Create scatter plot comparing final CER across speeds and frequencies vs LibriSpeech baseline"""
        if 'CER' not in self.master_df.columns:
            print("No CER data available")
            return
        
        # Use final checkpoint data for comparison
        final_checkpoint = self.master_df['checkpoint_step'].max()
        final_data = self.master_df[self.master_df['checkpoint_step'] == final_checkpoint]
        
        if final_data.empty:
            print(f"No data found for final checkpoint {final_checkpoint}")
            return
        
        # Load LibriSpeech baseline data by speed
        libri_data_by_speed = self._load_librispeech_baseline_by_speed()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        final_data.loc[:, 'CER'] = final_data['CER'].clip(lower=0, upper=1)

        speeds = sorted(final_data['speed'].unique())
        frequencies = sorted(final_data['frequency'].unique())
        
        # Create color map for frequencies (Set3 colormap)
        freq_colors = plt.cm.magma(np.linspace(0.8, 0.2, len(frequencies)))
        freq_color_map = {freq: freq_colors[i] for i, freq in enumerate(frequencies)}
        
        # Set up x-axis positions
        x_positions = {}
        x_labels = []
        x_ticks = []
        
        # Create positions for each speed
        for i, speed in enumerate(speeds):
            x_positions[speed] = i
            x_labels.append(f'Speed {speed}x')
            x_ticks.append(i)
        
        # Add LibriSpeech position
        libri_x_pos = len(speeds)
        x_positions['librispeech'] = libri_x_pos
        x_labels.append('LibriSpeech')
        x_ticks.append(libri_x_pos)
        
        # Plot canary data with jitter
        jitter_strength = 0.1
        for _, row in final_data.iterrows():
            speed = row['speed']
            freq = row['frequency']
            cer = row['CER']
            
            # Add jitter to x-position
            x_pos = x_positions[speed] + np.random.normal(0, jitter_strength)
            
            ax.scatter(x_pos, cer, 
                      c=[freq_color_map[freq]], 
                      s=50, alpha=1.0, edgecolors='none')
        
        # Plot mean and std for each speed category
        for speed in speeds:
            speed_data = final_data[final_data['speed'] == speed]
            if not speed_data.empty:
                mean_cer = speed_data['CER'].mean()
                std_cer = speed_data['CER'].std()
                
                x_pos = x_positions[speed]
                
                # Add error bar first
                ax.errorbar(x_pos, mean_cer, yerr=std_cer, 
                           fmt='none', color='yellowgreen', capsize=5, capthick=2)
                
                # Plot mean with diamond marker on top
                ax.scatter(x_pos, mean_cer, 
                          c='yellowgreen', s=150, alpha=1.0, 
                          edgecolors='yellowgreen', linewidth=2, marker='D',zorder=10)
        
        # Plot LibriSpeech baseline data (aggregate across all speeds)
        all_libri_cer = []
        for speed in speeds:
            if speed in libri_data_by_speed and not libri_data_by_speed[speed].empty:
                libri_data = libri_data_by_speed[speed]
                final_libri_data = libri_data[libri_data['checkpoint_step'] == final_checkpoint]
                
                if not final_libri_data.empty:
                    mean_cer = final_libri_data['mean_cer'].iloc[0]
                    all_libri_cer.append(mean_cer)
        
        # Plot single LibriSpeech baseline if we have data
        if all_libri_cer:
            overall_libri_mean = np.mean(all_libri_cer)
            overall_libri_std = np.std(all_libri_cer)
            
            # Add jitter to LibriSpeech x-position
            x_pos = x_positions['librispeech'] + np.random.normal(0, jitter_strength)
            
            # Add error bar first
            ax.errorbar(x_pos, overall_libri_mean, yerr=overall_libri_std, 
                       fmt='none', color='red', capsize=5, capthick=2)
            
            # Plot mean with diamond marker on top
            ax.scatter(x_pos, overall_libri_mean, 
                      c='red', s=150, alpha=1.0, 
                      edgecolors='red', linewidth=2, marker='D')
        
        # Customise plot
        ax.set_xlabel('Model Type', fontweight='bold', fontsize=14)
        ax.set_ylabel('Character Error Rate (CER)', fontweight='bold', fontsize=14)
        ax.set_title('Final CER Comparison: Canary Frequencies vs LibriSpeech Baseline', 
                    fontweight='bold', fontsize=16, pad=20)
        
        # Set x-axis
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Create legend for frequencies
        freq_legend_elements = []
        for freq in frequencies:
            freq_legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=freq_color_map[freq], 
                          markersize=8, label=f'Frequency {freq}',
                          markeredgecolor='none')
            )
        
        # Add LibriSpeech to legend
        freq_legend_elements.append(
            plt.Line2D([0], [0], marker='D', color='w', 
                      markerfacecolor='red', markersize=10, 
                      label='LibriSpeech Baseline',
                      markeredgecolor='red', markeredgewidth=1)
        )  
        freq_legend_elements.append( 
            plt.Line2D([0], [0], marker='D', color='w', 
                      markerfacecolor='yellowgreen', markersize=10, 
                      label='Speed mean ± 1 std ',
                      markeredgecolor='yellowgreen', markeredgewidth=0.2)
        )
        ax.legend(handles=freq_legend_elements, 
                 title='Training Frequency', 
                 loc='upper left',
                 shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cer_speed_freq_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: cer_speed_freq_combinations.png")

    def plot_cer_distribution_by_speed(self):
        """Create line histogram showing CER distribution for each speed (frequency disregarded)"""
        if 'CER' not in self.master_df.columns:
            print("No CER data available")
            return
        
        # Use final checkpoint data
        final_checkpoint = self.master_df['checkpoint_step'].max()
        final_data = self.master_df[self.master_df['checkpoint_step'] == final_checkpoint]
        
        if final_data.empty:
            print(f"No data found for final checkpoint {final_checkpoint}")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        speeds = sorted(final_data['speed'].unique())
        colors = plt.cm.magma(np.linspace(0.8, 0.2, len(speeds)))

        # Collect all CER values to determine common bin range
        all_cer_values = final_data['CER'].values
        cer_min, cer_max = all_cer_values.min(), all_cer_values.max()
        
        # Create common bins for all speeds
        bins = np.linspace(cer_min * 0.95, cer_max * 1.05, 30)
        
        for i, speed in enumerate(speeds):
            speed_data = final_data[final_data['speed'] == speed]
            cer_values = speed_data['CER'].values
            
            if len(cer_values) == 0:
                continue
                
            # Create histogram data
            counts, bin_edges = np.histogram(cer_values, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
          #  ax.hist(cer_values,bins=bins,density=False,label=f'Speed {speed}x (n={len(cer_values)})')
            # Plot as line
            ax.plot(bin_centers, counts, '-', 
                    label=f'Speed {speed}x (n={len(cer_values)})', 
                    linewidth=2.5, 
                    color=colors[i])
        
        ax.set_xlabel('Character Error Rate (CER)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Density', fontweight='bold', fontsize=14)
        ax.set_title('CER Distribution by Speed (Final Checkpoint)', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.legend(title='Speed Factor', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'cer_distribution_by_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: cer_distribution_by_speed.png")

    def plot_exposure_distribution_by_speed(self):
        """Create line histogram showing exposure distribution for each speed."""

        if 'exposure' not in self.master_df.columns:
            print("No exposure data available")
            return
        
        # Use final checkpoint data
        final_checkpoint = self.master_df['checkpoint_step'].max()
        final_data = self.master_df[self.master_df['checkpoint_step'] == final_checkpoint]
        
        if final_data.empty:
            print(f"No data found for final checkpoint {final_checkpoint}")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        speeds = sorted(final_data['speed'].unique())
        colors = plt.cm.magma(np.linspace(0.8, 0.2, len(speeds)))

        # Collect all exposure values to determine common bin range
        all_exposure_values = final_data['exposure'].values
        exposure_min, exposure_max = all_exposure_values.min(), all_exposure_values.max()
        
        # Create common bins for all speeds
        bins = np.linspace(exposure_min * 0.95, exposure_max * 1.05, 30)
        
        for i, speed in enumerate(speeds):
            speed_data = final_data[final_data['speed'] == speed]
            cer_values = speed_data['exposure'].values
            
            if len(cer_values) == 0:
                continue
                
            # Create histogram data
            counts, bin_edges = np.histogram(cer_values, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot as line
            ax.plot(bin_centers, counts, '-', 
                    label=f'Speed {speed}x (n={len(cer_values)})', 
                    linewidth=2.5, 
                    color=colors[i])
        
        ax.set_xlabel('Exposure', fontweight='bold', fontsize=14)
        ax.set_ylabel('Density', fontweight='bold', fontsize=14)
        ax.set_title('Exposure Distribution by Speed (Final Checkpoint)', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.legend(title='Speed Factor', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'exposure_distribution_by_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: exposure_distribution_by_speed.png")

    def plot_exposure_by_frequency_and_speed_over_time(self):
        """Plot exposure trends by frequency for each speed over time"""
        if 'exposure' not in self.baseline_df.columns or 'frequency' not in self.baseline_df.columns:
            print("Missing exposure or frequency data")
            return
            
        speeds = sorted(self.baseline_df['speed'].unique())
        n_speeds = len(speeds)
        
        # Create subplots for each speed
        fig, axes = plt.subplots(2, (n_speeds + 1) // 2, figsize=(5 * ((n_speeds + 1) // 2), 10))
        if n_speeds == 1:
            axes = [axes]
        elif n_speeds <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        frequencies = sorted(self.baseline_df['frequency'].unique())
        colors = plt.cm.magma(np.linspace(0.8, 0.2, len(frequencies)))
        
        all_exposure_values = []
        plot_data = {} 
        
        for speed in speeds:
            speed_data = self.baseline_df[self.baseline_df['speed'] == speed]
            plot_data[speed] = {}
            
            for freq in frequencies:
                freq_data = speed_data[speed_data['frequency'] == freq]
                if freq_data.empty:
                    continue
                    
                exposure_by_checkpoint = freq_data.groupby('checkpoint_step')['exposure'].agg(['mean', 'std']).reset_index()
                exposure_by_checkpoint.columns = ['checkpoint_step', 'mean_exposure', 'std_exposure']
                exposure_by_checkpoint['std_exposure'] = exposure_by_checkpoint['std_exposure'].fillna(0)  # Handle single data points
                
                plot_data[speed][freq] = {
                    'x': exposure_by_checkpoint['checkpoint_step'].values,
                    'y': exposure_by_checkpoint['mean_exposure'].values,
                    'yerr': exposure_by_checkpoint['std_exposure'].values
                }
                all_exposure_values.extend(exposure_by_checkpoint['mean_exposure'].values)
        
        # Set global y-axis limits with some padding
        upper_bound = np.log2(20001)  # Upper bound for exposure
        if all_exposure_values:
            y_min = min(all_exposure_values) * 0.95
            y_max = upper_bound + 0.5
        else:
            y_min, y_max = 0, upper_bound + 0.5
        
        # Check if we have baseline data (step 0)
        has_baseline = 0 in self.baseline_df['checkpoint_step'].values
        
        # Plot data with unified y-axis
        for idx, speed in enumerate(speeds):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                continue
            
            # Plot exposure frequency data
            for i, freq in enumerate(frequencies):
                if freq in plot_data[speed]:
                    data = plot_data[speed][freq]
                    
                    ax.errorbar(data['x'], data['y'], yerr=data['yerr'], 
                        fmt='o-', label=f'Freq {freq}', linewidth=2, 
                        markersize=5, color=colors[i], capsize=3)
            
            # Add exposure upper bound line
            ax.axhline(y=upper_bound, color='darkred', linestyle='--', 
                    linewidth=2, label='Upper Bound' if idx == 0 else "", alpha=0.8)
            
            # Force x-axis to show step 0 if baseline data exists
            ax.set_xlim(0, 100000)
            
            ax.set_title(f'Speed {speed}x', fontweight='bold')
            # Format x-axis to show values in thousands
            ax.set_xlabel('Training Steps (x10³)')
            current_ticks = ax.get_xticks()
            ax.set_xticks(current_ticks)
            ax.set_xticklabels([f'{int(tick/1000)}' for tick in current_ticks])
            ax.set_ylabel('Avg Exposure')
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_min, y_max)
        
        # Hide unused subplots
        for idx in range(n_speeds, len(axes) - 1):
            axes[idx].set_visible(False)

        # Create big legend in the last unused subplot
        if len(axes) > n_speeds:
            legend_ax = axes[-1]  # Use last subplot for legend
            legend_ax.set_visible(True)
            legend_ax.axis('off') 
            
            legend_elements = []
            
            # Add frequency legend entries
            for i, freq in enumerate(frequencies):
                legend_elements.append(plt.Line2D([0], [0], color=colors[i], 
                                                marker='o', linestyle='-', 
                                                label=f'Frequency {freq}', 
                                                markersize=8, linewidth=3))
            
            # Add upper bound line
            legend_elements.append(plt.Line2D([0], [0], color='darkred', linestyle='--', 
                                linewidth=3, label='Upper Bound Exposure'))
            
            legend_ax.legend(handles=legend_elements,
                labels=[f'Frequency {int(freq)}' for freq in frequencies] + ['Upper Bound Exposure'],
                loc='center',
                fontsize=14,
                title='Legend',
                title_fontsize=16,
                frameon=True,
                fancybox=True,
                shadow=True)
            
            legend_ax.set_title('Legend', fontsize=16, fontweight='bold', pad=20)
                
        title = 'Exposure by Frequency for Each Speed Over Time'
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'exposure_by_frequency_and_speed_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: exposure_by_frequency_and_speed_over_time.png")

    def _load_librispeech_baseline_by_speed(self):
        """Load LibriSpeech evaluation data from all checkpoints, organised by speed"""
        # Get base directory (parent of across_speed_analysis)
        base_dir = self.output_dir
        print(f"Looking for LibriSpeech data in base_dir: {base_dir}")
        
        libri_results_by_speed = {}
        
        # Find all speed result folders
        speed_folders = list(base_dir.glob("*_results"))
        print(f"Found speed folders: {[f.name for f in speed_folders]}")
        
        for speed_folder in speed_folders:
            # Extract speed from folder name
            speed = self.extract_speed_from_folder(speed_folder.name)
            if speed is None:
                continue
                
            eval_dir = speed_folder / "evaluation"
            print(f"Checking eval_dir for speed {speed}: {eval_dir}")
            
            if not eval_dir.exists():
                print(f"Eval dir doesn't exist for speed {speed}")
                continue
                
            # Find all checkpoint evaluation folders
            checkpoint_folders = list(eval_dir.glob("checkpoint-*"))
            speed_libri_results = []
            
            for checkpoint_folder in checkpoint_folders:
                libri_file = checkpoint_folder / "librispeech_evaluation_samples.csv"
                if not libri_file.exists():
                    print("Can't find LibriSpeech csv's")
                    continue
                    
                try:
                    df = pd.read_csv(libri_file)
                    if 'cer' in df.columns:
                        checkpoint_step = int(checkpoint_folder.name.split('-')[1])
                        mean_cer = df['cer'].mean()
                        std_cer = df['cer'].std()
                        
                        speed_libri_results.append({
                            'checkpoint_step': checkpoint_step,
                            'mean_cer': mean_cer,
                            'std_cer': std_cer
                        })
                except Exception as e:
                    print(f"  Error loading {libri_file}: {e}")
            
            if speed_libri_results:
                libri_results_by_speed[speed] = pd.DataFrame(speed_libri_results).sort_values('checkpoint_step')
                print(f"Loaded {len(speed_libri_results)} LibriSpeech evaluations for speed {speed}")
            else:
                print(f"No LibriSpeech evaluation data found for speed {speed}")
        
        return libri_results_by_speed
    
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
        elif "1" in folder_name:
            return 1.0
        elif "A" in folder_name:
            return 2.5
        elif "B" in folder_name:
            return 1
        elif "AB" in folder_name:
            return 2.5
        elif "C" in folder_name:
            return 1
        elif "AC" in folder_name:
            return 2.5
        elif "BC" in folder_name:
            return 1
        elif "ABC" in folder_name:
            return 2.5
        else:
            return None

    def _load_librispeech_baseline(self):
        """Load LibriSpeech evaluation data from all checkpoints (combined across speeds)"""
        # Get base directory (parent of across_speed_analysis)
        base_dir = self.output_dir
        print(f"Looking for LibriSpeech data in base_dir: {base_dir}")
        
        libri_results = []
        
        # Find all speed result folders
        speed_folders = list(base_dir.glob("*_results"))
        print(f"Found speed folders: {[f.name for f in speed_folders]}")
        
        for speed_folder in speed_folders:
            eval_dir = speed_folder / "evaluation"
            print(f"Checking eval_dir: {eval_dir}")
            if not eval_dir.exists():
                print(f"Eval dir doesn't exist")
                continue
                
            # Find all checkpoint evaluation folders
            checkpoint_folders = list(eval_dir.glob("checkpoint-*"))
            
            for checkpoint_folder in checkpoint_folders:
                libri_file = checkpoint_folder / "librispeech_evaluation_samples.csv"
                if not libri_file.exists():
                    print(f"  File doesn't exist")
                    continue
                    
                try:
                    df = pd.read_csv(libri_file)
                    if 'cer' in df.columns:
                        checkpoint_step = int(checkpoint_folder.name.split('-')[1])
                        mean_cer = df['cer'].mean()
                        std_cer = df['cer'].std()
                        
                        libri_results.append({
                            'checkpoint_step': checkpoint_step,
                            'mean_cer': mean_cer,
                            'std_cer': std_cer
                        })
                except Exception as e:
                    print(f"  Error loading {libri_file}: {e}")
        
        if not libri_results:
            print("No LibriSpeech evaluation data found")
            return pd.DataFrame()
        
        return pd.DataFrame(libri_results).sort_values('checkpoint_step')
            
    def generate_all_plots(self):
        """Generate all trend plots from MASTER.csv"""
        print(f"Generating trend plots from MASTER.csv...")
        
        self.plot_cer_by_frequency_and_speed()
        self.plot_exposure_by_frequency_and_speed()
        self.plot_exposure_heatmap_all_speeds()
        self.plot_cer_with_librispeech_baseline()
        self.plot_exposure_trends_by_speed()
        self.plot_cer_distribution_by_speed()
        self.plot_exposure_by_frequency_and_speed_over_time()
        self.plot_exposure_distribution_by_speed()
        print(f"All plots saved to: {self.plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyse trends from MASTER.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing MASTER.csv file')
    parser.add_argument('--baseline_csv', type=str, default="baseline_canaries_random.csv",
                        help='Path to baseline CSV file (optional)')
    parser.add_argument('--plot_type', type=str, 
                        choices=['all', 'heatmap', 'frequency', 'exposure_freq'],
                        default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist")
        return
    
    master_file = Path(args.output_dir) / "across_speed_analysis/MASTER.csv"
    if not master_file.exists():
        print(f"Error: MASTER.csv not found at {master_file}")
        return
    

    # Initialise analyser with optional baseline data
    trend_analyser = MasterCSVTrendAnalyser(args.output_dir, args.baseline_csv)
    
    trend_analyser.generate_all_plots()
    
    print(f"Analysis complete! Plots saved to: {trend_analyser.plots_dir}")



if __name__ == "__main__":
    main()