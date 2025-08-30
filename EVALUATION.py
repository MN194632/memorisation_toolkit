import os
import argparse
import torch
from tracked_canary_evaluator import TrackedCanaryEvaluator
import argparse
from utils import get_config_class
from plot_evaluate import Plotter
import matplotlib.pyplot as plt
from test_generalisation import LibriSpeechModelTester
# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

def evaluate(config,checkpoint_dir,hold_out_folder, speed, max_samples,plots =False):

    # get evaluation_dir from checkpoint_dir 
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
    evaluation_dir = os.path.join(config.output_dir, "evaluation", checkpoint_name)
    print(f"Using evaluation directory: {evaluation_dir}")

    # Create evaluator
    evaluator = TrackedCanaryEvaluator(
        config,checkpoint_dir
    )
    
    # Run evaluation
    print("\nEvaluating fine-tuned model on tracked canaries...")
    results_df = evaluator.evaluate(config,
        batch_size=1,
    )
    
    # Run exposure analysis
    print("\nCalculating exposure metrics...")
    _ = evaluator.analyse_exposure(
        config,
        results_df,
        hold_out_folder, 
        speed,
        max_samples
        )

    #create librispeech tester 
    print("\nEvaluating performance on Librispeech test for generalisation...")
    libri_tester = LibriSpeechModelTester(config, checkpoint_dir, evaluation_dir)
    _ = libri_tester.evaluate_model( max_samples)
    if plots:
        #create plotter
        plotter = Plotter(evaluation_dir)
        plotter.generate_all_plots()
        print(f"\nPlots generated and saved in:{evaluation_dir}/plots")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config class name')
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                       help='Path to model checkpoint(s) directory')
    parser.add_argument("--speed", type=str, default=None,
                        help="Speed factor to process (1.5, 2, 2.5, 3, 3.5, 4, or 'normal')")
    parser.add_argument("--max_samples", type=int, default=20_000,
                        help="Maximum number of hold-out samples to process (useful for testing)")
    parser.add_argument("--hold_out_folder",
                        help="Path to folder with hold_out samples")
    parser.add_argument("--plots", default=False,
                        help="Whether to plot a suit of plots showing the resutls")
    args = parser.parse_args()
    
    config = get_config_class(args.config)
    evaluate(config,args.checkpoint_dir,args.hold_out_folder,args.speed,args.max_samples)