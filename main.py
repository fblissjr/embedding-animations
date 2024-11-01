import os
import pickle
import argparse
from sklearn.manifold import TSNE
from dataset_loader import load_dataset
from tsne_hack import extract_sequence
from visualize import savegif

def main(args):
    # Create results and figures directories if they don't exist
    for dir_name in ['results', 'figures']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
    
    # Process embedding fields if specified
    embed_fields = args.embed.split(',') if args.embed else None
    if embed_fields:
        embed_fields = [field.strip() for field in embed_fields]
        print(f"\nWill embed fields: {embed_fields}")

    # Load and process the dataset
    print(f"\nProcessing dataset: {args.dataset}")
    X, labels, label_names, tooltip_texts = load_dataset(
        args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        embed_fields=embed_fields,
        embed_model=args.embed_model,
        embed_task=args.embed_task,
        label_field=args.label_field
    )
    print(f"\nProcessed data shape: {X.shape}")
    
    # Initialize and run t-SNE
    print("\nInitializing t-SNE...")
    tsne = TSNE(
        n_iter=args.num_iters,
        verbose=True,
        n_jobs=-1  # Use all available cores
    )
    tsne._EXPLORATION_N_ITER = args.early_iters
    
    print("\nExtracting t-SNE sequence...")
    Y_seq = extract_sequence(tsne, X)
    
    # Save results
    results_path = os.path.join('results', f'{args.dataset.replace("/", "_")}_res.pkl')
    print(f"\nSaving results to {results_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(Y_seq, f)
    
    # Calculate plot limits
    lo = Y_seq.min(axis=0).min(axis=0).max()
    hi = Y_seq.max(axis=0).max(axis=0).min()
    limits = ([lo, hi], [lo, hi])
    
    # Generate and save animation
    dataset_name = args.dataset.split('/')[-1]
    title = f"{dataset_name} t-SNE Visualization"
    if args.label_field:
        title += f" (colored by {args.label_field})"
    
    fig_name = f'{dataset_name}-{args.num_iters}-{args.early_iters}-tsne'
    fig_path = os.path.join('figures', f'{fig_name}.gif')
    
    print(f"\nSaving animation to {fig_path}")
    savegif(
        Y_seq,
        labels,
        texts=tooltip_texts,
        title=title,
        filename=fig_path,
        limits=limits,
        label_names=label_names,
        figsize=args.figsize,
        fps=args.fps,
        dpi=args.dpi,
        style=args.style
    )
    print("\nDone!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run t-SNE visualization on various datasets')
    parser.add_argument('--dataset', default='mnist70k',
                      help='Dataset name (e.g., mnist70k or HuggingFace dataset path)')
    parser.add_argument('--split', type=str, default=None,
                      help='Dataset split to use (e.g., train_sft, test_sft)')
    parser.add_argument('--num_iters', type=int, default=1000,
                      help='Total number of iterations')
    parser.add_argument('--early_iters', type=int, default=250,
                      help='Number of early exaggeration iterations')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use from the dataset')
    parser.add_argument('--embed', type=str, required=True,
                      help='Comma-separated list of fields to embed (e.g., "prompt,messages")')
    parser.add_argument('--embed-model', type=str, required=True,
                      help='Name of the Hugging Face embedding model to use')
    parser.add_argument('--embed-task', type=str,
                      help='Task type for models supporting multiple tasks')
    parser.add_argument('--label-field', type=str,
                      help='Field to use for coloring points in the visualization')
    
    # Visualization parameters
    parser.add_argument('--figsize', type=int, nargs=2, default=(12, 8),
                      help='Figure size in inches (width height)')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second in the animation')
    parser.add_argument('--dpi', type=int, default=100,
                      help='DPI for the animation')
    parser.add_argument('--style', type=str, default='dark_background',
                      choices=['dark_background', 'default'],
                      help='Visual style for the animation')
    
    args = parser.parse_args()
    main(args)