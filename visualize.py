from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm

def create_custom_colormap():
    """Create a custom colormap for sentiment visualization"""
    colors = ['#ff4444', '#ffad4d', '#4dff4d', '#4dffff']  # Red to Cyan
    return LinearSegmentedColormap.from_list('custom', colors)

def init_plot(figsize=(12, 8), style='dark_background'):
    """Initialize the plot with custom styling
    
    Args:
        figsize: Tuple of (width, height) in inches
        style: Matplotlib style to use
    """
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color
    fig.patch.set_facecolor('#1C1C1C')
    ax.set_facecolor('#1C1C1C')
    
    # Remove grid
    ax.grid(False)
    
    return fig, ax

def create_legend(ax, labels, label_names=None, colormap=None):
    """Create a custom legend for the plot
    
    Args:
        ax: Matplotlib axis
        labels: Array of label values
        label_names: Optional mapping of label values to names
        colormap: Optional custom colormap
    """
    unique_labels = sorted(set(labels))
    if colormap is None:
        colormap = create_custom_colormap()
    
    norm = Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
    
    patches = []
    for label in unique_labels:
        color = colormap(norm(label))
        name = label_names[label] if label_names else f"Class {label}"
        patches.append(mpatches.Patch(color=color, label=name))
    
    ax.legend(handles=patches, loc='upper right', 
             bbox_to_anchor=(1.15, 1), frameon=True,
             facecolor='#2C2C2C', edgecolor='white')

def create_tooltips(ax, x, y, texts, labels):
    """Create tooltips for data points
    
    Args:
        ax: Matplotlib axis
        x, y: Coordinates
        texts: Array of tooltip texts
        labels: Array of point labels
    """
    from matplotlib.offsetbox import AnnotationBbox, TextArea
    
    def hover(event):
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                # Remove existing tooltips
                for child in ax.get_children():
                    if isinstance(child, AnnotationBbox):
                        child.remove()
                
                # Add tooltip for closest point
                idx = ind["ind"][0]
                tooltip = TextArea(texts[idx][:100] + "...", 
                                textprops=dict(color='white', size=8))
                ab = AnnotationBbox(tooltip, (x[idx], y[idx]),
                                  xybox=(20, 20), xycoords='data',
                                  boxcoords="offset points",
                                  box_alignment=(0., 0.),
                                  bboxprops=dict(fc='#2C2C2C', ec='white'))
                ax.add_artist(ab)
                plt.draw()
    
    return hover

def savegif(Y_seq, labels, texts=None, title="t-SNE Animation", 
            filename="tsne.gif", limits=None, label_names=None,
            figsize=(12, 8), fps=30, dpi=100, style='dark_background'):
    """Create and save an enhanced t-SNE animation
    
    Args:
        Y_seq: Sequence of t-SNE coordinates
        labels: Array of labels for coloring points
        texts: Optional array of texts for tooltips
        title: Title for the animation
        filename: Output filename
        limits: Optional fixed axis limits
        label_names: Optional mapping of label values to names
        figsize: Figure size in inches
        fps: Frames per second
        dpi: Dots per inch
        style: Matplotlib style to use
    """
    fig, ax = init_plot(figsize, style)
    
    # Create custom colormap
    colormap = create_custom_colormap()
    norm = Normalize(vmin=min(labels), vmax=max(labels))
    
    # Create legend
    create_legend(ax, labels, label_names, colormap)
    
    def update(i):
        if (i+1) % 50 == 0:
            print(f'[{i+1} / {len(Y_seq)}] Animating frames')
            
        ax.clear()
        if limits is not None:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        
        # Plot points with custom colors
        colors = [colormap(norm(label)) for label in labels]
        scatter = ax.scatter(Y_seq[i][:, 0], Y_seq[i][:, 1], 
                           c=colors, s=5, alpha=0.6)
        
        # Add tooltips if texts are provided
        if texts is not None:
            fig.canvas.mpl_connect('motion_notify_event', 
                create_tooltips(ax, Y_seq[i][:, 0], Y_seq[i][:, 1], 
                              texts, labels))
        
        # Style the plot
        ax.set_title(f"{title} (frame {i})", color='white', pad=20)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Recreate legend
        create_legend(ax, labels, label_names, colormap)
        
        return scatter,
    
    # Create animation
    print('[*] Creating animation...')
    anim = FuncAnimation(fig, update, frames=len(Y_seq), interval=1000/fps)
    
    # Save animation
    print(f'[*] Saving animation to {filename}')
    anim.save(filename, writer='imagemagick', fps=fps, dpi=dpi)
    plt.close()