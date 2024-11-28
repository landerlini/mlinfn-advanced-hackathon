import math
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# LABELS = ["unfocus hit", "focus hit"]

def event_id_to_sample_id(events_df, event_id):
    return events_df.query('event_number == @event_id').sample_ID.unique()


def plot_point_cloud(df, x_col='x', y_col='y', z_col='z', 
                               category_col='category', figsize=(12, 8),
                               alpha=0.6, point_size=10, 
                               category_labels=None, 
                               view_angle=None,
                               equal_aspect=True,
                                color_map=None,
                                fig=None, ax=None):
    """
    Create an advanced 3D scatter plot of points with customizable features.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the coordinates and categories
    x_col, y_col, z_col : str
        Column names for x, y, z coordinates
    category_col : str
        Column name for categories
    figsize : tuple
        Figure size as (width, height)
    alpha : float
        Transparency of points (0 to 1)
    point_size : int
        Size of the scatter points
    category_labels : dict
        Dictionary mapping category values to labels
        Example: {-1: 'padding', 0: 'focus track', 1: 'cell'}
    view_angle : tuple
        Initial view angle as (elevation, azimuth)
    equal_aspect : bool
        Whether to force equal aspect ratio for all axes
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated 3D scatter plot
    """
    # Default category labels if none provided
    if category_labels is None:
        category_labels = {cat: f'Category {cat}' 
                         for cat in df[category_col].unique()}
    
    if color_map is None:
        color_map = {
    -1: '#4059AD',  # Royal Blue
    0: '#6B9AC4',   # Sky Blue
    1: '#97D8C4',   # Mint
    2: '#EF8354'    # Coral
}
    
    # Create figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot each category
    for category in sorted(df[category_col].unique()):
        mask = df[category_col] == category
        # ax.scatter(df[x_col][mask], df[y_col][mask], df[z_col][mask],
        #           c=color_map[category],
        #           label=category_labels[category],
        #           alpha=alpha,
        #           s=point_size)
        if category == 0:
            ax.plot(df[x_col][mask], df[y_col][mask], df[z_col][mask],
                    c=color_map[category], marker='s', linestyle='--', 
                    label=category_labels[category], alpha=alpha, markersize=point_size/3)
        else:
            ax.scatter(df[x_col][mask], df[y_col][mask], df[z_col][mask],
                       c=color_map[category], label=category_labels[category],
                       alpha=alpha, s=point_size)
            
    # Set labels
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    
    # Set initial view angle if provided
    if view_angle is not None:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Force equal aspect ratio if requested
    if equal_aspect:
        # Get axis limits
        x_lim = ax.get_xlim3d()
        y_lim = ax.get_ylim3d()
        z_lim = ax.get_zlim3d()
        
        # Calculate ranges
        x_range = abs(x_lim[1] - x_lim[0])
        y_range = abs(y_lim[1] - y_lim[0])
        z_range = abs(z_lim[1] - z_lim[0])
        
        # Get the largest range
        max_range = max(x_range, y_range, z_range)
        
        # Set equal aspect ratio
        ax.set_box_aspect((x_range/max_range, 
                          y_range/max_range, 
                          z_range/max_range))
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()


def plot_histograms(data_dict, figsize=(12, 8), bins='auto', density=False):
    """
    Create multiple histograms arranged in a grid layout.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are variable names and values are the data arrays
        Example: {'var1': data1, 'var2': data2, ...}
    figsize : tuple, optional
        Figure size as (width, height)
    bins : int or str, optional
        Number of bins or method to calculate bins
        Options include: 'auto', 'fd', 'scott', 'rice', 'sturges', 'sqrt'
    density : bool, optional
        If True, plot density instead of counts
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure with all histograms
    """
    n_vars = len(data_dict)
    
    # Calculate number of rows and columns for the grid
    n_cols = min(3, n_vars)  # Max 3 columns
    n_rows = math.ceil(n_vars / n_cols)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Convert axes to 1D array if there's only one row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.ravel()
    
    # Plot each histogram
    for idx, (var_name, data) in enumerate(data_dict.items()):
        # Calculate row and column indices
        row = idx // n_cols
        col = idx % n_cols
        
        # Get current axis
        if n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Create histogram
        ax.hist(data, bins=bins, density=density,
               alpha=0.8, color='royalblue', edgecolor='black')
        
        # Customize the subplot
        ax.set_title(var_name, fontsize=10, pad=10)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Density' if density else 'Frequency', fontsize=8)
        ax.grid(True, axis='both', linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=8)
    
    # Remove empty subplots if any
    if n_vars < (n_rows * n_cols):
        for idx in range(n_vars, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows == 1:
                fig.delaxes(axes[col])
            elif n_cols == 1:
                fig.delaxes(axes[row])
            else:
                fig.delaxes(axes[row, col])
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def get_logbins(arr, n_bins):
    min_val = arr.min()
    max_val = arr.max()
    return np.logspace(np.log10(min_val), np.log10(max_val), n_bins)

def plot_result(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def _visualize_data(point_cloud, labels, label_map, ax=None, class_colors = {0: 'firebrick', 1: 'forestgreen',}
):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection="3d")
    for index, label in enumerate(label_map):
        c_df = df[df["label"] == label]
        try:
            ax.scatter(
                c_df["x"], c_df["y"], c_df["z"], label=label, alpha=0.5, c=class_colors[index]
            )
        except IndexError:
            pass
    ax.legend()
    # plt.show()
    
    
def visualize_prediction(point_clouds, label_clouds, idx, ax=None, label_map=["unfocus hit", "focus hit"], class_colors = {0: 'firebrick', 1: 'forestgreen',}
):
    point_cloud = point_clouds[idx]
    label_cloud = label_clouds[idx]
    _visualize_data(point_cloud, [label_map[np.argmax(label)] for label in label_cloud], label_map, ax)
