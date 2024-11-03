import matplotlib.pyplot as plt
import os


def plot_single_history(history, root_path = './', save_path_prefix='loss_plot', xlabel='Epoch', ylabel='Loss'):
    """
    Plot individual line graphs for each key in the input dictionary.
    Titles indicate the type ('Train' or 'Valid') and the specific loss type.
    Saves each graph as an image file.

    Parameters
    ----------
    history: dict 
        Dictionary containing history data.
    root_path: str, default ``'./'``
        Root path to store the plots
    save_path_prefix: str, default ``'loss_plot'``
        Prefix for saving the images.
    xlabel: str, default ``'Epoch'``
        X axis label.
    ylabel: str, default ``'Loss'``
        Y axis label.

    Examples
    --------
    >>> mAP_history = {'train': {'mAP': [1.0, 0.5]}, 'valid': {'mAP': [1.0, 0.5]}}
    >>> plot_single_history(history=mAP_history, root_path='./', save_path_prefix='', xlabel='Epoch', ylabel='mAP')
    
    """
    for loss_type, data in history.items():
        for split_type, losses in data.items():
            plt.plot(losses, marker='o')
            plt.title(f'{split_type.capitalize()}: {loss_type}')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            save_path = f'{save_path_prefix}_{split_type}_{loss_type}.png'
            train_save_path = os.path.join(root_path, save_path)
            plt.savefig(train_save_path)
            plt.close()


def plot_all_history(history, root_path='./', plot_name='loss_plot.png', xlabel='Epoch', ylabel='Loss'):
    """Creates a combined history plot

    This can be used to view all the loss or mAP plots in the same chart.

    Parameters
    ----------
    history: dict 
        Dictionary containing history data.
    root_path: str, default ``'./'``
        Root path to store the plots
    plot_name: str, default ``'loss_plot.png'``
        Name of the plot used for saving.
    xlabel: str, default ``'Epoch'``
        X axis label.
    ylabel: str, default ``'Loss'``
        Y axis label.

    Examples
    --------
    >>> loss_history = {'train': {'total_loss': [0.8,0.9], 'box_loss': [0.8, 0.9]}, 'valid': {'total_loss': [0.8,0.9], 'box_loss': [0.8, 0.9]}}
    >>> plot_all_history(history=loss_history, root_path='./', plot_name='loss_plot.png', xlabel='Epoch', ylabel='Loss')

    """
    # Create subplots
    fig, axs = plt.subplots(2, len(history['train']), figsize=(12, 6))

    # Plot 'train' and 'valid' graphs for each key
    for i, key in enumerate(history['train']):
        axs[0, i].plot(history['train'][key], label='train', marker='o')
        axs[0, i].set_title(f'Train: {key}')
        axs[0, i].set_xlabel(xlabel)
        axs[0, i].set_ylabel(ylabel)
        
        axs[1, i].plot(history['valid'][key], label='valid', marker='o')
        axs[1, i].set_title(f'Valid: {key}')
        axs[1, i].set_xlabel(xlabel)
        axs[1, i].set_ylabel(ylabel)

    # Adjust layout
    plt.tight_layout()

    # Save plot as an image
    plot_save_path = os.path.join(root_path, plot_name)
    plt.savefig(plot_save_path)
    plt.close()

def plot_loss(loss_df, root_path='./', plot_name='loss_plot.png', xlabel='Epoch', ylabel='Loss'):
    """Plots graph from the dataframe"""

    # create subplots
    fig, axes = plt.subplots(2, len(loss_df['train'].columns), figsize=(12, 6))

    # plot 'train' and 'valid' graphs
    for idx, phase in enumerate(loss_df):
        plot_df = loss_df[phase]
        for i, col in enumerate(plot_df.columns):
            axes[idx, i].plot(list(loss_df[phase][col]), label=phase, marker='o')
            axes[idx, i].set_title(f'{phase}: {col}')
            axes[idx, i].set_xlabel(xlabel)
            axes[idx, i].set_ylabel(ylabel)
            axes[idx, i].grid()

    # Adjust layout
    plt.tight_layout()

    # Save plot as an iamge
    plot_save_path = os.path.join(root_path, plot_name)
    plt.savefig(plot_save_path)
    plt.close()