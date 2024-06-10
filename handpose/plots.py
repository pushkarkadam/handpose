import matplotlib.pyplot as plt
import os


def plot_loss_history(loss_history, root_path = './', save_path_prefix='loss_plot'):
    """
    Plot individual line graphs for each key in the input dictionary.
    Titles indicate the type ('Train' or 'Valid') and the specific loss type.
    Saves each graph as an image file.

    Parameters
    ----------
    loss_history: dict 
        Dictionary containing loss history data.
    root_path: str, default ``'./'``
        Root path to store the plots
    save_path_prefix: str, default ``'loss_plot'``
        Prefix for saving the images.
    
    """
    for loss_type, data in loss_history.items():
        for split_type, losses in data.items():
            plt.plot(losses, marker='o')
            plt.title(f'{split_type.capitalize()}: {loss_type}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.tight_layout()
            save_path = f'{save_path_prefix}_{split_type}_{loss_type}.png'
            train_save_path = os.path.join(root_path, save_path)
            plt.savefig(train_save_path)
            plt.close()


def all_loss_history(loss_history, root_path='./', plot_name='loss_plot.png'):
    """Creates a combined loss plot

    Parameters
    ----------
    loss_history: dict 
        Dictionary containing loss history data.
    root_path: str, default ``'./'``
        Root path to store the plots
    save_path_prefix: str, default ``'loss_plot'``
        Prefix for saving the images.

    """
    # Create subplots
    fig, axs = plt.subplots(2, len(loss_history['train']), figsize=(12, 6))

    # Plot 'train' and 'valid' graphs for each key
    for i, key in enumerate(loss_history['train']):
        axs[0, i].plot(loss_history['train'][key], label='train', marker='o')
        axs[0, i].set_title(f'Train: {key}')
        axs[0, i].set_xlabel('Epoch')
        axs[0, i].set_ylabel('Loss')
        
        axs[1, i].plot(loss_history['valid'][key], label='valid', marker='o')
        axs[1, i].set_title(f'Valid: {key}')
        axs[1, i].set_xlabel('Epoch')
        axs[1, i].set_ylabel('Loss')

    # Adjust layout
    plt.tight_layout()

    # Save plot as an image
    plot_save_path = os.path.join(root_path, plot_name)
    plt.savefig(plot_save_path)