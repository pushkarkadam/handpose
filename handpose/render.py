import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def render_pose(images, head, is_relative=True, show_keypoint_label=True, classes={0: "Right", 1:"Left"}, box_color={0: "red", 1:"orange"}, text_color={0: "white", 1:"white"}):
    """Renders the image from the image head.

    Parameters
    ----------
    images: torch.Tensor
        Images of the size ``(m, 3, H, W)``
    head: dict
        The dictionary of ground truth.
    is_relative: bool, default ``True``
        Checks if the center of bounding box coordinates are relative to the cell.
    show_keypoint_label: bool, default ``True``
        Shows the labels on rendered image.
    classes: dict, default ``{0: "Right", 1:"Left"}``
        The classes in the dataset and their labels.
    box_color: dict, default ``{0: "red", 1:"orange"}``
        Color of the bounding box.
    text_color: str, default ``{0: "white", 1:"white"}``
        Color of the class text over bounding box.

    Returns
    -------
    list
        A list of rendered images.

    Examples
    --------
    >>> rendered_images = render_pose(train_features, head, is_relative=True, show_keypoint_label=True)
        
    """
    # Edges graph to connect keypoints
    EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    # List to store rendered images
    rendered_images = []  
    
    for idx, image in enumerate(images):
        # Image dimension
        _, H, W = image.shape
        
        # Converting torch tensor image to numpy format
        image = image.permute(1, 2, 0)

        # Objects
        objs = head['obj_indices'][idx].tolist()

        _, _, S = head['conf'][idx].shape

        # Number of keypoints
        nkpt = int(len(list(head['kpt'].keys())) / 2)

        bounding_boxes = []
        kpts = []
        edges = []
        obj_classes = []

        fig, ax = plt.subplots(figsize=(W / 100, H / 100))
        ax.axis('off')
        ax.imshow(image)
        
        for obj in objs:
            i, j = obj
            
            x_cell = float(head['x'][idx][:,i, j])
            y_cell = float(head['y'][idx][:, i, j])

            # Converting from cell relative to image relative
            if is_relative:
                x = (x_cell + j) / S
                y = (y_cell + i) / S
            else:
                x = x_cell
                y = y_cell

            # Scaling the coordinates
            x = x * W
            y = y * H
            
            w = float(head['w'][idx][:,i, j]) * W
            h = float(head['h'][idx][:, i, j]) * H
        
            x_min = x - w / 2
            y_min = y - h / 2
        
            obj_class = int(torch.argmax(head['classes'][idx][:,i,j]))
            obj_classes.append(obj_class)
            
            rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor=box_color[obj_class], facecolor='none')

            bounding_boxes.append(rect)

            # Add text annotation for class name
            class_name = classes[obj_class]
            ax.text(x_min, y_min, class_name, color=text_color[obj_class], fontsize=10, ha='left', va='bottom', backgroundcolor=box_color[obj_class])

            # keypoints
            kpts_list = []
            kpt = head['kpt']
            for k in range(nkpt):
                kx = float(kpt[f'kx_{k}'][idx][:, i,j]) * W
                ky = float(kpt[f'ky_{k}'][idx][:,i,j]) * H
                kpts_list.append((kx, ky))

            kpts.append(kpts_list)

            edges_list = []

            for e in EDGES:
                start_idx, end_idx = e
                start_point = kpts_list[start_idx]
                end_point = kpts_list[end_idx]
                edges_list.append([[start_point[0], end_point[0]], [start_point[1], end_point[1]]])

            edges.append(edges_list)

        for bounding_box in bounding_boxes:
            ax.add_patch(rect)

        for keypoints in kpts:
            for k_idx, (kx, ky) in enumerate(keypoints):
                ax.plot(kx, ky, 'ro')
                if show_keypoint_label:
                    ax.text(kx, ky, str(k_idx + 1), color='blue', fontsize=10, ha='center', va='center')

        for edge in edges:
            for e in edge:
                start, end = e
                ax.plot(start, end, 'w-')

        plt.tight_layout(pad=0)
        # Convert the plot to an image
        fig.canvas.draw()
        rendered_image = np.array(fig.canvas.renderer._renderer)
        rendered_images.append(rendered_image)
        plt.close(fig)

    return rendered_images

def display_images_in_grid(rendered_images, grid_shape=None, save_path=None):
    """Displays rendered images in grid.

    Parameters
    ----------
    rendered_images: list
        A list of rendered images.
    grid_shape: tuple, default ``None``
        The grid to display the images.
    save_path: str, default ``None``
        The path where the figure must be saved.

    Examples
    --------
    >>> display_images_in_grid(rendered_images, grid_shape=None)

    """
    if grid_shape is None:
        num_images = len(rendered_images)
        num_cols = int(num_images ** 0.5)
        num_rows = (num_images + num_cols - 1) // num_cols
    else:
        num_rows, num_cols = grid_shape

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < len(rendered_images):
            ax.imshow(rendered_images[i])
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout(pad=0)

    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()