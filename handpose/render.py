import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os


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
        # Ensuring the image is copied to cpu if it is on gpu
        image = image.cpu()

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
        rendered_image = np.array(fig.canvas.renderer._renderer)[:,:,:3]
        rendered_images.append(rendered_image)
        plt.close(fig)

    return rendered_images

def display_images_in_grid(rendered_images, grid_shape=None, save_path=None, figsize=(12,6), show_image=False):
    """Displays rendered images in grid.

    Parameters
    ----------
    rendered_images: list
        A list of rendered images.
    grid_shape: tuple, default ``None``
        The grid to display the images.

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

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < len(rendered_images):
            ax.imshow(rendered_images[i])
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    if show_image:
        plt.show()

def render_detection(images, 
                     head, 
                     head_nms, 
                     conf_threshold, 
                     show_grid=False, 
                     classes={0: "right", 1:"left"}, 
                     box_color={0: "red", 1:"orange"}, 
                     text_color={0: "white", 1:"white"}):
    r"""Returns a rendered image

    head --> [batch -> [image objects]]

    Parameters
    ----------
    images: torch.Tensor
        Images of size ``(m, 3, H, W)``
    head: dict
        A dictionary of the predicted head.
    head_nms: dict
        A dictionary of non-max suppression
    conf_threshold: float
        Confidence threshold for rendering.
    show_grid: bool, default `False`
        Shows the grid of the prediction size.
    classes: dict, default ``{0: "right", 1:"left"}``
        The classes in the dataset and their labels.
    box_color: dict, default ``{0: "red", 1:"orange"}``
        Color of the bounding box.
    text_color: dict, default ``{0: "white", 1:"white"}``
        Color of the text on top of the bounding box.
        
    """

    # Edges connecting different keypoints
    EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    # List to store rendered images
    rendered_images = []

    for idx, image in enumerate(images):

        # Converting image from gpu to cpu if required.
        image = image.cpu()
    
        # Transforming the image from (ch, H, W) -> (H, W, ch)
        
        image = image.permute(1, 2, 0)
    
        W, H, _ = image.shape
    
        # counting prediction object per images
        for k, v in head.items():
            pred_count = len(v[0])
            break
    
        # rendering the detected objects
        for i in range(pred_count):
            if head['conf_score'][idx][i] > conf_threshold:
                x = head['x'][idx][i] * W
                y = head['y'][idx][i] * H
                w = head['w'][idx][i] * W
                h = head['h'][idx][i] * H
    
                kx = (torch.Tensor(head['kx'][idx][i]) * torch.tensor(W)).tolist()
                ky = (torch.Tensor(head['ky'][idx][i]) * torch.tensor(H)).tolist()
    
                class_index = int(head['class_idx'][idx][i])
                class_name = classes[class_index]
                conf_score = float(head['conf_score'][idx][i])
                
                kpts_list = [(kxn, kyn) for kxn, kyn in zip(kx, ky)]
    
                edges_list = []
    
                for e in EDGES:
                    start_idx, end_idx = e
                    start_point = kpts_list[start_idx]
                    end_point = kpts_list[end_idx]
                    edges_list.append([[start_point[0], end_point[0]], [start_point[1], end_point[1]]])
    
                xmin, ymin, xmax, ymax = (head_nms['boxes'][idx] * torch.Tensor([W, H, W, H])).tolist()[i]
    
                fig, ax = plt.subplots(figsize=(W / 100, H / 100))

                ax.axis('off')
                ax.imshow(image)
    
                rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=2, edgecolor=box_color[class_index], facecolor='none')
    
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{class_name} [{conf_score:.2f}]", color=text_color[class_index], fontsize=10, ha='left', va='bottom', backgroundcolor=box_color[class_index])
    
                for k_idx, (kx, ky) in enumerate(kpts_list):
                    ax.plot(kx, ky, 'ro')
                    ax.text(kx, ky, str(k_idx + 1), color='blue', fontsize=10, ha='center', va='center')
    
                for e in edges_list:
                    start, end = e
                    ax.plot(start, end, 'w-')
    
                if show_grid:
                    x_g = 0
                    while x_g < (int(H)):
                        plt.axvline(x=x_g, color="white", linewidth=0.2)
                        plt.axhline(y=x_g, color="white", linewidth=0.2)
                        x_g = x_g + int(H/S)

                plt.tight_layout(pad=0)
                # Convert the plot to an image
                fig.canvas.draw()
                rendered_image = np.array(fig.canvas.renderer._renderer)[:,:,:3]
                rendered_images.append(rendered_image)
                plt.close(fig)

    return rendered_images

def save_sample_images(images, 
                       truth_head, 
                       truth_data_nms, 
                       pred_data, 
                       pred_data_nms,
                       train_path,
                       sample_grid_shape
                      ):
    """Saves the sample images in the training directory"""

    truth_rendered_images = render_pose(images, 
                                        truth_head, 
                                        is_relative=True,
                                        show_keypoint_label=True,
                                        classes={0: "Right", 1: "Left"}, 
                                        box_color={0: "red", 1: "orange"}
                                       )

    display_images_in_grid(truth_rendered_images, grid_shape=sample_grid_shape, save_path=os.path.join(train_path, "truth_sample.png"))

    pred_rendered_images = render_detection(images, 
                                            pred_data, 
                                            pred_data_nms,
                                            conf_threshold=0.5, 
                                            show_grid=False, 
                                            classes={0: "Right", 1: "Left"}, 
                                            box_color={0: "red", 1: "orange"}
                                           )
    display_images_in_grid(pred_rendered_images, grid_shape=sample_grid_shape, save_path=os.path.join(train_path, "pred_sample.png"))