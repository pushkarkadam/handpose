import torch 
import sys
import time 
import os 
from tqdm import tqdm 
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle

sys.path.append('../')
from handpose import *


def train_model(dataloaders,
                dataset_sizes,
                model,
                loss_fn,   
                num_epochs=100,
                S=7,
                B=2,
                nkpt=21,
                nc=2,
                require_kpt_conf=True,
                iou_threshold=0.5,
                lambda_coord=5,
                lambda_noobj=0.5,
                epsilon=1e-6,
                lambda_kpt=0.5,
                lambda_kpt_conf=0.5,
                verbose=True,
                save_model_path='../data/runs',
                train_dir='train',
                optimizer='default',
                learning_rate=0.01,
                lr_momentum=0.9,
                scheduler='default'
               ):
    r"""Training function.
    
    Parameters
    ----------
    dataloaders: dict
        A dictionary of ``train`` and ``valid`` dataloaders.
    dataset_size: dict
        A dictionary of number of images in the dataset.
    model: torch.nn.Module
        A CNN model.
    loss_fn: function
        Loss function from
        :func:`handpose.loss.loss_fn`
    num_epochs: int, default ``100``
        Number of epochs.
    S: int, default ``7``
        Grid size.
    B: int, default ``2``
    nkpt: int, default ``21``
        Number of keypoints.
    nc: int, default ``2``
        Number of classes.
    require_kpt_conf: bool, default ``True``
        Boolean flag to select keypoint confidence.
    iou_threshold: float, default ``0.5``
        IOU threshold.
    lambda_coord: float, default ``5.0``
        A multiplier for coordinates.
    lambda_noobj: float, default ``0.5``
        A multiplier for no object detection.
    epsilon: float, default ``1e-6``
        Epsilon value to avoid ``nan`` output.
    lambda_kpt: float, default ``0.5``
        A multiplier for keypoint.
    lambda_kpt_conf: float, default ``0.5``
        A multipler for keypoint confident.
    verbose: bool, default ``True``
        Prints statements.
    save_model_path: str, default ``'../data/runs'``
        Path to save the model.
    train_dir: str, default``'train'``
        Directory to name for training.
    optimizer: str, default ``'default'``
        Optimizer
        Use from ``torch.optim`` or 
        :func:`handpose.optimizer.Optimizer`
    learning_rate: float, default ``0.01``
        Learning rate for optimizer.
    lr_momentum: float, default ``0.9``
        Learning rate momentum.
    scheduler: str, default ``'default'``
        Scheduler for learning.
        If using default, then
        :func:`handpose.optimizer.Scheduler`
        is used.
        Otherwise, use `torch.optim` scheduler and directly pass
        it as a parameter.

    Returns
    -------
    dict
        A dictionary of trained model and losses.
    
    """

    # Record start time
    since = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if save_model_path:
        timestamp = str(int(since))
        train_dir = train_dir + '_' + timestamp
        train_path = os.path.join(save_model_path, train_dir)

        # Last model path to save model after every epoch
        last_model_path = os.path.join(train_path, 'last.pt')

        # Final model that is saved after the training is complete
        best_model_path = os.path.join(train_path, 'best.pt')

    losses = {"total_loss": [],
              "box_loss": [],
              "conf_loss": [],
              "class_loss": [],
              "kpt_loss": [],
              "kpt_conf_loss": []
             }

    valid_losses = copy.deepcopy(losses)

    all_losses = {'train': losses, 'valid': valid_losses}
    
    eval_metrics = dict()

    
    epochs_mAP = {'train': [], 'valid': []}

    if optimizer == 'default':
        if verbose:
            print(f'Using default SGD optimizer with learning rate={learning_rate}, momentum={lr_momentum}')
        optimizer = Optimizer(model, optim.SGD, **{'lr':learning_rate, 'momentum':lr_momentum}).optimizer
    
    if scheduler == 'default':
        if verbose:
            print('Using default scheduler with step_size=7 and gamma=0.1')
        scheduler = Scheduler(optimizer, lr_scheduler.StepLR ,**{'step_size': 7, 'gamma': 0.1}).scheduler
        
    
    for epoch in tqdm(range(num_epochs), unit='batch', total=num_epochs):
        if verbose:
            print(f'Epoch: {epoch + 1}')
        for phase in ['train', 'valid']:

            # Running losses
            running_losses = dict()
            for k in losses:
                running_losses[k] = 0.0
        
            # Running corrects
            mAP = 0.0
            
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()
    
            for i, train_data in enumerate(dataloaders[phase], start=0):
                images = train_data[0].to(device)
                data = train_data[1]['head']
                image_names = train_data[1]['image_name']
    
                # Setting the target data to device
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device)
                    else:
                        # Adding each tensor of keypoints to the device
                        for k1, v1 in data[k].items():
                            data[k][k1] = v1.to(device)
    
                # Assigning the current batch size
                current_batch_size = images.size(0)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
    
                    # Network head
                    pred_head = network_head(outputs,
                                        require_kpt_conf=require_kpt_conf,
                                        S=S,
                                        B=B,
                                        nkpt=nkpt,
                                        nc=nc
                                       )
                    # Passing through head activation
                    pred_head_act = head_activations(pred_head)
    
                    # Using the best boxes
                    best_pred_head = best_box(pred_head_act, iou_threshold=iou_threshold)

                    for k, v in best_pred_head.items():
                        if isinstance(v, torch.Tensor):
                            best_pred_head[k] = v.to(device)
                        elif isinstance(v, dict):
                            for k1, v1 in best_pred_head[k].items():
                                best_pred_head[k][k1] = v1.to(device)
    
                    phase_losses = loss_fn(data, best_pred_head, lambda_coord, lambda_noobj, epsilon, lambda_kpt, lambda_kpt_conf)

                    loss = phase_losses['total_loss']
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # Extracting the head data
                    truth_data = extract_head(data)
                    truth_data_nms = non_max_suppression(truth_data)
                    pred_data = extract_head(best_pred_head)
                    pred_data_nms = non_max_suppression(pred_data)

                    pred_map = {'conf_score': pred_data['conf_score'],
                                'labels': pred_data['class_idx'],
                                'boxes': pred_data_nms['boxes']
                               }
                    truth_map = {'boxes': truth_data_nms['boxes'],
                                 'labels': truth_data['class_idx']
                                }

                    # Evaluation metric
                    eval_metric = mean_average_precision(pred_map, truth_map, iou_threshold, num_classes=nc)

                for k in running_losses:
                    running_losses[k] += phase_losses[k].item() * current_batch_size

                mAP += eval_metric['mAP'] 

            if phase == 'train':
                scheduler.step()

            epoch_losses = dict()
 
            for k, v in running_losses.items():
                epoch_loss = v / dataset_sizes[phase]
                epoch_losses[k] = epoch_loss
                all_losses[phase][k].append(epoch_loss)

            mAP_epoch = mAP / dataset_sizes[phase]
            epochs_mAP[phase].append(mAP_epoch)

            if verbose:
                print(f'{phase}')
                print('-' * len(str(phase)))
                for k,v in epoch_losses.items():
                    print(f'{k}: {v:.3f}')
                print("\n")

            if save_model_path:
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                    print(f'New directory {train_dir} created at {train_path}')

                torch.save(model.state_dict(), last_model_path)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if save_model_path:
        torch.save(model.state_dict(), best_model_path)
        with open(os.path.join(train_path, 'all_losses.pickle'), 'wb') as handle:
            pickle.dump(all_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_path, 'mAP.pickle'), 'wb') as handle:
            pickle.dump(epochs_mAP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    history = {'model': model,
               'all_losses': all_losses,
               'mAP': epochs_mAP
              }

    plot_history = dict()
    mAP_plot_history = dict()

    for p in ['train', 'valid']:
        plot_history[p] = dict()
        mAP_plot_history[p] = dict()
 
        for k, v in history['all_losses'][p].items():
            plot_history[p][k] = [float(i) for i in v]
            
        for k, v in epochs_mAP.items():
            mAP_plot_history[p]['mAP'] = [float(i) for i in v]

    if save_model_path:
        loss_history = plot_history

        plot_single_history(loss_history, root_path=train_path)

        plot_single_history(mAP_plot_history, root_path=train_path, save_path_prefix='', xlabel='Epoch', ylabel='mAP')

        plot_all_history(loss_history, root_path=train_path)
      
    return history