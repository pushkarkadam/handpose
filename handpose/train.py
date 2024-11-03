import torch 
import sys
import time 
import os 
from tqdm import tqdm 
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import pandas as pd
from pathlib import Path

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
                scheduler='default',
                individual_plots=False,
                losses_types=["total_loss", "box_loss", "conf_loss", "class_loss", "kpt_loss"],
                sample_grid_shape=(2,4),
                **kwargs
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
    individual_plots: bool, default ``False``
        Does not store the individual plots.
    losses_types: list
        Types of losses given as a list to be the keys of the dictionary.

    Returns
    -------
    dict
        A dictionary of trained model and losses.
    
    """

    if kwargs:
        resume_training = kwargs['resume_training']
        loss_df = kwargs['loss_df']
        train_path = kwargs['train_save_path']
        epochs_passed, _ = loss_df['train'].shape

    else:
        resume_training = False
        epochs_passed = 0
        loss_df = {'train': pd.DataFrame(), 'valid': pd.DataFrame()}

    # Record start time
    since = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if save_model_path:
        if not resume_training:
            # Creating training directory
            train_path = create_train_dir(save_model_path)
            print(f"Created a new directory at {train_path}")

        # Last model path to save model after every epoch
        last_model_path = os.path.join(train_path, 'last.pt')

        # Final model that is saved after the training is complete
        best_model_path = os.path.join(train_path, 'best.pt')

    losses = dict()
    for loss_type in losses_types:
        losses[loss_type] = []

    valid_losses = copy.deepcopy(losses)

    all_losses = {'train': losses, 'valid': valid_losses}

    # If resume training is true populate the previous losses in the all_losses dict
    if resume_training:
        for phase in all_losses:
            for c in loss_df[phase].columns:
                all_losses[phase][c] = list(loss_df[phase][c])
    
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
        
    
    for epoch in tqdm(range(epochs_passed, num_epochs), unit='batch', total=num_epochs):
        if verbose:
            print('\n\n')
            print('=' * len(f"Epoch: {epoch + 1}"))
            print(f'Epoch: {epoch + 1}')
            print('=' * len(f"Epoch: {epoch + 1}"))
        
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
    
            for i, train_data in enumerate(tqdm(dataloaders[phase]), start=0):
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
                    mAP_class, prec_class_epoch, recall_class_epoch, F_score_class_epoch  = mean_average_precision(truth_map, pred_map, iou_threshold)

                for k in running_losses:
                    running_losses[k] += phase_losses[k].item() * current_batch_size

                mAP += mAP_class

            if phase == 'train':
                scheduler.step()

            epoch_losses = dict()
 
            for k, v in running_losses.items():
                epoch_loss = v / dataset_sizes[phase]
                epoch_losses[k] = epoch_loss
                all_losses[phase][k].append(epoch_loss)

            # Append all_losses to dataframe here
            loss_df[phase] = loss_df[phase].from_dict(all_losses[phase])

            if save_model_path:
                loss_df[phase].to_csv(os.path.join(train_path, f'{phase}_loss.csv'), index=False)

            mAP_epoch = mAP / dataset_sizes[phase]
            epochs_mAP[phase].append(mAP_epoch)

            if verbose:
                print('\n')
                print(f'{phase}')
                print('-' * len(str(phase)))

                df_metrics = loss_df[phase].iloc[-1:].copy()

                df_metrics.loc[:, f'mAP{int(iou_threshold * 100)}'] = [mAP_epoch]

                print(df_metrics)

            if save_model_path:
                torch.save(model.state_dict(), last_model_path)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if save_model_path:
        torch.save(model.state_dict(), best_model_path)

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

        if individual_plots:
            plot_single_history(loss_history, root_path=train_path)

            plot_single_history(mAP_plot_history, root_path=train_path, save_path_prefix='', xlabel='Epoch', ylabel='mAP')

        # plot_all_history(loss_history, root_path=train_path)

        plot_loss(loss_df, root_path=train_path)

        # Save sample images
        save_sample_images(images, 
                       data, 
                       truth_data_nms, 
                       pred_data, 
                       pred_data_nms,
                       train_path,
                       sample_grid_shape
                      )
      
    return history

def start_training(config, verbose=True):
    r"""Sets up the job using the parameters in the config file.
    
    This is the overall train function that does the job of setting up the dataloders and the computer settings
    to run the training programme.

    The results are stored in the directory mentioned in the config file.
    This function is different from
    :func:`handpose.train.train_model`
    in a way that ``train_model()`` does not setup the job.
    
    Parameters
    ----------
    config: str
        Path to config yaml file. Example: ``'~/path/to/config.yaml'``
    verbose: bool, default ``True``.
        Prints the training output.

    """

    config = load_variables(config)

    try:
        REPO = config['REPO']
        S = config['S']
        B = config['B']
        nkpt = config['nkpt']
        nkpt_dim = config['nkpt_dim']
        nc = config['nc']
        batch_size = config['batch_size']
        cell_relative = config['cell_relative']
        input_size = tuple(config['input_size'])
        require_kpt_conf = config['require_kpt_conf']
        weights = config['weights']
        model_name = config['model_name']
        freeze_weights = config['freeze_weights']
        data_dir = config['data_dir']
        save_model_path = config['save_model_path']
        num_epochs = config['num_epochs']
        iou_threshold = config['iou_threshold']
        lambda_coord = config['lambda_coord']
        lambda_noobj = config['lambda_noobj']
        epsilon = float(config['epsilon'])
        lambda_kpt = config['lambda_kpt']
        lambda_kpt_conf = config['lambda_kpt_conf']
        shuffle_data = config['shuffle_data']
        num_workers = config['num_workers']
        drop_last = config['drop_last']
        optimizer = config['optimizer']
        learning_rate = config['learning_rate']
        lr_momentum = config['lr_momentum']
        scheduler = config['scheduler']
    except Exception as e:
        print(f"{e}")

    # Dataloaders
    dataloaders, dataset_sizes = get_dataloaders(data_dir,
                                             S,
                                             nc,
                                             nkpt,
                                             cell_relative,
                                             require_kpt_conf,
                                             batch_size,
                                             shuffle_data,
                                             num_workers,
                                             drop_last
                                            )

    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        weights=weights,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    ).to(DEVICE)

    print(model.summary())

    history = train_model(dataloaders,
                dataset_sizes,
                model,
                loss_fn,   
                num_epochs=num_epochs,
                S=S,
                B=B,
                nkpt=nkpt,
                nc=nc,
                require_kpt_conf=require_kpt_conf,
                iou_threshold=iou_threshold,
                lambda_coord=lambda_coord,
                lambda_noobj=lambda_noobj,
                epsilon=epsilon,
                lambda_kpt=lambda_kpt,
                lambda_kpt_conf=lambda_kpt_conf,
                verbose=verbose,
                save_model_path=save_model_path,
                optimizer=optimizer,
                learning_rate=learning_rate,
                lr_momentum=lr_momentum,
                scheduler=scheduler
               )

def resume_training(train_dir, config_file, verbose=True):
    r"""Restarts training from the last checkpoint.
    
    This function will continue the training where it was left off before.
    The function uses ``train_loss.csv`` and ``valid_loss.csv`` from the ``train#`` directory.
    It recounts the total number of epochs from ``config.yaml`` file and subtracts the epochs passed.
    If the training has already passed the number of epochs, then update the ``config.yaml`` file increase the number of epochs.

    Parameters
    ----------
    train_dir: str
        Path where the training directory was created.
        This will be inside ``data/runs/`` directory but can be at any other location.
    config_file: str
        Path to the same config file that was used to begin the first instance of training.
        Make sure to not change the values such as box numbers or grid size in the config file
        as it make cause problem with dimension on resuming training.
    verbose: bool, default ``True``
        Prints the training outcome.
    
    """

    # Sanity check - Make sure the required files are present
    
    # Picks up last.pt
    try:
        model_path = os.path.join(train_dir, 'last.pt')
        if not Path(model_path).is_file():
            print(f"{model_path} not found!")
            sys.exit(1)
    except Exception as e:
        print(f'{e}')
    
    # Looks for `train_loss.csv` and `valid_loss.csv`
    loss_df = dict()
    
    try:
        loss_df['train'] = pd.read_csv(os.path.join(train_dir, 'train_loss.csv'))
        loss_df['valid'] = pd.read_csv(os.path.join(train_dir, 'valid_loss.csv'))
    except Exception as e:
        print(f'{e}')

    # Config file 
    config = load_variables(config_file)

    try:
        REPO = config['REPO']
        S = config['S']
        B = config['B']
        nkpt = config['nkpt']
        nkpt_dim = config['nkpt_dim']
        nc = config['nc']
        batch_size = config['batch_size']
        cell_relative = config['cell_relative']
        input_size = tuple(config['input_size'])
        require_kpt_conf = config['require_kpt_conf']
        weights = config['weights']
        model_name = config['model_name']
        freeze_weights = config['freeze_weights']
        data_dir = config['data_dir']
        save_model_path = config['save_model_path']
        num_epochs = config['num_epochs']
        iou_threshold = config['iou_threshold']
        lambda_coord = config['lambda_coord']
        lambda_noobj = config['lambda_noobj']
        epsilon = float(config['epsilon'])
        lambda_kpt = config['lambda_kpt']
        lambda_kpt_conf = config['lambda_kpt_conf']
        shuffle_data = config['shuffle_data']
        num_workers = config['num_workers']
        drop_last = config['drop_last']
        optimizer = config['optimizer']
        learning_rate = config['learning_rate']
        lr_momentum = config['lr_momentum']
        scheduler = config['scheduler']
    except Exception as e:
        print(f"{e}")

    # Dataloaders
    dataloaders, dataset_sizes = get_dataloaders(data_dir,
                                             S,
                                             nc,
                                             nkpt,
                                             cell_relative,
                                             require_kpt_conf,
                                             batch_size,
                                             shuffle_data,
                                             num_workers,
                                             drop_last
                                            )

    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = TransferNetwork(
        repo_or_dir=REPO,
        model_name=model_name,
        weights=weights,
        S=S,
        B=B,
        nkpt=nkpt,
        nc=nc,
        input_size=input_size,
        require_kpt_conf=require_kpt_conf,
        freeze_weights=freeze_weights
    ).to(DEVICE)

    # Reload the saved model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Counts the epochs passed from the epochs given by the rows of 
    epochs_passed, _ = loss_df['train'].shape

    # sanity check on num_epochs
    if num_epochs <=0:
        print("""Cannot perform training on epochs <= 0! 
        Make sure the config file num_epochs is larger value or start a new training by preloading weights.
        """)
        sys.exit(1)

    print(model.summary())

    print(f"""
    \n
    Restarting training from {epochs_passed + 1}.
    Using weights from {model_path}.
    """)

    retraining_args = {'train_save_path': train_dir,
                       'resume_training': True,
                       'loss_df': loss_df
                      }

    history = train_model(dataloaders,
                dataset_sizes,
                model,
                loss_fn,   
                num_epochs=num_epochs,
                S=S,
                B=B,
                nkpt=nkpt,
                nc=nc,
                require_kpt_conf=require_kpt_conf,
                iou_threshold=iou_threshold,
                lambda_coord=lambda_coord,
                lambda_noobj=lambda_noobj,
                epsilon=epsilon,
                lambda_kpt=lambda_kpt,
                lambda_kpt_conf=lambda_kpt_conf,
                verbose=verbose,
                save_model_path=save_model_path,
                optimizer=optimizer,
                learning_rate=learning_rate,
                lr_momentum=lr_momentum,
                scheduler=scheduler,
                **retraining_args
               )