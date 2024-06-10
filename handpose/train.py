import torch 
import sys
import time 
import os 
from tqdm import tqdm 
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.append('../')
from handpose import *


def train_model(dataloaders,
                dataset_sizes,
                model,
                loss_fn,   
                num_epochs,
                S,
                B,
                nkpt,
                nc,
                require_kpt_conf=True,
                iou_threshold=0.5,
                lambda_coord = 5,
                lambda_noobj = 0.5,
                epsilon = 1e-6,
                lambda_kpt = 0.5,
                lambda_kpt_conf = 0.5,
                verbose=True,
                save_model_path='../data/runs',
                train_dir='train',
                optimizer='default',
                learning_rate=0.01,
                lr_momentum=0.9,
                scheduler='default'
               ):
    """Training function."""

    # Record start time
    since = time.time()

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

    all_losses = {'train': losses, 'valid': losses}
    
    eval_metrics = dict()
    
    running_loss = 0.0

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
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()
    
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
            for i, train_data in enumerate(dataloaders[phase], start=0):
                images = train_data[0].to(device)
                data = train_data[1]['head']
                image_names = train_data[1]['image_name']
    
                # Setting the target data to device
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device)
    
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
    
                    phase_losses = loss_fn(data, best_pred_head, lambda_coord, lambda_noobj, epsilon, lambda_kpt, lambda_kpt_conf)
    
                    loss = phase_losses['total_loss']
                    
                    for k, v in phase_losses.items():
                        all_losses[phase][k].append(v)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                    if phase == 'valid':
                        truth_data = extract_head(data)
                        truth_data_nms = non_max_suppression(truth_data)
                        pred_data = extract_head(best_pred_head)
                        pred_data_nms = non_max_suppression(pred_data)
    
                        pred_map = {'conf_score': pred_data['conf_score'],
                                    'labels': pred_data['class_idx'],
                                    'boxes': pred_data_nms['boxes']
                                   }
                        truth_map = {'boxes': pred_data_nms['boxes'],
                                     'labels': truth_data['class_idx']
                                    }
                        eval_metric = mean_average_precision(pred_map, truth_map, iou_threshold, num_classes=nc)
                        if verbose:
                            print(f'mAP: {eval_metric["mAP"]}')
                        eval_metrics[epoch] = eval_metric
    
                running_loss += loss.item() * images.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            
            if verbose:
                print(f'{phase}')
                print('-' * len(str(phase)))
                for k,v in phase_losses.items():
                    print(f'{k}: {v:.3f}')
                    all_losses[phase][k].append(float(v))
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

    history = {'model': model,
               'all_losses': all_losses
              }

    return history