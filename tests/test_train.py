import torch 
import sys 

sys.path.append('../')
from handpose import *


def test_train_model():
    """Trains the model"""

    REPO = 'pytorch/vision:v0.17.1'

    S = 7
    B = 2
    nkpt = 21
    nc = 2
    batch_size=16
    cell_relative=True
    input_size = (3, 224, 224)
    require_kpt_conf = True
    weights = 'ResNet18_Weights.IMAGENET1K_V1'
    model_name = 'resnet18'
    freeze_weights = False

    data_dir = 'data/dev'

    image_datasets = {x: HandDataset(img_dir=os.path.join(data_dir, x, 'images'), 
                                    label_dir=os.path.join(data_dir, x, 'labels'),
                                    S=S,
                                    nc=nc,
                                    nkpt=nkpt,
                                    cell_relative=cell_relative,
                                    require_kpt_conf=require_kpt_conf
                                    )
                    for x in ['train', 'valid']}
    
    # Dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model output feature
    test_out_features = S * S * (B * (5 + 3 * nkpt) + nc)

    # Creating model
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
    )

    num_epochs = 1
    iou_threshold = 0.5
    lambda_coord = 5
    lambda_noobj = 0.5
    epsilon = 1e-6
    lambda_kpt = 0.5 
    lambda_kpt_conf = 0.5

    history = train_model(dataloaders,
                    dataset_sizes,
                    model,
                    loss_fn,   
                    num_epochs,
                    S,
                    B,
                    nkpt,
                    nc,
                    verbose=False,
                    save_model_path='',
                    train_dir='train',
                    optimizer='default',
                    learning_rate=0.01,
                    lr_momentum=0.9,
                    scheduler='default'
                )
    
    assert(isinstance(history, dict))
    assert(isinstance(history['all_losses'], dict))