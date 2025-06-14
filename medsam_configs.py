# medsam_configs.py
from itertools import product

def get_loss_lr_configs(model_name):
    losses = ["CombinedLoss"]
    # losses = ["CombinedLoss", "BCEWithLogitsLoss"]
    lrs = [1e-3]
    freeze_options = [True]
    # pos_weights = [0.5, 1.0, 1,5, 2.0]
    pos_weights = [0.5]

    configs = []
    exp_id = 1

    for pw in pos_weights:
                 
        configs.append({
            "model_name": model_name,
            "decoder_type": "original",  # fixed decoder for now
            "activation": None,
            "dropout": 0.1,
            "freeze_blocks": True,
            "loss": "WeightedDiceLoss",
            "pos_weight": pw,
            "epochs": 20,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "save_path": f"checkpoints/exp{exp_id}_lr{pw}.pth",
        })
        exp_id += 1

    return configs


def get_full_grid_configs(model_name):
    decoder_types = ["original", "conv_relu_dropout", "conv_bn_relu", 
                     "conv_transpose", "aspp", "se_block", "unet_block", "attention"]
    activations = [None, "ReLU", "LeakyReLU", "GELU"]
    dropouts = [0.0, 0.1, 0.25, 0.3]
    freeze_options = [True, False]
    losses = ["CombinedLoss", "BCEWithLogitsLoss"]
    lrs = [1e-3, 1e-4]
    weight_decays = [0.0, 1e-2, 1e-3, 1e-5, 1e-4]
    epochs = 20

    configs = []
    exp_id = 1

    for (decoder,) in product(
        decoder_types):
        # Skip invalid configs (e.g., activation used with original decoder)
        # if decoder == "original" and act is not None:
        #     continue

        config = {
            "model_name": model_name,
            "decoder_type": decoder,
            "activation": None,
            "dropout": 0.1,
            "freeze_blocks": [],
            "loss": "CombinedLoss",
            "epochs": epochs,
            "lr": 1e-3,
            "weight_decay": 0.0,
            #"save_path": f"checkpoints/exp{exp_id}_{decoder}_{act}_drop{dropout}_lr{lr}_wd{wd}_{loss}.pth"
            "save_path": f"checkpoints/exp{exp_id}_{decoder}.pth"
            
        }   
        configs.append(config)
        exp_id += 1

    return configs
