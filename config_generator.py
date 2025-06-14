# config_generator.py
from itertools import product

def get_loss_lr_configs(model_name):
    losses = ["CombinedLoss", "BCEWithLogitsLoss", "WeightedDiceLoss"]
    # losses = ["CombinedLoss", "BCEWithLogitsLoss"]
    lrs = [1e-3]
    pos_weights = [0.5, 1.0, 1.5, 2.0]

    configs = []
    exp_id = 1

    for loss in losses:
        for lr in lrs:  
            for pw in pos_weights:
                configs.append({
                    "model_name": model_name,
                    "decoder_type": "original",  # fixed decoder for now
                    "activation": None,
                    "dropout": 0.1,
                    "freeze_blocks": [],
                    "loss": loss,
                    "pos_weight": pw,
                    "epochs": 20,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "save_path": f"checkpoints/exp{exp_id}_{loss}_lr{lr}.pth",
                })
                exp_id += 1

    return configs


def get_full_grid_configs(model_name):
    # decoder_types = ["original", "conv_relu_dropout", "conv_bn_relu", 
    #                  "conv_transpose", "aspp", "se_block", "unet_block", "attention"]
    decoder_types = ["original"]
    activations = [None, "ReLU", "LeakyReLU", "GELU"]
    dropouts = [0.0, 0.1, 0.25, 0.3]
    freeze_options = [[], [".encoder"], [".encoder", ".embeddings"]]
    losses = ["CombinedLoss", "BCEWithLogitsLoss"]
    lrs = [1e-3, 1e-4]
    weight_decays = [0.0, 1e-2, 1e-3, 1e-5, 1e-4]
    epochs = [20, 30, 50]

    configs = []
    exp_id = 1

    for (wd, ep) in product(
        weight_decays, epochs):
        # Skip invalid configs (e.g., activation used with original decoder)
        # if decoder == "original" and act is not None:
        #     continue

        config = {
            "model_name": model_name,
            "decoder_type": "original",
            "activation": None,
            "dropout": 0.1,
            "freeze_blocks": [],
            "loss": "CombinedLoss",
            "epochs": ep,
            "lr": 1e-3,
            "weight_decay": wd,
            #"save_path": f"checkpoints/exp{exp_id}_{decoder}_{act}_drop{dropout}_lr{lr}_wd{wd}_{loss}.pth"
            "save_path": f"checkpoints/exp{exp_id}_{ep}_{wd}.pth"
            
        }   
        configs.append(config)
        exp_id += 1

    return configs
