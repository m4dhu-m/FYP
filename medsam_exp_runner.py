import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import SegformerForSemanticSegmentation

class WeightedDiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.pos_weight = pos_weight  # foreground class weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Foreground and background weights
        w = torch.ones_like(targets)
        w[targets == 1] = self.pos_weight  # weight foreground higher

        intersection = (inputs * targets * w).sum(dim=1)
        union = (inputs * w + targets * w).sum(dim=1)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(config, train_dataloader, val_dataloader, device):
    from run_MedSAM import CombinedLoss  # Your custom loss
    from run_MedSAM import train_one_epoch, evaluate_model, load_medsam_model  # Your existing training/eval functions
    
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    
    model = load_medsam_model("MedSAM/work_dir/MedSAM/medsam_vit_b.pth", device)  # Or config path
    model.to(device)
    #model = build_model(config, base_model).to(device)

    # Freeze encoder if specified
    if config.get("freeze_blocks", True):  # default to True if not specified
        for name, param in model.named_parameters():
            if "mask_decoder" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Encoder frozen; only decoder is trainable.")
    else:
        print("Full model is trainable.")
        
    # Loss function
    if config["loss"] == "CombinedLoss":
        criterion = CombinedLoss().to(device)
    elif config["loss"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif config["loss"] == "WeightedDiceLoss":
        criterion = WeightedDiceLoss(pos_weight=config.get("pos_weight", 1.0)).to(device)
    else:
        raise ValueError("Unsupported loss function")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0) 
    )

    best_dice = 0
    best_iou = 0
    best_loss = float('inf')
    
    trainable_params = count_trainable_params(model)

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        eval_metrics = evaluate_model(model, val_dataloader, device, criterion)

        print(f"Epoch [{epoch + 1}/{config['epochs']}]: "
        f"Train Loss = {train_loss:.4f}, "
        f"Val Dice = {eval_metrics['Dice']:.4f}")
        
        if eval_metrics['Dice'] > best_dice:
            best_dice = eval_metrics['Dice']
            best_iou = eval_metrics['IoU']
            best_acc = eval_metrics['Accuracy']
            best_precision = eval_metrics['Precision']
            best_recall = eval_metrics['Recall']
            best_f1 = eval_metrics['F1']
            best_loss = eval_metrics['Loss']
            torch.save(model.state_dict(), config["save_path"])

    # Return summary for table
    return {
        "Model": config["model_name"].split("/")[-1],
        "Decoder": config["decoder_type"],
        "Activation": config.get("activation"),
        "Dropout": config.get("dropout"),
        "Loss Func": config["loss"],
        "LR": config["lr"],
        "WeightDecay": config.get("weight_decay", 0.0),
        "Posweight": config.get("pos_weight", 1.0),
        "Epochs": config.get("epochs", "N/A"),
        "Encoder FT": "No" if config.get("freeze_blocks", True) else "Yes",
        "Freeze Blocks": config.get("freeze_blocks", []),
        "Params": trainable_params,
        "Acc": round(best_acc, 7),
        "Prec": round(best_precision, 7),
        "Rec": round(best_recall, 7),
        "F1": round(best_f1, 7),
        "mDSC": round(best_dice, 7),
        "mIoU": round(best_iou, 7),
        "Loss": round(best_loss, 4),
    }

    
def run_all_medsam_experiments(configs, train_dataloader, val_dataloader, device, csv_path="medsam_w_dice_05.csv"):
    results = []

    for i, config in enumerate(configs):
        print(f"Running experiment {i+1}/{len(configs)}")
        result = run_experiment(config, train_dataloader, val_dataloader, device)
        result["Exp"] = i + 1
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nAll experiments completed. Results saved to {csv_path}")
    print(df)
