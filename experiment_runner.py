import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import SegformerForSemanticSegmentation

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=64):
        super(ASPP, self).__init__()

        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_1x1_output = nn.Sequential(
            nn.Conv2d(inter_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)

        # Global context branch
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate all features
        x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.conv_1x1_output(x_cat)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.global_avg_pool(x).view(b, c)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        return x * excitation.expand_as(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.up(x)
        x = self.conv2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn

        return x


def build_decoder(in_channels, out_channels, decoder_type="conv_relu_dropout",
                  activation="ReLU", dropout=0.1):
    layers = []

    if decoder_type == "original":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    if decoder_type == "conv_relu_dropout":
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
        # if activation == "ReLU":
        #     layers.append(nn.ReLU(inplace=True))
        # elif activation == "LeakyReLU":
        layers.append(nn.LeakyReLU(inplace=True))
        # elif activation == "GELU":
        #     layers.append(nn.GELU())
        # else:
        #     raise ValueError(f"Unsupported activation: {activation}")
        layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.Conv2d(128, out_channels, kernel_size=1))

    elif decoder_type == "conv_bn_relu":
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, out_channels, kernel_size=1))

    elif decoder_type == "conv_transpose":
        layers.append(nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, out_channels, kernel_size=1))
    
    elif decoder_type == "aspp":
        return ASPP(in_channels, out_channels)

    elif decoder_type == "se_block":
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            SEBlock(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    elif decoder_type == "unet_block":
        return UNetDecoderBlock(in_channels, out_channels)
    
    elif decoder_type == "attention":
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            AttentionBlock(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

    return nn.Sequential(*layers)

def build_model(config, base_model):
    original_classifier = base_model.decode_head.classifier

    if config["decoder_type"] == "original":
        # Keep classifier but change output to 1 class
        base_model.decode_head.classifier = nn.Conv2d(
            in_channels=base_model.decode_head.classifier.in_channels,
            out_channels=1, kernel_size=1
        )
    
    else:  # Build extra layers on top of MLP
        new_layers = build_decoder(
            in_channels=original_classifier.out_channels,  # output channels of MLP, usually 150
            out_channels=1,  # final binary output
            decoder_type=config["decoder_type"],
            activation=config["activation"],
            dropout=config["dropout"]
        )

        # Stack original + new layers
        base_model.decode_head.classifier = nn.Sequential(
            original_classifier,
            new_layers
        )
    
    # Optional encoder freezing
    if "freeze_blocks" in config:
        for name, param in base_model.segformer.named_parameters():
            if any(block in name for block in config["freeze_blocks"]):
                param.requires_grad = False

    return base_model

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
    from segformer import CombinedLoss  # Your custom loss
    from segformer import train_one_epoch, evaluate_model  # Your existing training/eval functions
    
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    
    base_model = SegformerForSemanticSegmentation.from_pretrained(config["model_name"])
    model = build_model(config, base_model).to(device)

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
        "Loss": config["loss"],
        "LR": config["lr"],
        "WeightDecay": config.get("weight_decay", 0.0),
        "Posweight": config.get("pos_weight", 1.0),
        "Epochs": config.get("epochs", "N/A"),
        "Encoder FT": "No" if config["freeze_blocks"] else "Yes",
        "Freeze Blocks": config.get("freeze_blocks", []),
        "Params": trainable_params,
        "Acc": round(best_acc, 4),
        "Prec": round(best_precision, 4),
        "Rec": round(best_recall, 4),
        "F1": round(best_f1, 4),
        "mDSC": round(best_dice, 4),
        "mIoU": round(best_iou, 4),
    }


def run_all_experiments(configs, train_dataloader, val_dataloader, device, csv_path = "experiment_results_b0_lossfn.csv"):
    results = []
    #, csv_path="experiment_results_losses.csv"

    for i, config in enumerate(configs):
        print(f"Running experiment {i+1}/{len(configs)}")
        result = run_experiment(config, train_dataloader, val_dataloader, device)
        result["Exp"] = i + 1
        results.append(result)

    df = pd.DataFrame(results)

    df.to_csv(csv_path, index=False)
    # print(f"\nAll experiments completed for {model_id}. Results saved to {csv_path}")
    print(f"\nAll experiments completed. Results saved to {csv_path}")
    print(df)
    
