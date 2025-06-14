import torch
#import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import AutoModel
# from mmseg.models import SwinTransformer
# print(AutoModel._model_mapping.keys())  # Lists all supported models
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
# from transformers import SwinTransformerForSemanticSegmentation, SwinConfig
from experiment_runner import run_all_experiments
from config_generator import get_loss_lr_configs, get_full_grid_configs  # Explained below
from PIL import Image
#import requests
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import login

class HistologyDataset(Dataset):
    def __init__(self, images_dir, ground_truth_dir, transform=None, processor=None):
        self.images_dir = images_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.processor = processor

        # Detailed directory structure validation
        print("\nChecking directory structure...")

        # Check if directories exist
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        if not os.path.exists(ground_truth_dir):
            raise ValueError(f"Ground truth directory not found: {ground_truth_dir}")

        # Print directory structure
        print("\nCurrent directory structure:")
        print(f"├── {os.path.basename(images_dir)}/")
        image_files = sorted(os.listdir(images_dir))
        for f in image_files[:5]:
            print(f"│   ├── {f}")
        if len(image_files) > 5:
            print("│   ├── ...")

        print(f"├── {os.path.basename(ground_truth_dir)}/")
        mask_files = sorted(os.listdir(ground_truth_dir))
        for f in mask_files[:5]:
            print(f"│   ├── {f}")
        if len(mask_files) > 5:
            print("│   ├── ...")

        # Check file extensions
        print("\nValidating file formats...")

        invalid_images = [f for f in image_files if not f.endswith('.tif')]
        invalid_masks = [f for f in mask_files if not f.endswith('.png')]

        if invalid_images:
            print("\nWarning: Found non-TIF files in images directory:")
            for f in invalid_images[:5]:
                print(f"- {f}")
            if len(invalid_images) > 5:
                print(f"... and {len(invalid_images) - 5} more")

        if invalid_masks:
            print("\nWarning: Found non-PNG files in ground truth directory:")
            for f in invalid_masks[:5]:
                print(f"- {f}")
            if len(invalid_masks) > 5:
                print(f"... and {len(invalid_masks) - 5} more")

        # Get valid files
        self.image_files = sorted([f for f in image_files if f.endswith('.tif')])
        self.mask_files = sorted([f for f in mask_files if f.endswith('.png')])

        print(f"\nFound {len(self.image_files)} TIF files and {len(self.mask_files)} PNG files")

        # Check matching filenames
        self.paired_files = []
        unmatched_images = []
        mask_basenames = {os.path.splitext(f)[0]: f for f in self.mask_files}

        for image_file in self.image_files:
            image_basename = os.path.splitext(image_file)[0]
            if image_basename in mask_basenames:
                self.paired_files.append((image_file, mask_basenames[image_basename]))
            else:
                unmatched_images.append(image_file)

        # Report matching status
        print(f"\nFile matching summary:")
        print(f"- Total image files: {len(self.image_files)}")
        print(f"- Total mask files: {len(self.mask_files)}")
        print(f"- Successfully paired files: {len(self.paired_files)}")
        print(f"- Unmatched files: {len(unmatched_images)}")

        if unmatched_images:
            print("\nWarning: The following images have no matching masks:")
            for img in unmatched_images[:5]:
                print(f"- {img} (expected: {img.replace('.tif', '.png')})")
            if len(unmatched_images) > 5:
                print(f"... and {len(unmatched_images) - 5} more")

        if len(self.paired_files) == 0:
            raise ValueError(
                "\nNo matching image-mask pairs found! Please ensure:\n"
                f"1. Images directory ({images_dir}) contains .tif files\n"
                f"2. Ground truth directory ({ground_truth_dir}) contains .png files\n"
                "3. Filenames match (e.g., 'image1.tif' should have 'image1.png')\n"
                "4. You have read permissions for both directories"
            )

        print("\nDataset initialization complete!")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        image_file, mask_file = self.paired_files[idx]

        # Load image
        image_path = os.path.join(self.images_dir, image_file)
        # image = Image.open(image_path).convert('RGB')

        # Load mask
        mask_path = os.path.join(self.ground_truth_dir, mask_file)
        mask = Image.open(mask_path).convert('RGB')
        image = mask
        grayscale_mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        # Apply transforms
        #print(self.transform)
        #print("Before transform:", type(image))

        if self.processor:
            # Process the image (SegFormer expects normalization and resizing)
            inputs = self.processor(images=image, return_tensors="pt", do_resize=True, do_normalize=True)
            image = inputs['pixel_values'].squeeze(0)  # Remove batch dimension

            # Manually resize the mask to match the processed image's size
            target_size = (image.shape[1], image.shape[2])  # (H, W)
            #print(target_size)
            grayscale_mask = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST)(grayscale_mask)
            #print(grayscale_mask.shape)

            # Convert mask to tensor and ensure it's in integer format
            grayscale_mask = transforms.ToTensor()(grayscale_mask)  # Converts to [0, 1] float
            grayscale_mask = (grayscale_mask > 0.5).float().squeeze(0)  # Convert to class labels
        #print(f"Image shape: {image.shape}, Mask shape: {grayscale_mask.shape}")

        # print("before get_item prints")
        # print("Input shape:", image.shape)  # Expected: [batch_size, channels, height, width]
        # print("Mask shape:", grayscale_mask.shape)    # Expected: [batch_size, height, width]

        return image, grayscale_mask

class SegmentationMetrics:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth

    def dice_coefficient(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred).view(-1).cpu().numpy()
        y_true = y_true.view(-1).cpu().numpy()

        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + self.smooth) / \
               (np.sum(y_true) + np.sum(y_pred) + self.smooth)

    def iou_score(self, y_pred, y_true):
        y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        y_pred = y_pred.view(-1).cpu().numpy()
        y_true = y_true.view(-1).cpu().numpy()

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return (intersection + self.smooth) / (union + self.smooth)

    def pixel_accuracy(self, y_pred, y_true):
        y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        y_pred = y_pred.view(-1).cpu().numpy()
        y_true = y_true.view(-1).cpu().numpy()

        return accuracy_score(y_true, y_pred)

    def pixel_precision_recall(self, y_pred, y_true):
        y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        y_pred = y_pred.view(-1).cpu().numpy()
        y_true = y_true.view(-1).cpu().numpy()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return precision, recall, f1

class CombinedLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        # BCE Loss
        bce_loss = self.bce(predictions, targets)

        # Dice Loss
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice_loss = 1 - ((2. * intersection + self.smooth) /
                        (predictions.sum() + targets.sum() + self.smooth))

        # Combine losses
        return 0.5 * bce_loss + 0.5 * dice_loss

def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)  # Reshape for broadcasting
    std = torch.tensor(std).view(1, -1, 1, 1)
    return image * std + mean


def plot_segmentation_output(images, masks, outputs, batch_idx=0):
    """
    Helper function to visualize the input image, ground truth mask, and predicted mask.

    Parameters:
    - images (Tensor): Batch of input images.
    - masks (Tensor): Ground truth masks.
    - outputs (Tensor): Model's predicted masks.
    - batch_idx (int): Index of the image in the batch to visualize.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Move tensors to CPU and detach
    images = images[batch_idx].cpu().detach()
    masks = masks[batch_idx].cpu().detach()
    outputs = outputs[batch_idx].cpu().detach()

    # De-normalize the images
    images = images.permute(1, 2, 0).numpy()  # Convert to HWC
    images = images * std + mean  # Reverse normalization
    images = np.clip(images, 0, 1)  # Clip values to valid range [0, 1]

    if outputs.ndim == 2:  # Already a [H, W] mask
        # print("in if")
        predicted_mask = outputs.numpy()
    elif outputs.shape[0] == 1:  # Binary, still has channel dim
        predicted_mask = (torch.sigmoid(outputs[0]) > 0.5).numpy().astype(np.uint8)
    else:
        predicted_mask = torch.argmax(outputs, dim=0).numpy()

    ground_truth_mask = masks.numpy()

    # Plot the first image in the batch
    plt.figure(figsize=(12, 6))

    # Display the input image
    plt.subplot(1, 3, 1)
    plt.imshow(images)
    plt.title("Input Image")
    plt.axis('off')

    # Display the ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # Display the predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()

def evaluate_model(model, dataloader, device, criterion=None, visualize=True):
    model.eval()
    metrics = SegmentationMetrics()

    total_dice = 0
    total_iou = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0.0
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # outputs = model(images)['out']
            outputs = model(images)
            # print(outputs)  # Check the structure of the output dictionary
            outputs = model(images).logits
            # print("Input shape in eval:", images.shape)  # Expected: [batch_size, channels, height, width]
            # print("Mask shape in eval:", masks.shape)
            # print("Output shape in eval:", outputs.shape)

            if outputs.shape[-2:] != masks.shape[-2:]:
                # print("Upsampling output to match target size...")

                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
                # print(f"Upsampled output shape: {outputs.shape}")

            # Threshold or argmax predictions
            if outputs.shape[1] == 1:  # Binary segmentation
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float().squeeze(1)
                # print("in if")
            else:  # Multi-class segmentation
                outputs = torch.argmax(outputs, dim=1)
                # print("in else")
            # print("Output shape in eval2:", outputs.shape)

            # Calculate metrics
            # print('calculating metrics...')
            dice = metrics.dice_coefficient(outputs, masks)
            iou = metrics.iou_score(outputs, masks)
            accuracy = metrics.pixel_accuracy(outputs, masks)
            precision, recall, f1 = metrics.pixel_precision_recall(outputs, masks)

            total_dice += dice
            total_iou += iou
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Calculate validation loss
            if criterion:
                loss = criterion(outputs, masks)
                total_loss += loss.item()

            # Optional: Visualize outputs
            # if visualize:
            #     plot_segmentation_output(images, masks, outputs, batch_idx=0)

            # # Visualize the outputs (just the first image in the batch)
            # plot_segmentation_output(images, masks, outputs, batch_idx=0)

    # Calculate averages
    avg_metrics = {
        'Dice': total_dice / num_batches,
        'IoU': total_iou / num_batches,
        'Accuracy': total_accuracy / num_batches,
        'Precision': total_precision / num_batches,
        'Recall': total_recall / num_batches,
        'F1': total_f1 / num_batches,
        'Loss': total_loss / num_batches if criterion else None
    }
    # print("end of evaluate_model")
    return avg_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # Debugging: Print shapes
        # print("Input shape in loop:", images.shape)  # Expected: [batch_size, channels, height, width]
        # print("Mask shape in loop:", masks.shape)    # Expected: [batch_size, height, width]

        # outputs = model(images)['out']
        outputs = model(images)
        # print(outputs)  # Check the structure of the output dictionary
        outputs = model(images).logits
        # print("Output shape in loop:", outputs.shape)  # Expected: [batch_size, num_classes, height, width]

        if outputs.shape[-2:] != masks.shape[-2:]:
            # print("Upsampling output to match target size...")

            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False
            )
            outputs= outputs.squeeze(1)
            # print(f"Upsampled output shape: {outputs.shape}")


        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def plot_training_metrics(epochs, train_losses, val_losses, dice_scores, iou_scores):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, label='Training Loss', color='blue')
    axes[0].plot(epochs, val_losses, label='Validation Loss', color='orange')
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Dice score plot
    axes[1].plot(epochs, dice_scores, label='Dice Score', color='green')
    axes[1].set_title("Dice Coefficient")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()

    # IoU plot
    axes[2].plot(epochs, iou_scores, label='IoU', color='red')
    axes[2].set_title("Intersection over Union (IoU)")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("IoU")
    axes[2].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images_dir = "/rds/general/user/mm4321/home/code/FYP/7379186/BreCaHAD/images"
    ground_truth_dir = "/rds/general/user/mm4321/home/code/FYP/7379186/BreCaHAD/groundTruth_display"
    # ground_truth_json = r'C:\Users\madhu\Documents\UNI-IMPERIAL\FYP\7379186\BreCaHAD\groundTruth'

    try:
        # Initialize model
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        # model_names = [
        #     "nvidia/segformer-b0-finetuned-ade-512-512",
        #     "nvidia/segformer-b4-finetuned-ade-512-512",
        #     "nvidia/segformer-b5-finetuned-ade-640-640"
        # ]
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        processor = SegformerImageProcessor.from_pretrained(model_name)
        # model.classifier = torch.nn.Conv2d(in_channels=model.config.decoder_hidden_size, out_channels=1, kernel_size=1)
        model.decode_head.classifier = torch.nn.Conv2d(in_channels=model.decode_head.classifier.in_channels, out_channels=1, kernel_size=1)
        #model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=1)

        # Create dataset with PNG masks
        print("\nInitializing dataset...")
        dataset = HistologyDataset(
            images_dir=images_dir ,  # Directory containing .tif files
            ground_truth_dir=ground_truth_dir,  # Directory containing .png mask files
            # ground_truth_json=ground_truth_json,
            transform=transform,
            processor=processor
        )

        print(f"\nTotal dataset size: {len(dataset)}")

        # Split dataset into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        print(f"\nSplitting dataset:")
        print(f"- Training samples: {train_size}")
        print(f"- Validation samples: {val_size}")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=True)

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Evaluate pretrained model with different losses
        for loss_fn_name, criterion in [
            ("Combined Loss", CombinedLoss().to(device)),
            ("BCEWithLogitsLoss", nn.BCEWithLogitsLoss().to(device))
        ]:
            print(f"\nRunning baseline evaluation on pretrained model using {loss_fn_name}...")
            
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            with torch.no_grad():
                baseline_metrics = evaluate_model(model, val_dataloader, device, criterion)

            print(f"\n📊 Baseline pretrained model performance with {loss_fn_name} (no fine-tuning):")
            for metric_name, value in baseline_metrics.items():
                print(f"{metric_name}: {value:.4f}")

        configs = get_loss_lr_configs(model_name)
        run_all_experiments(configs, train_dataloader, val_dataloader, device)
        
        # configs = get_full_grid_configs(model_name)
        # run_all_experiments(configs, train_dataloader, val_dataloader, device)

        # # Initialize loss function and optimizer
        # criterion = CombinedLoss().to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # # Training loop with evaluation
        # num_epochs = 20
        # best_dice = 0
        # best_model_path = 'best_model.pth'

        # train_losses = []
        # val_losses = []
        # dice_scores = []
        # iou_scores = []

        # for epoch in range(num_epochs):
        #     # Training
        #     avg_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        #     train_losses.append(avg_loss)

        #     # Evaluation
        #     print("\nEvaluating model...")
        #     eval_metrics = evaluate_model(model, val_dataloader, device, criterion)
        #     val_losses.append(eval_metrics['Loss'])  # Add if you calculate validation loss
        #     print(f"Validation Loss: {eval_metrics['Loss']}")
        #     dice_scores.append(eval_metrics['Dice'])
        #     iou_scores.append(eval_metrics['IoU'])

        #     # Print metrics
        #     print(f"Epoch {epoch+1}/{num_epochs}")
        #     print(f"Training Loss: {avg_loss:.4f}")
        #     print("Validation Metrics:")
        #     for metric_name, value in eval_metrics.items():
        #         print(f"{metric_name}: {value:.4f}")

        #     # Save best model based on Dice score
        #     if eval_metrics['Dice'] > best_dice:
        #         best_dice = eval_metrics['Dice']
        #         torch.save(model.state_dict(), best_model_path)
        #         print(f"New best model saved with Dice score: {best_dice:.4f}")

        #     print("-" * 50)

        # plot_training_metrics(range(1, num_epochs + 1), train_losses, val_losses, dice_scores, iou_scores)

    except Exception as e:
        print("\nError during training:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nPlease check the error message above.")
        return

if __name__ == "__main__":
    main()