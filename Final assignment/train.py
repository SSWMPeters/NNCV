"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    ToTensor,
    CenterCrop,
    ConvertImageDtype
)
from torchvision import models
from model_segformer_ood import Model

from MLCDiceLoss import MultiClassDiceLoss

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="segformer_train", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    # Original:
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    # transform = Compose([            #[1]
    #     ToImage(),                     #[2]
    #     Resize((256, 256)),                    #[2]
    #     CenterCrop((224, 224)),                #[3]                 #[4]
    #     ToDtype(torch.float32, scale=True),
    #     Normalize(                      #[5]
    #     mean=[0.485, 0.456, 0.406],                #[6]
    #     std=[0.229, 0.224, 0.225]                  #[7]
    # )])


    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load pretrained ViT model
    # vit_model = models.vit_b_16(pretrained=True)

    segformer_b5_config = {
        "in_channels": 3,
        "widths": [64, 128, 320, 512],             # Feature map widths for each stage
        "depths": [3, 6, 40, 3],                   # Number of transformer blocks per stage
        "all_num_heads": [1, 2, 5, 8],             # Attention heads per stage
        "patch_sizes": [7, 3, 3, 3],               # Patch sizes for Overlap Patch Merging
        "overlap_sizes": [4, 2, 2, 2],             # Overlap stride sizes
        "reduction_ratios": [8, 4, 2, 1],          # Attention spatial reduction per stage
        "mlp_expansions": [4, 4, 4, 4],            # MLP hidden expansion factor
        "decoder_channels": 256,                  # Channels used in decoder
        "scale_factors": [8, 4, 2, 1],             # For upsampling in decoder (reverse of resolution drops)
        "num_classes": 19,                        # Replace with your target number of classes
        "drop_prob": 0.1,                          # Stochastic depth drop probability
        "threshold": -12.0,                      # Threshold for the score
    }


    model = Model(**segformer_b5_config).to(device)

    # model = SegFormer(
    #     in_channels=3,
    #     widths=[64, 128, 256, 512],
    #     depths=[3, 4, 6, 3],
    #     all_num_heads=[1, 2, 4, 8],
    #     patch_sizes=[7, 3, 3, 3],
    #     overlap_sizes=[4, 2, 2, 2],
    #     reduction_ratios=[8, 4, 2, 1],
    #     mlp_expansions=[4, 4, 4, 4],
    #     decoder_channels=256,
    #     scale_factors=[8, 4, 2, 1],
    #     num_classes=19,
    # ).to(device)

    # model = Model( 
    #     num_classes=19,
    # ).to(device)

    # Define the model
    # Oringinal:
    # model = Model(
    #     in_channels=3,  # RGB images
    #     n_classes=19,  # 19 classes in the Cityscapes dataset
    # ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    dice_loss = MultiClassDiceLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5)


    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        train_energy = []
        train_flag = []
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs, flag, score = model(images)
            loss_dice = dice_loss(outputs, labels)
            loss_CE = criterion(outputs, labels)
            loss = loss_dice

            train_energy.append(torch.mean(score))
            train_flag.append(1 - (torch.sum(flag)/len(flag)))
            # dice_score = dice_loss(outputs, labels)  # Calculate the Dice loss
            loss.backward()
            optimizer.step()

            # dice = nn.dice_score(outputs, labels)  # Calculate the Dice score
            # print(f"Dice training Score: {dice:.4f}")
            # "learning_rate": optimizer.param_groups[0]['lr'],

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
        train_energy_avg = sum(train_energy) / len(train_energy)
        print(f"Training Energy: {train_energy_avg:.4f}")
        train_flag_avg = sum(train_flag) / len(train_flag)
        print(f"Training Flag: {train_flag_avg:.4f}")
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            losses_CE = []
            losses_dice = []
            valid_energy = []
            valid_flag = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs, flag, score = model(images)
                loss_CE = criterion(outputs, labels)
                losses_CE.append(loss_CE.item())

                valid_energy.append(torch.mean(score))

                loss_dice = dice_loss(outputs, labels)  # Calculate the Dice score
                losses_dice.append(loss_dice.item())

                loss = loss_dice
                losses.append(loss.item())

                valid_flag.append(1 - (torch.sum(flag)/len(flag)))

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            scheduler.step(valid_loss)
            
            valid_flag_avg = sum(valid_flag) / len(valid_flag)
            print(f"Validation Flag: {valid_flag_avg:.4f}")

            valid_energy_avg = sum(valid_energy) / len(valid_energy)
            print(f"Validation Energy: {valid_energy_avg:.4f}")

            valid_loss_CE = sum(losses_CE) / len(losses_CE)
            valid_loss_dice = sum(losses_dice) / len(losses_dice)
            print(f"Validation CE Loss: {valid_loss_CE:.4f}")
            print(f"Validation Dice Loss: {valid_loss_dice:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # print(f"Dice validation Score: {sum(dices) / len(dices):.4f}")
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)


