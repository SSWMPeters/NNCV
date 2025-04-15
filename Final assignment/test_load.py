from model_segformer import SegFormer
import torch

PATH = r"best_model-epoch=0038-val_loss=0.471356887370348.pth"


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
    "drop_prob": 0.1                          # Stochastic depth drop probability
}
model = SegFormer(**segformer_b5_config)

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
# )
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()


