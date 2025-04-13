from model_segformer import SegFormer
import torch

PATH = r"C:\Users\20213002\OneDrive - TU Eindhoven\Master Jaar 1\Q3\Neural Networks\NNCV\Final assignment\submission\model.pth"

model = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=19,
)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
