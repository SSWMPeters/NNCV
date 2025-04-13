from model_vit import Model
import torch

PATH = r"C:\Users\20213002\OneDrive - TU Eindhoven\Master Jaar 1\Q3\Neural Networks\NNCV\Final assignment\submission\model.pth"

model = Model()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
