import numpy as np
from torchvision import transforms
import torch
from torchvision.models import ViT_B_16_Weights

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""

    # weights = ViT_B_16_Weights.DEFAULT
    # trans = weights.transforms()

    # img = transforms.functional.resize(img, size=(512, 512), interpolation=transforms.InterpolationMode.LANCZOS)
    # trans = transforms.Compose([transforms.ToTensor()])
    
    trans = transforms.Compose([            #[1]
        transforms.ToImage(),                     #[2]
        transforms.Resize((256, 256)),                    #[2]
        transforms.CenterCrop((224, 224)),                #[3]                 #[4]
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
    )])
    img = trans(img)
    img = img.unsqueeze(0)

    return img

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy









