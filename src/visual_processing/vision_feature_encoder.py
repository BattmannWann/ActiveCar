import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T 
import numpy as np


class FovealVisionEncoder(nn.Module):

    """
    Processes foveal crops through a frozen ResNet-18 backbone into a 514-d vector
    """

    def __init__(self, device: torch.device):

        super().__init__()
        self.device = device

        resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

        #Truncate at the avgpool to remove the 1000-class classification head 
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, 
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        ).to(self.device)

        #Freeze the encoder weights
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False


        self.transform = T.compose([T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])

        #This is just to highlight that we'll have:
            # output state dimension = 512 visual features
            # + 2 gaze coordinates 
            # Therefore, is 514 in total

        self.output_dim = 512 + 2


    def encode(self, crop_rgb: np.ndarray, gaze_coords: np.ndarray) -> np.ndarray:

        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.backbone(tensor).squeeze(0).cpu().numpy()

        #Concatenate: [512 visual latent features, norm_x, and norm_y]
        return np.concatenate([features, gaze_coords], axis = 0).astype(np.float32)


    