import torch
import cv2
from FCN_network import FullyConvNetwork
from torch.utils.data import DataLoader

# Set device to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FullyConvNetwork().to(device)

model_path = './checkpoints/pix2pix_model_epoch_20.pth'

model.load_state_dict(torch.load(model_path))

model.eval()

image_path = './datasets/facades/test/12.jpg'
img_color_semantic = cv2.imread(image_path)
image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
image_rgb = image[:, :, :256].to(device)
image_semantic = image[:, :, 256:]
