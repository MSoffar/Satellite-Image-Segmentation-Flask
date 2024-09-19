from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import rasterio
from io import BytesIO
import base64
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

# Initialize Flask app
app = Flask(__name__)

# Define custom decoder (classifier)
class CustomDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CustomDecoder, self).__init__()
        self.dropout_rate = 0.45
        
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.final_conv(x)
        return x

# Load model and set to evaluation mode
model_path = 'model/best_model.pth'
device = torch.device('cpu')  # Use 'cpu' if you're not using a GPU

# Initialize the model architecture (adjusted for 12-channel input)
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

# Adjust the first convolutional layer to accept 12-channel input
first_conv_layer = model.backbone.conv1
model.backbone.conv1 = nn.Conv2d(12, first_conv_layer.out_channels,
                                 kernel_size=first_conv_layer.kernel_size,
                                 stride=first_conv_layer.stride,
                                 padding=first_conv_layer.padding,
                                 bias=first_conv_layer.bias)

# Replace the classifier with the custom decoder
model.classifier = CustomDecoder(2048)

# Load the model's state_dict (weights) into the architecture
model.load_state_dict(torch.load(model_path, map_location=device))

# Set model to evaluation mode
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.tif'):
            # Process the file in-memory
            image_bytes = file.read()
            original_img_base64, mask_base64 = predict(image_bytes)
            
            # Render the result page with both images side by side
            return render_template('result.html', original_image=original_img_base64, predicted_image=mask_base64)

def predict(image_bytes):
    # Read the 12-channel .tif image from memory
    with rasterio.MemoryFile(image_bytes) as memfile:
        with memfile.open() as src:
            image = src.read().astype(np.float32)  # Reading all 12 bands

    # Normalize the image
    image_normalized = normalize_data(image)

    # Convert the image to a tensor (PyTorch format)
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).float().to(device)  # Add batch dimension
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output['out'])  # Extract 'out' from OrderedDict and apply sigmoid
        mask = (output > 0.5).float().cpu().numpy().squeeze()  # Binarize and remove extra dimension

    # Convert original image (first 3 bands for display) to PIL Image
    original_img = np.transpose(image[:3, :, :], (1, 2, 0))  # Taking the first 3 channels as RGB
    original_img_pil = Image.fromarray((original_img * 255).astype(np.uint8))

    # Convert mask to a PIL Image
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to 8-bit grayscale

    # Convert both images to base64 format
    original_img_base64 = image_to_base64(original_img_pil)
    mask_base64 = image_to_base64(mask_pil)

    return original_img_base64, mask_base64

def image_to_base64(image):
    """Converts a PIL image to base64 string."""
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def normalize_data(images):
    """Normalizes each channel of the input image."""
    for i in range(images.shape[0]):
        band = images[i, :, :]
        min_val = np.min(band)
        max_val = np.max(band)
        images[i, :, :] = (band - min_val) / (max_val - min_val)
    return images

if __name__ == '__main__':
    app.run(debug=True)
