import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

# Define the ECNDNet model (this should match the architecture used during training)
class ECNDNet(nn.Module):
    def __init__(self):
        super(ECNDNet, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size=3, dilation=1, use_bn=True, use_relu=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.initial_conv = conv_block(1, 64, use_bn=False)

        self.conv_blocks = nn.ModuleList()
        for i in range(1, 16):
            self.conv_blocks.append(conv_block(64, 64))
            if i % 2 == 0:
                self.conv_blocks.append(nn.Conv2d(64, 64, kernel_size=1))  # Residual connection

            if i in [2, 5, 9, 12]:
                self.conv_blocks.append(conv_block(64, 64, dilation=2))

        self.final_conv = conv_block(64, 64, use_bn=False)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.initial_conv(x)
        residual = x

        for i in range(0, len(self.conv_blocks), 2):
            x = self.conv_blocks[i](x)
            if i % 4 == 0:  # Apply residual connections
                residual = self.conv_blocks[i + 1](x)
                x = x + residual

        x = self.final_conv(x)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model and load the trained weights
model = ECNDNet().to(device)
model_path = "/bit_denoising_model_epoch_2000.pth"

# Load the model state dict, removing 'module.' prefix if necessary
checkpoint = torch.load(model_path)
new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
model.load_state_dict(new_state_dict)
model.eval()  # Set the model to evaluation mode

# Function to load and preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.where(img > 0.5, 1, 0)  # Convert to binary
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Function to perform denoising on a single image
def denoise_image(image_path):
    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    output = output.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    return output

# Perform denoising on a folder of images
def denoise_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                denoised_image = denoise_image(input_path)
                denoised_image = (denoised_image * 255).astype(np.uint8)  # Convert to 8-bit image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, denoised_image)
                print(f"Saved denoised image: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

# Compute PSNR and SSIM between clean and denoised images
def compute_metrics(clean_folder, denoised_folder):
    psnr_list = []
    ssim_list = []

    for filename in os.listdir(clean_folder):
        clean_path = os.path.join(clean_folder, filename)
        denoised_path = os.path.join(denoised_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            denoised_img = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)

            if clean_img is None or denoised_img is None:
                print(f"Failed to read images for PSNR/SSIM calculation: {filename}")
                continue

            # Normalize images to [0, 1] for SSIM
            clean_img = clean_img.astype(np.float32) / 255.0
            denoised_img = denoised_img.astype(np.float32) / 255.0

            # Compute PSNR and SSIM
            psnr_value = compare_psnr(clean_img, denoised_img, data_range=1.0)
            ssim_value = compare_ssim(clean_img, denoised_img, data_range=1.0)

            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)

            print(f"Image: {filename} - PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")

    # Compute average PSNR and SSIM
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    print(f"\nAverage PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")

# Paths to input, output, and clean images folders
input_folder = '/noisy_bit_288/bit_plane_288_4'
output_folder = '/output/denoised_4'
clean_folder = '/clean_bits_288/bit_4_288'

# Perform denoising on the folder of images
denoise_images_in_folder(input_folder, output_folder)

# Compute PSNR and SSIM between clean and denoised images
compute_metrics(clean_folder, output_folder)




