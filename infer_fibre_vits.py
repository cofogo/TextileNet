import torch
import os
import ast
import random
import PIL.Image as Image
from collections import OrderedDict
from torchvision import transforms
from src.vits_models.vision_transformer import vit_tiny_patch16_224 as vit_test

# Directory containing images for inference
image_dir = '/Users/ties/Documents/GitHub/mmfashion/data/Landmark_Detect/Img/img' # Or update to a relevant directory for fibre images if needed
num_random_images = 10 # Number of random images to infer

# Load the model weights
model = vit_test(num_classes=33) # Updated number of classes for fibre

# Load the checkpoint
checkpoint_path = 'baselines/TextileNet-fibre/vits_ckpt.pth' # Updated checkpoint path for fibre
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # or your device
checkpoint = checkpoint['model']
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Create a new state dict without 'module.' prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_key = k[7:]  # remove 'module.' (7 characters)
    else:
        new_key = k
    new_state_dict[new_key] = v

# Load the modified state dict into your model
model.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model.eval().cpu()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Load fibre labels
with open('labels/fibre_label.txt', 'r') as f: # Updated label file path
    content = f.read()

fibre_dict = ast.literal_eval(content) # Renamed variable
fibre_dict = {v: k for k, v in fibre_dict.items()} # Renamed variable

# Get list of all image files
try:
    all_image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
except FileNotFoundError:
    print(f"Error: Image directory not found at {image_dir}")
    exit()

if not all_image_files:
    print(f"Error: No image files found in {image_dir}")
    exit()

# Select random images
random_image_files = random.sample(all_image_files, min(num_random_images, len(all_image_files)))

# Run inference on random images
with torch.no_grad():
    for img_name in random_image_files:
        img_path = os.path.join(image_dir, img_name)
        try:
            example_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Could not open or process image {img_path}: {e}")
            continue

        transformed_image = transform(example_image).unsqueeze(0)
        output = model(transformed_image)
        predicted_label_index = torch.argmax(output).item()

        # Check if the predicted index is in the fibre_dict
        if predicted_label_index in fibre_dict: # Updated variable name
            output_name = fibre_dict[predicted_label_index] # Updated variable name
        else:
            # Handle cases where the prediction index might be out of bounds
            # or not present in the loaded label dictionary.
            output_name = f"Unknown Label Index: {predicted_label_index}"

        # show image
        try:
            example_image.show(f'Image: {img_name}')
        except Exception as e:
            print(f"Could not display image {img_path}: {e}")


        print(f"Image: {img_name}, Prediction: {output_name}")
