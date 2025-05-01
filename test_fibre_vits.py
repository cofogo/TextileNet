import torch
import os
import ast
import PIL.Image as Image
from collections import OrderedDict
from torchvision import transforms
from src.vits_models.vision_transformer import vit_tiny_patch16_224 as vit_test

test_data = 'data/fibre/test' # Updated test data path for fibre

# Load the model weights
model = vit_test(num_classes=10) # Updated number of classes for fibre

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

fibres = fibre_dict.values() # Renamed variable

# Run inference
with torch.no_grad():
    for fibre in fibres: # Renamed variable
        fibre_path = os.path.join(test_data, fibre) # Renamed variable
        # Handle cases where the directory might not exist or is empty
        if not os.path.exists(fibre_path) or not os.listdir(fibre_path):
            print(f"Warning: Directory not found or empty for fibre: {fibre}. Skipping.")
            continue
        img_name = os.listdir(fibre_path)[0]
        img_path = os.path.join(fibre_path, img_name)
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
            output_name = f"Unknown Label Index: {predicted_label_index}"

        # show image
        try:
            example_image.show('img')
        except Exception as e:
            print(f"Could not display image {img_path}: {e}")

        print(f"{(fibre == output_name)*1} Fibre: {fibre}, Prediction: {output_name}") # Updated print statement
