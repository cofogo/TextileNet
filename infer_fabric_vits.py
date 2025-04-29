import torch
import os
import ast
import PIL.Image as Image
from collections import OrderedDict
from torchvision import transforms
from src.vits_models.vision_transformer import vit_tiny_patch16_224 as vit_test

test_data = 'data/fabric/test'

# Load the model weights
model = vit_test(num_classes=27)

# Load the checkpoint
checkpoint_path = 'baselines/TextileNet-fabric/vits_ckpt.pth'
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

with open('labels/fabric_label.txt', 'r') as f:
    content = f.read()

fabric_dict = ast.literal_eval(content)
fabric_dict = {v: k for k, v in fabric_dict.items()}

fabrics = fabric_dict.values()

# Run inference
with torch.no_grad():
    for fabric in fabrics:
        fabric_path = os.path.join(test_data, fabric)
        img_name = os.listdir(fabric_path)[0]
        img_path = os.path.join(fabric_path, img_name)
        example_image = Image.open(img_path).convert('RGB')
        transformed_image = transform(example_image).unsqueeze(0)
        output = model(transformed_image)
        output_name = fabric_dict[torch.argmax(output).item()]
        
        # show image
        example_image.show('img')
        
        print(f"{(fabric == output_name)*1} Fabric: {fabric}, Prediction: {output_name}")
