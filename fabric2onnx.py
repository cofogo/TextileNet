import torch
from collections import OrderedDict
from src.vits_models.vision_transformer import vit_tiny_patch16_224 as vit_test

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

input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
output = model(input_tensor)

# convert the model to ONNX format
onnx_path = 'onnxmodels/fabric.onnx'
torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    export_params=True,
    opset_version=20,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['fabric'],
    dynamic_axes={'input': {0: 'batch_size'}, 'fabric': {0: 'batch_size'}}
)