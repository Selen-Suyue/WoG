import timm
import torch

# Define model names and output paths
models_to_download = {
    "vit_large_patch14_reg4_dinov2.lvd142m": "dinov2_weights.pth",
    "vit_so400m_patch14_siglip_224": "siglip_weights.pth",
}

# Download and save weights
for model_name, output_path in models_to_download.items():
    print(f"Downloading {model_name}...")
    # Create the model, `pretrained=True` will download the weights
    model = timm.create_model(model_name, num_classes=0, pretrained=True, img_size=224)

    # Save the model's state dictionary
    torch.save(model.state_dict(), output_path)
    print(f"Saved weights to {output_path}")
