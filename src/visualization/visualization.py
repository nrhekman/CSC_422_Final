# Core libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
\

# Data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Progress tracking and utilities
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"PyTorch version: {torch.__version__}")
print("Setup complete!")


# Implement Grad-CAM for CNN interpretability
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks (implemented for you)
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        """Generate Grad-CAM heatmap"""
        
        # Forward pass to get model output
        self.model.eval()
        output = self.model(input_tensor)    
        
        # Get target class index
        if class_idx is None:
            class_idx = torch.argmax(output)   
        
        # TODO: Backward pass to compute gradients
        self.model.zero_grad()

        output[0, class_idx].backward()  
        
        # Compute Grad-CAM using gradients and activations
        
        # Remove batch dimension
        gradients = self.gradients[0]    # Shape: [channels, height, width]
        activations = self.activations[0]  # Shape: [channels, height, width]
        
        # Compute importance weights by global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))    # global average pool
        
        # Compute weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # [height, width]
        
          
          #- weighted combination
        for i, w in enumerate(weights):
            cam += w * activations[i]  # w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)  # Apply ReLU
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)  # Normalize by max value
        
        return cam.detach().cpu().numpy()


# Apply Grad-CAM to analyze model decisions
def visualize_gradcam_results(model, test_loader, num_samples=8):
    """Visualize Grad-CAM results for sample predictions"""
    
    gradcam = GradCAM(model, 'layer4.1.conv2')
    model.eval()
    
    images, labels, predictions, cams = [], [], [], []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        for i in range(min(num_samples, data.size(0))):
            if len(images) >= num_samples:
                break
            
            # Get single image
            img = data[i:i+1]
            label = target[i].item()

            # Normal forward pass (no grad)
            with torch.no_grad():
                output = model(img)
                pred = torch.argmax(output, dim=1).item()

            # ✅ Grad-CAM requires gradients: enable them here
            with torch.enable_grad():
                img = img.clone().detach().requires_grad_(True)
                cam = gradcam.generate_cam(img, class_idx=pred)

            # Denormalize image
            img_denorm = img[0].cpu()
            img_denorm = img_denorm * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img_denorm = img_denorm + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            images.append(img_denorm.permute(1, 2, 0).detach().numpy())
            labels.append(label)
            predictions.append(pred)
            cams.append(cam)
        
        if len(images) >= num_samples:
            break

    
    # TODO: Visualize results
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 10))
    fig.suptitle('Grad-CAM Analysis: What Does the CNN See?', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'Original\nTrue: {CLASSES[labels[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Grad-CAM heatmap
        im1 = axes[1, i].imshow(cams[i], cmap='jet', alpha=0.7)
        axes[1, i].set_title(f'Grad-CAM\nPred: {CLASSES[predictions[i]]}', fontsize=10)
        axes[1, i].axis('off')
        
        # Overlay
        axes[2, i].imshow(images[i])
        # Resize CAM to match image size
        cam_resized = np.array(Image.fromarray(cams[i]).resize((32, 32)))
        axes[2, i].imshow(cam_resized, cmap='jet', alpha=0.4)
        
        correct = '✓' if labels[i] == predictions[i] else '✗'
        axes[2, i].set_title(f'Overlay {correct}', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze results
    correct_predictions = sum(1 for i in range(num_samples) if labels[i] == predictions[i])
    print(f"\n📊 Analysis Results:")
    print(f"   Accuracy on samples: {correct_predictions}/{num_samples} ({100*correct_predictions/num_samples:.1f}%)")
    print(f"   ✅ Red areas show where the CNN focuses attention")
    
    return images, labels, predictions, cams


# Apply Grad-CAM analysis
test_loader=DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print("🎯 Applying Grad-CAM Analysis...")
for name, model in list(transfer_models.items())[:2]:
    print(name)
    gradcam_results = visualize_gradcam_results(model, test_loader, num_samples=6)

