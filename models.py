# Core libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

# Data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Progress tracking and utilities
from tqdm import tqdm
import time
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


# Setup for transfer learning
print("🔄 Setting up Transfer Learning")

def create_pretrained_model(model_name='resnet18', num_classes=13, feature_extract=True):
    """Create a pre-trained model for transfer learning"""
    
    model = None
    
    if model_name == 'resnet18':
        #Load pre-trained ResNet18
        model = models.resnet18(pretrained=True)   
        
        # Freeze parameters for feature extraction
        if feature_extract == True:
            for param in model.parameters(): 
                param.requires_grad = False
            pass
        
        # Replace final layer for CIFAR-10 (10 classes instead of 1000)
        num_features = model.fc.in_features  # Get model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)      # Create new Linear layer
        
    
    return model

# Create different transfer learning approaches

print("🔧 Creating Transfer Learning Models...")

try:
    transfer_models = {
        #  Create ResNet18 with feature extraction
        'ResNet18 Feature Extraction': create_pretrained_model('resnet18', num_classes=13, feature_extract=True),   
        
        # Create ResNet18 with fine-tuning  
        'ResNet18 Fine-tuning': create_pretrained_model('resnet18', num_classes=13, feature_extract=False),   
    }

    # Move models to device and analyze
    print("📊 Transfer Learning Model Analysis:")
    print("=" * 70)
    
    for name, model in transfer_models.items():
        if model is not None:
            model = model.to(device)
            
            # Count trainable vs total parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"{name:25} | {trainable_params:>8,} / {total_params:>8,} trainable params")
        else:
            print(f"{name:25} | NOT IMPLEMENTED")

    
except Exception as e:
    print(f"❌  {e}")

# Compare transfer learning approaches
print("⚖️ Comparing Transfer Learning Approaches")
print("=" * 50)

# Modified training function for transfer learning
def train_transfer_model(model, train_loader, val_loader, model_name, num_epochs=5):
    """Train transfer learning model with appropriate learning rate"""
    
    # Different learning rates for different approaches
    if 'Feature Extraction' in model_name:
        learning_rate = 0.001  # Higher LR for feature extraction
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        learning_rate = 0.0001  # Lower LR for fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🔄 Training {model_name}")
    print(f"Learning Rate: {learning_rate}, Epochs: {num_epochs}")
    
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"   Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    return history, val_acc

# Train and compare transfer learning models

transfer_results = {}
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(valid_ds, batch_size=32)

# Train each transfer learning approach
for name, model in list(transfer_models.items())[:2]:  # Train first 2 models
    start_time = time.time()
    history, final_acc = train_transfer_model(
        model, train_loader, val_loader, name, num_epochs=5
    )
    training_time = time.time() - start_time
    
    transfer_results[name] = {
        'history': history,
        'final_accuracy': final_acc,
        'training_time': training_time
    }
    
    print(f"   ✅ Completed in {training_time:.1f}s, Final Accuracy: {final_acc:.2f}%")

print(f"\n📊 Transfer Learning Results Summary:")
for name, results in transfer_results.items():
    print(f"   {name}: {results['final_accuracy']:.2f}% in {results['training_time']:.1f}s")

# Save Models
model1= transfer_models['ResNet18 Feature Extraction']

torch.save(model1.state_dict(), 'feature_extraction_pest.pth')

model2= transfer_models['ResNet18 Fine-tuning']
torch.save(model2.state_dict(), 'fine_tuning_pest.pth')