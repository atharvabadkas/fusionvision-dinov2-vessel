import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTModel
import random
from sklearn.model_selection import train_test_split

# Check if MPS is available (for M1 Mac)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths to datasets
INGREDIENTS_PATH = '/Users/atharvabadkas/Coding /DINO/verandah_prep_ingredients'
VESSELS_PATH = '/Users/atharvabadkas/Coding /DINO/verandah_vessel_prep_images'

# Enhanced transformations with stronger augmentations for better generalization
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Larger input size
    transforms.RandomCrop(224),     # Random crop for more variation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # Increased rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # Added scaling
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a custom dataset class
class FoodVesselDataset(Dataset):
    def __init__(self, ingredients_path, vessels_path, transform=None, validation_split=0.1):
        self.transform = transform
        self.validation_split = validation_split
        self.is_validation = False  # Default to training set
        
        # Load ingredient classes and images
        self.ingredient_classes = sorted([d for d in os.listdir(ingredients_path) 
                                         if os.path.isdir(os.path.join(ingredients_path, d)) and not d.startswith('.')])
        self.ingredient_images = []
        self.ingredient_labels = []
        
        print(f"1. Ingredient Classes - {len(self.ingredient_classes)}")
        
        # Process each ingredient class
        for i, ingredient_class in enumerate(self.ingredient_classes):
            class_path = os.path.join(ingredients_path, ingredient_class)
            class_images = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path) 
                           if img_name.endswith(('.jpg', '.jpeg', '.png')) and not img_name.startswith('.')]
            
            # Process images
            valid_images = 0
            for img_path in class_images:
                try:
                    # Just try to open the image to verify it's valid
                    Image.open(img_path).convert('RGB')
                    self.ingredient_images.append(img_path)
                    self.ingredient_labels.append(i)
                    valid_images += 1
                except Exception:
                    pass
            
            print(f"   - {ingredient_class}: {valid_images} images")
        
        # Load vessel classes and images
        self.vessel_classes = sorted([d for d in os.listdir(vessels_path) 
                                     if os.path.isdir(os.path.join(vessels_path, d)) and not d.startswith('.')])
        self.vessel_images = []
        self.vessel_labels = []
        
        print(f"\n2. Vessel Classes - {len(self.vessel_classes)}")
        
        # Process each vessel class
        for i, vessel_class in enumerate(self.vessel_classes):
            class_path = os.path.join(vessels_path, vessel_class)
            class_images = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path) 
                           if img_name.endswith(('.jpg', '.jpeg', '.png')) and not img_name.startswith('.')]
            
            # Process images
            valid_images = 0
            for img_path in class_images:
                try:
                    # Just try to open the image to verify it's valid
                    Image.open(img_path).convert('RGB')
                    self.vessel_images.append(img_path)
                    self.vessel_labels.append(i)
                    valid_images += 1
                except Exception:
                    pass
            
            print(f"   - {vessel_class}: {valid_images} images")
        
        # Combine all images and create dual labels
        self.all_images = self.ingredient_images + self.vessel_images
        
        # For ingredient images, vessel label is -1 (not applicable)
        # For vessel images, ingredient label is -1 (not applicable)
        self.all_ingredient_labels = self.ingredient_labels + [-1] * len(self.vessel_images)
        self.all_vessel_labels = [-1] * len(self.ingredient_images) + self.vessel_labels
        
        print(f"\n3. Total images trained - {len(self.all_images)}")
        print(f"   - Total ingredient images: {len(self.ingredient_images)}")
        print(f"   - Total vessel images: {len(self.vessel_images)}")
        
        # Create indices for train/validation split
        self.indices = list(range(len(self.all_images)))
        
    def set_validation(self, is_validation=False):
        """Set whether this dataset should return validation or training data"""
        self.is_validation = is_validation
        
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        ingredient_label = self.all_ingredient_labels[idx]
        vessel_label = self.all_vessel_labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, ingredient_label, vessel_label

# Load ViT model - using a larger model for better feature extraction
def load_vit_model():
    print("\n5. Loading ViT model for feature extraction...")
    # Use ViT model for feature extraction - using the large model for better features
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    model = ViTModel.from_pretrained("google/vit-large-patch16-224")
    return processor, model

# Enhanced classifier model with deeper architecture and residual connections
class DualClassifier(nn.Module):
    def __init__(self, vit_model, num_ingredient_classes, num_vessel_classes):
        super(DualClassifier, self).__init__()
        self.vit_model = vit_model
        
        # Freeze ViT model parameters
        for param in self.vit_model.parameters():
            param.requires_grad = False
            
        # Get ViT output dimension
        self.feature_dim = 1024  # ViT-Large output dimension
        
        # Deeper ingredient classifier with residual connections
        self.ingredient_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(1024, 768),  # Residual block
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_ingredient_classes)
        )
        
        # Deeper vessel classifier with residual connections
        self.vessel_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(1024, 768),  # Residual block
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_vessel_classes)
        )
        
    def forward(self, x):
        # Extract features using ViT
        with torch.no_grad():
            features = self.vit_model(x).last_hidden_state[:, 0, :]  # Use CLS token
        
        # Pass through classifiers
        ingredient_logits = self.ingredient_classifier(features)
        vessel_logits = self.vessel_classifier(features)
        
        return ingredient_logits, vessel_logits

# Residual block for better gradient flow
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # If dimensions don't match, use a projection shortcut
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

# Enhanced training function with validation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, ingredient_classes, vessel_classes):
    model.train()
    
    # Training statistics
    train_losses = []
    val_losses = []
    ingredient_accuracies = []
    vessel_accuracies = []
    val_ingredient_accuracies = []
    val_vessel_accuracies = []
    
    best_val_accuracy = 0.0
    best_model_state = None
    
    print("\n3. Training model...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        ingredient_correct = 0
        ingredient_total = 0
        vessel_correct = 0
        vessel_total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for images, ingredient_labels, vessel_labels in progress_bar:
            images = images.to(device)
            
            # Filter valid labels (not -1)
            valid_ingredient_mask = ingredient_labels >= 0
            valid_vessel_mask = vessel_labels >= 0
            
            valid_ingredient_labels = ingredient_labels[valid_ingredient_mask].to(device)
            valid_vessel_labels = vessel_labels[valid_vessel_mask].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            ingredient_logits, vessel_logits = model(images)
            
            # Compute loss only for valid labels
            loss = 0
            if valid_ingredient_labels.size(0) > 0:
                ingredient_loss = criterion(ingredient_logits[valid_ingredient_mask], valid_ingredient_labels)
                loss += ingredient_loss
            
            if valid_vessel_labels.size(0) > 0:
                vessel_loss = criterion(vessel_logits[valid_vessel_mask], valid_vessel_labels)
                # Give more weight to vessel loss to improve vessel prediction accuracy
                loss += 2.0 * vessel_loss  # Increased weight for vessel loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Calculate accuracy for ingredients
            if valid_ingredient_labels.size(0) > 0:
                _, ingredient_preds = torch.max(ingredient_logits[valid_ingredient_mask], 1)
                ingredient_correct += (ingredient_preds == valid_ingredient_labels).sum().item()
                ingredient_total += valid_ingredient_labels.size(0)
            
            # Calculate accuracy for vessels
            if valid_vessel_labels.size(0) > 0:
                _, vessel_preds = torch.max(vessel_logits[valid_vessel_mask], 1)
                vessel_correct += (vessel_preds == valid_vessel_labels).sum().item()
                vessel_total += valid_vessel_labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'ing_acc': 100 * ingredient_correct / max(1, ingredient_total),
                'ves_acc': 100 * vessel_correct / max(1, vessel_total)
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        ingredient_accuracy = 100 * ingredient_correct / max(1, ingredient_total)
        vessel_accuracy = 100 * vessel_correct / max(1, vessel_total)
        
        train_losses.append(epoch_loss)
        ingredient_accuracies.append(ingredient_accuracy)
        vessel_accuracies.append(vessel_accuracy)
        
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"Training Ingredient Accuracy: {ingredient_accuracy:.2f}%")
        print(f"Training Vessel Accuracy: {vessel_accuracy:.2f}%")
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_ingredient_correct = 0
        val_ingredient_total = 0
        val_vessel_correct = 0
        val_vessel_total = 0
        
        with torch.no_grad():
            for images, ingredient_labels, vessel_labels in val_loader:
                images = images.to(device)
                
                # Filter valid labels (not -1)
                valid_ingredient_mask = ingredient_labels >= 0
                valid_vessel_mask = vessel_labels >= 0
                
                valid_ingredient_labels = ingredient_labels[valid_ingredient_mask].to(device)
                valid_vessel_labels = vessel_labels[valid_vessel_mask].to(device)
                
                # Forward pass
                ingredient_logits, vessel_logits = model(images)
                
                # Compute loss only for valid labels
                val_loss = 0
                if valid_ingredient_labels.size(0) > 0:
                    ingredient_loss = criterion(ingredient_logits[valid_ingredient_mask], valid_ingredient_labels)
                    val_loss += ingredient_loss
                
                if valid_vessel_labels.size(0) > 0:
                    vessel_loss = criterion(vessel_logits[valid_vessel_mask], valid_vessel_labels)
                    val_loss += 2.0 * vessel_loss
                
                val_running_loss += val_loss.item()
                
                # Calculate accuracy for ingredients
                if valid_ingredient_labels.size(0) > 0:
                    _, ingredient_preds = torch.max(ingredient_logits[valid_ingredient_mask], 1)
                    val_ingredient_correct += (ingredient_preds == valid_ingredient_labels).sum().item()
                    val_ingredient_total += valid_ingredient_labels.size(0)
                
                # Calculate accuracy for vessels
                if valid_vessel_labels.size(0) > 0:
                    _, vessel_preds = torch.max(vessel_logits[valid_vessel_mask], 1)
                    val_vessel_correct += (vessel_preds == valid_vessel_labels).sum().item()
                    val_vessel_total += valid_vessel_labels.size(0)
        
        # Calculate validation statistics
        val_epoch_loss = val_running_loss / len(val_loader)
        val_ingredient_accuracy = 100 * val_ingredient_correct / max(1, val_ingredient_total)
        val_vessel_accuracy = 100 * val_vessel_correct / max(1, val_vessel_total)
        
        val_losses.append(val_epoch_loss)
        val_ingredient_accuracies.append(val_ingredient_accuracy)
        val_vessel_accuracies.append(val_vessel_accuracy)
        
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"Validation Ingredient Accuracy: {val_ingredient_accuracy:.2f}%")
        print(f"Validation Vessel Accuracy: {val_vessel_accuracy:.2f}%")
        
        # Save the best model based on validation accuracy
        current_val_accuracy = (val_ingredient_accuracy + val_vessel_accuracy) / 2
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
        
        # Step the learning rate scheduler
        scheduler.step(val_epoch_loss)
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_accuracy:.2f}%")
    
    # Print final model performance
    print("\n4. Final model accuracy:")
    print(f"   - Ingredient accuracy: {val_ingredient_accuracies[-1]:.2f}%")
    print(f"   - Vessel accuracy: {val_vessel_accuracies[-1]:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(ingredient_accuracies, label='Train')
    plt.plot(val_ingredient_accuracies, label='Validation')
    plt.title('Ingredient Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(vessel_accuracies, label='Train')
    plt.plot(val_vessel_accuracies, label='Validation')
    plt.title('Vessel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to 'training_curves.png'")
    
    return model

# Main function
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset
    dataset = FoodVesselDataset(
        ingredients_path=INGREDIENTS_PATH,
        vessels_path=VESSELS_PATH,
        transform=train_transform
    )
    
    # Create train/validation split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"\nSplit dataset into {train_size} training and {val_size} validation samples")
    
    # Load ViT model
    _, vit_model = load_vit_model()
    vit_model = vit_model.to(device)
    
    # Create classifier model
    model = DualClassifier(
        vit_model=vit_model,
        num_ingredient_classes=len(dataset.ingredient_classes),
        num_vessel_classes=len(dataset.vessel_classes)
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the model for more epochs
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=25,  # Increased number of epochs
        ingredient_classes=dataset.ingredient_classes,
        vessel_classes=dataset.vessel_classes
    )
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'ingredient_classes': dataset.ingredient_classes,
        'vessel_classes': dataset.vessel_classes,
        'feature_dim': model.feature_dim
    }, 'food_vessel_classifier.pth')
    
    print("\nModel saved to 'food_vessel_classifier.pth'")

if __name__ == "__main__":
    main() 