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
from transformers import AutoImageProcessor, AutoModel
import random
from sklearn.model_selection import train_test_split
import cv2
from skimage.feature import hog
import datetime
import traceback

# ============================
# Configuration
# ============================

# Device setup - Enhanced to match model_test.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
print(f"Using device: {device}")

# Dataset paths
INGREDIENTS_PATH = '/Users/atharvabadkas/Coding /DINO/dino_feature_extraction_model/verandah_prep_ingredients'
VESSELS_PATH = '/Users/atharvabadkas/Coding /DINO/dino_feature_extraction_model/verandah_vessel_prep_images'

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================
# Data Augmentation
# ============================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================
# Feature Extraction
# ============================

def extract_shape_features(image_pil, feature_size=256):
    """
    Enhanced shape feature extraction with additional features:
    1. Basic metrics (area, perimeter, aspect ratio)
    2. Shape complexity and symmetry
    3. Texture analysis
    4. Multi-scale features
    5. Spatial relationships
    6. Enhanced HOG features
    """
    # Convert and preprocess image
    image = np.array(image_pil.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Multi-scale processing
    scales = [0.5, 1.0, 2.0]
    features = []
    
    for scale in scales:
        # Resize image for current scale
        current_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
        scaled_gray = cv2.resize(gray, current_size)
        
        # Edge detection with multiple methods
        edges_canny = cv2.Canny(scaled_gray, 30, 150)
        sobel_x = cv2.Sobel(scaled_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(scaled_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        laplacian = cv2.Laplacian(scaled_gray, cv2.CV_64F)
        
        # Find contours
        edges_combined = cv2.addWeighted(edges_canny, 0.7, 
            cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), 0.3, 0)
        contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic shape metrics
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img_area = scaled_gray.shape[0] * scaled_gray.shape[1]
        
        features.extend([
            area / img_area,
            perimeter / np.sqrt(img_area),
            float(w) / h if h > 0 else 0,
            float(area) / (w * h) if w * h > 0 else 0
        ])
        
        # Shape complexity and symmetry
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        moments = cv2.moments(largest_contour)
        
        # Symmetry features
        if moments['m00'] != 0:
            centroid_x = moments['m10'] / moments['m00']
            centroid_y = moments['m01'] / moments['m00']
            
            left_points = [pt for pt in largest_contour[:, 0, :] if pt[0] < centroid_x]
            right_points = [pt for pt in largest_contour[:, 0, :] if pt[0] > centroid_x]
            
            if left_points and right_points:
                left_mean = np.mean(left_points, axis=0)
                right_mean = np.mean(right_points, axis=0)
                symmetry_score = np.linalg.norm(left_mean - right_mean) / w
                features.append(symmetry_score)
        
        # Texture features using GLCM
        if scaled_gray.size > 0:
            glcm = cv2.resize(scaled_gray, (32, 32))
            features.extend([
                np.mean(glcm),
                np.std(glcm),
                np.percentile(glcm, 25),
                np.percentile(glcm, 75)
            ])
        
        # Enhanced HOG features with multiple cell sizes
        for cell_size in [(8, 8), (16, 16)]:
            if scaled_gray.size > 0:
                hog_image = cv2.resize(scaled_gray, (64, 64))
                hog_features = hog(hog_image, orientations=9,
                                pixels_per_cell=cell_size,
                                cells_per_block=(2, 2),
                                feature_vector=True)
                features.extend(hog_features[:50])
    
    # Ensure consistent feature size
    if len(features) > feature_size:
        features = features[:feature_size]
    else:
        features.extend([0] * (feature_size - len(features)))
    
    return np.array(features)

# ============================
# Dataset Implementation
# ============================

class FoodVesselDataset(Dataset):
    """
    Custom dataset for loading and processing food and vessel images.
    Handles both ingredient and vessel images with their respective labels.
    """
    def __init__(self, ingredients_path, vessels_path, transform=None):
        self.transform = transform
        
        # Load ingredient classes and images
        self.ingredient_classes = sorted([d for d in os.listdir(ingredients_path) 
                                       if os.path.isdir(os.path.join(ingredients_path, d)) 
                                       and not d.startswith('.')])
        self.ingredient_images = []
        self.ingredient_labels = []
        
        print(f"Loading ingredient classes: {len(self.ingredient_classes)}")
        
        # Process ingredient images
        for i, ingredient_class in enumerate(self.ingredient_classes):
            class_path = os.path.join(ingredients_path, ingredient_class)
            class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) 
                          if img.endswith(('.jpg', '.jpeg', '.png')) and not img.startswith('.')]
            
            valid_images = 0
            for img_path in class_images:
                try:
                    Image.open(img_path).convert('RGB')
                    self.ingredient_images.append(img_path)
                    self.ingredient_labels.append(i)
                    valid_images += 1
                except Exception:
                    continue
            
            print(f"  - {ingredient_class}: {valid_images} images")
        
        # Load vessel classes and images
        self.vessel_classes = sorted([d for d in os.listdir(vessels_path) 
                                    if os.path.isdir(os.path.join(vessels_path, d)) 
                                    and not d.startswith('.')])
        self.vessel_images = []
        self.vessel_labels = []
        self.vessel_shape_features = []
        
        print(f"\nLoading vessel classes: {len(self.vessel_classes)}")
        
        # Process vessel images
        for i, vessel_class in enumerate(self.vessel_classes):
            class_path = os.path.join(vessels_path, vessel_class)
            class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) 
                          if img.endswith(('.jpg', '.jpeg', '.png')) and not img.startswith('.')]
            
            valid_images = 0
            for img_path in class_images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    shape_features = extract_shape_features(img)
                    
                    self.vessel_images.append(img_path)
                    self.vessel_labels.append(i)
                    self.vessel_shape_features.append(shape_features)
                    valid_images += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            print(f"  - {vessel_class}: {valid_images} images")
        
        # Combine all data
        self.all_images = self.ingredient_images + self.vessel_images
        self.all_ingredient_labels = self.ingredient_labels + [-1] * len(self.vessel_images)
        self.all_vessel_labels = [-1] * len(self.ingredient_images) + self.vessel_labels
        self.all_shape_features = [None] * len(self.ingredient_images) + self.vessel_shape_features
        
        print(f"\nDataset Summary:")
        print(f"  - Total images: {len(self.all_images)}")
        print(f"  - Ingredient images: {len(self.ingredient_images)}")
        print(f"  - Vessel images: {len(self.vessel_images)}")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        # Load image and labels
        img_path = self.all_images[idx]
        ingredient_label = self.all_ingredient_labels[idx]
        vessel_label = self.all_vessel_labels[idx]
        shape_features = self.all_shape_features[idx]
        
        # Process image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Handle shape features
        if shape_features is None:
            shape_features = torch.zeros(256, dtype=torch.float32)
        else:
            shape_features = torch.tensor(shape_features, dtype=torch.float32)
        
        return image, ingredient_label, vessel_label, shape_features

# ============================
# Model Architecture
# ============================

class EnhancedResidualBlock(nn.Module):
    """Enhanced residual block with squeeze-excitation and multi-branch processing"""
    def __init__(self, in_features, out_features):
        super(EnhancedResidualBlock, self).__init__()
        
        self.expansion = 4
        hidden_features = out_features // self.expansion
        
        # Main branch
        self.conv1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        
        # Multi-branch processing
        self.branch1 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        
        # Combine branches
        self.combine = nn.Sequential(
            nn.Linear(hidden_features * 2, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Squeeze-Excitation
        self.se = nn.Sequential(
            nn.Linear(out_features, out_features // 16),
            nn.ReLU(),
            nn.Linear(out_features // 16, out_features),
            nn.Sigmoid()
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features)
        ) if in_features != out_features else nn.Identity()
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Main processing
        out = self.conv1(x)
        
        # Multi-branch processing
        branch1 = self.branch1(out)
        branch2 = self.branch2(out)
        
        # Combine branches
        out = torch.cat([branch1, branch2], dim=1)
        out = self.combine(out)
        
        # Apply SE attention
        se_weights = self.se(out)
        out = out * se_weights
        
        # Add shortcut and apply ReLU
        out += identity
        return self.relu(out)

class DualClassifier(nn.Module):
    """
    Enhanced classifier model with improved feature processing and fusion
    """
    def __init__(self, dinov2_model, num_ingredient_classes, num_vessel_classes, 
                 shape_feature_dim=256, feature_dim=768):
        super(DualClassifier, self).__init__()
        self.dinov2_model = dinov2_model
        self.shape_feature_dim = shape_feature_dim
        self.feature_dim = feature_dim
        
        # Freeze DINOv2 parameters
        for param in self.dinov2_model.parameters():
            param.requires_grad = False
        
        # Visual feature processing
        self.visual_features = nn.Sequential(
            EnhancedResidualBlock(self.feature_dim, 1024),
            nn.Dropout(0.3),
            EnhancedResidualBlock(1024, 512),
            nn.Dropout(0.2)
        )
        
        # Shape feature processing
        self.shape_features = nn.Sequential(
            EnhancedResidualBlock(self.shape_feature_dim, 512),
            nn.Dropout(0.2),
            EnhancedResidualBlock(512, 512),
            nn.Dropout(0.2)
        )
        
        # Multi-head cross attention for feature fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.attention_norm = nn.LayerNorm(512)
        
        # Feature pyramid for multi-scale analysis - Fixed dimensions 
        self.pyramid_layer1 = nn.Sequential(
            EnhancedResidualBlock(512, 256),
            nn.Dropout(0.2)
        )
        
        self.pyramid_layer2 = nn.Sequential(
            EnhancedResidualBlock(256, 256),
            nn.Dropout(0.2)
        )
        
        self.pyramid_layer3 = nn.Sequential(
            EnhancedResidualBlock(256, 256),
            nn.Dropout(0.2)
        )
        
        # Classification heads with deep supervision
        self.ingredient_classifier = nn.Sequential(
            EnhancedResidualBlock(512, 512),
            nn.Dropout(0.3),
            EnhancedResidualBlock(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, num_ingredient_classes)
        )
        
        self.vessel_classifier = nn.Sequential(
            EnhancedResidualBlock(768, 512),  # Changed from 1024 to 768 (512+256)
            nn.Dropout(0.3),
            EnhancedResidualBlock(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, num_vessel_classes)
        )
        
    def forward(self, x, shape_features):
        # Extract and process visual features
        with torch.no_grad():
            visual_features = self.dinov2_model(x).last_hidden_state[:, 0, :]
        processed_visual = self.visual_features(visual_features)
        
        # Process shape features
        processed_shape = self.shape_features(shape_features)
        
        # Multi-scale feature processing - Apply layers sequentially
        pyramid_out1 = self.pyramid_layer1(processed_visual)
        pyramid_out2 = self.pyramid_layer2(pyramid_out1)
        pyramid_out3 = self.pyramid_layer3(pyramid_out2)
        
        # Feature fusion with cross attention
        shape_attn = processed_shape.unsqueeze(0)
        visual_attn = processed_visual.unsqueeze(0)
        attended_features, _ = self.cross_attention(shape_attn, visual_attn, visual_attn)
        
        # Combine features with residual connection
        fused_features = self.attention_norm(attended_features.squeeze(0) + shape_attn.squeeze(0))
        
        # Concatenate with final pyramid feature for vessel classification
        vessel_features = torch.cat([fused_features, pyramid_out3], dim=1)
        
        # Classification
        return self.ingredient_classifier(processed_visual), self.vessel_classifier(vessel_features)

# ============================
# Training Functions
# ============================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """Train the model with validation"""
    best_val_accuracy = 0.0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 
              'train_acc_ing': [], 'train_acc_ves': [],
              'val_acc_ing': [], 'val_acc_ves': []}
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_stats = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validation phase
        model.eval()
        val_stats = validate_epoch(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step(val_stats['loss'])
        
        # Save statistics
        for key in train_stats:
            history[f'train_{key}'].append(train_stats[key])
        for key in val_stats:
            history[f'val_{key}'].append(val_stats[key])
        
        # Save best model
        current_acc = (val_stats['acc_ing'] + val_stats['acc_ves']) / 2
        if current_acc > best_val_accuracy:
            best_val_accuracy = current_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_stats['loss']:.4f}, "
              f"Ing Acc: {train_stats['acc_ing']:.2f}%, "
              f"Ves Acc: {train_stats['acc_ves']:.2f}%")
        print(f"Val Loss: {val_stats['loss']:.4f}, "
              f"Ing Acc: {val_stats['acc_ing']:.2f}%, "
              f"Ves Acc: {val_stats['acc_ves']:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training curves
    plot_training_curves(history)
    
    return model, history

def train_epoch(model, train_loader, criterion, optimizer):
    """Run one training epoch"""
    running_loss = 0.0
    stats = {'correct_ing': 0, 'total_ing': 0, 'correct_ves': 0, 'total_ves': 0}
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, ing_labels, ves_labels, shape_features in progress_bar:
        # Prepare data
        images = images.to(device)
        shape_features = shape_features.to(device)
        valid_ing_mask = ing_labels >= 0
        valid_ves_mask = ves_labels >= 0
        
        # Forward pass
        optimizer.zero_grad()
        ing_logits, ves_logits = model(images, shape_features)
        
        # Compute loss
        loss = 0
        if valid_ing_mask.any():
            ing_loss = criterion(ing_logits[valid_ing_mask], 
                               ing_labels[valid_ing_mask].to(device))
            loss += ing_loss
        
        if valid_ves_mask.any():
            ves_loss = criterion(ves_logits[valid_ves_mask], 
                               ves_labels[valid_ves_mask].to(device))
            loss += ves_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        update_accuracy_stats(stats, ing_logits, ing_labels, valid_ing_mask,
                            ves_logits, ves_labels, valid_ves_mask)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'ing_acc': 100 * stats['correct_ing'] / max(1, stats['total_ing']),
            'ves_acc': 100 * stats['correct_ves'] / max(1, stats['total_ves'])
        })
    
    return {
        'loss': running_loss / len(train_loader),
        'acc_ing': 100 * stats['correct_ing'] / max(1, stats['total_ing']),
        'acc_ves': 100 * stats['correct_ves'] / max(1, stats['total_ves'])
    }

def validate_epoch(model, val_loader, criterion):
    """Run one validation epoch"""
    running_loss = 0.0
    stats = {'correct_ing': 0, 'total_ing': 0, 'correct_ves': 0, 'total_ves': 0}
    
    with torch.no_grad():
        for images, ing_labels, ves_labels, shape_features in val_loader:
            # Prepare data
            images = images.to(device)
            shape_features = shape_features.to(device)
            valid_ing_mask = ing_labels >= 0
            valid_ves_mask = ves_labels >= 0
            
            # Forward pass
            ing_logits, ves_logits = model(images, shape_features)
            
            # Compute loss
            loss = 0
            if valid_ing_mask.any():
                ing_loss = criterion(ing_logits[valid_ing_mask], 
                                   ing_labels[valid_ing_mask].to(device))
                loss += ing_loss
            
            if valid_ves_mask.any():
                ves_loss = criterion(ves_logits[valid_ves_mask], 
                                   ves_labels[valid_ves_mask].to(device))
                loss += ves_loss
            
            # Update statistics
            running_loss += loss.item()
            update_accuracy_stats(stats, ing_logits, ing_labels, valid_ing_mask,
                                ves_logits, ves_labels, valid_ves_mask)
    
    return {
        'loss': running_loss / len(val_loader),
        'acc_ing': 100 * stats['correct_ing'] / max(1, stats['total_ing']),
        'acc_ves': 100 * stats['correct_ves'] / max(1, stats['total_ves'])
    }

def update_accuracy_stats(stats, ing_logits, ing_labels, ing_mask, 
                         ves_logits, ves_labels, ves_mask):
    """Update accuracy statistics for both ingredient and vessel predictions"""
    if ing_mask.any():
        _, ing_preds = torch.max(ing_logits[ing_mask], 1)
        stats['correct_ing'] += (ing_preds == ing_labels[ing_mask].to(device)).sum().item()
        stats['total_ing'] += ing_mask.sum().item()
    
    if ves_mask.any():
        _, ves_preds = torch.max(ves_logits[ves_mask], 1)
        stats['correct_ves'] += (ves_preds == ves_labels[ves_mask].to(device)).sum().item()
        stats['total_ves'] += ves_mask.sum().item()

def plot_training_curves(history):
    """Plot training and validation curves"""
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Ingredient accuracy curves
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc_ing'], label='Train')
    plt.plot(history['val_acc_ing'], label='Validation')
    plt.title('Ingredient Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Vessel accuracy curves
    plt.subplot(2, 2, 3)
    plt.plot(history['train_acc_ves'], label='Train')
    plt.plot(history['val_acc_ves'], label='Validation')
    plt.title('Vessel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved as 'training_curves.png'")

# ============================
# Main Training Loop
# ============================

def main():
    """Main training function"""
    print("=== Food Vessel Classification with DINOv2 and Shape Features ===")
    
    # Create dataset
    dataset = FoodVesselDataset(INGREDIENTS_PATH, VESSELS_PATH, transform=train_transform)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 4  # Small batch size for M1 Mac
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    try:
        # Initialize model
        print("\nInitializing DINOv2 model...")
        dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        model = DualClassifier(
            dinov2_model=dinov2_model,
            num_ingredient_classes=len(dataset.ingredient_classes),
            num_vessel_classes=len(dataset.vessel_classes),
            shape_feature_dim=256,
            feature_dim=768  # Explicitly set feature dimension
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=30
        )
        
        # Save model with complete information
        save_path = 'food_vessel_classifier_dinov2.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'ingredient_classes': dataset.ingredient_classes,
            'vessel_classes': dataset.vessel_classes,
            'feature_dim': model.feature_dim,
            'shape_feature_dim': model.shape_feature_dim,
            'model_type': 'dinov2-base',
            'training_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'model_config': {
                'num_ingredient_classes': len(dataset.ingredient_classes),
                'num_vessel_classes': len(dataset.vessel_classes),
                'feature_dim': model.feature_dim,
                'shape_feature_dim': model.shape_feature_dim
            }
        }, save_path)
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {save_path}")
        print(f"Model configuration:")
        print(f"  - Ingredient classes: {len(dataset.ingredient_classes)}")
        print(f"  - Vessel classes: {len(dataset.vessel_classes)}")
        print(f"  - Feature dimension: {model.feature_dim}")
        print(f"  - Shape feature dimension: {model.shape_feature_dim}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 