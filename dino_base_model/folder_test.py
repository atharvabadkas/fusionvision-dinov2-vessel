import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from transformers import ViTImageProcessor, ViTModel
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
import time
from pathlib import Path

# Check if MPS is available (for M1 Mac)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define transformations for testing (no augmentations)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Residual block for better gradient flow (same as in training)
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

# Define the classifier model (same as in training)
class DualClassifier(nn.Module):
    def __init__(self, vit_model, feature_dim, num_ingredient_classes, num_vessel_classes):
        super(DualClassifier, self).__init__()
        self.vit_model = vit_model
        self.feature_dim = feature_dim
        
        # Freeze ViT model parameters
        for param in self.vit_model.parameters():
            param.requires_grad = False
            
        # Deeper ingredient classifier with residual connections
        self.ingredient_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(1024, 768),
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
            ResidualBlock(1024, 768),
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

# Load the trained model
def load_trained_model(model_path):
    # Load ViT model
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    vit_model = ViTModel.from_pretrained("google/vit-large-patch16-224").to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get classes and feature dimension
    ingredient_classes = checkpoint['ingredient_classes']
    vessel_classes = checkpoint['vessel_classes']
    feature_dim = checkpoint['feature_dim']
    
    # Create model
    model = DualClassifier(
        vit_model=vit_model,
        feature_dim=feature_dim,
        num_ingredient_classes=len(ingredient_classes),
        num_vessel_classes=len(vessel_classes)
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print model information
    print("\n===== Model Information =====")
    print(f"Total Classes: {len(ingredient_classes) + len(vessel_classes)}")
    print(f"- Ingredient Classes: {len(ingredient_classes)}")
    print(f"- Vessel Classes: {len(vessel_classes)}")
    
    return model, processor, ingredient_classes, vessel_classes

# Enhanced prediction function with ensemble approach
def predict_image(image_path, model, processor, ingredient_classes, vessel_classes):
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None, None, None
    
    # Create multiple crops of the image for ensemble prediction
    width, height = image.size
    crops = []
    
    # Center crop
    crops.append(image)
    
    # Corner crops
    if width > 300 and height > 300:
        crops.append(image.crop((0, 0, width*0.8, height*0.8)))  # Top-left
        crops.append(image.crop((width*0.2, 0, width, height*0.8)))  # Top-right
        crops.append(image.crop((0, height*0.2, width*0.8, height)))  # Bottom-left
        crops.append(image.crop((width*0.2, height*0.2, width, height)))  # Bottom-right
    
    # Apply transformations to all crops
    crop_tensors = [test_transform(crop).unsqueeze(0).to(device) for crop in crops]
    
    # Get predictions for all crops
    ingredient_probs_avg = torch.zeros(len(ingredient_classes)).to(device)
    vessel_probs_avg = torch.zeros(len(vessel_classes)).to(device)
    
    with torch.no_grad():
        for crop_tensor in crop_tensors:
            ingredient_logits, vessel_logits = model(crop_tensor)
            
            # Apply softmax to get probabilities
            ingredient_probs = torch.nn.functional.softmax(ingredient_logits, dim=1)[0]
            vessel_probs = torch.nn.functional.softmax(vessel_logits, dim=1)[0]
            
            # Accumulate probabilities
            ingredient_probs_avg += ingredient_probs
            vessel_probs_avg += vessel_probs
    
    # Average the probabilities
    ingredient_probs_avg /= len(crop_tensors)
    vessel_probs_avg /= len(crop_tensors)
    
    # Get top 3 predictions for ingredients
    top3_ingredient_values, top3_ingredient_indices = torch.topk(ingredient_probs_avg, 3)
    top3_ingredients = [(ingredient_classes[idx], val.item()) for idx, val in zip(top3_ingredient_indices, top3_ingredient_values)]
    
    # Get top 3 predictions for vessels
    top3_vessel_values, top3_vessel_indices = torch.topk(vessel_probs_avg, 3)
    top3_vessels = [(vessel_classes[idx], val.item()) for idx, val in zip(top3_vessel_indices, top3_vessel_values)]
    
    # Get final predictions (top 1)
    ingredient_pred = ingredient_classes[ingredient_probs_avg.argmax().item()]
    vessel_pred = vessel_classes[vessel_probs_avg.argmax().item()]
    
    return top3_ingredients, top3_vessels, ingredient_pred, vessel_pred

def get_sorted_image_files(folder_path):
    """Get all image files in the folder and subfolders, sorted by folder structure and filename"""
    image_files = []
    
    # First, collect all image files with their full paths
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('.'):
                image_files.append(os.path.join(root, file))
    
    # Sort by directory first, then by filename
    # This ensures files in the same directory stay together and are sorted by name
    image_files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))
    
    return image_files

def process_folder(folder_path, model, processor, ingredient_classes, vessel_classes, output_dir=None):
    # Normalize folder path
    folder_path = os.path.normpath(os.path.expanduser(folder_path))
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Create output directory if specified (only for visualization)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the folder and subfolders, sorted by folder structure
    image_files = get_sorted_image_files(folder_path)
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Initialize counters and results storage
    results = []
    ingredient_correct = 0
    vessel_correct = 0
    total_images = 0
    ingredient_predictions = Counter()
    vessel_predictions = Counter()
    
    # Process each image with a progress bar
    for image_path in tqdm(image_files, desc="Processing images"):
        # Extract ground truth from folder structure (if available)
        try:
            # For ingredient images: /path/to/ingredients/ingredient_name/image.jpg
            # For vessel images: /path/to/vessels/vessel_name/image.jpg
            parts = Path(image_path).parts
            
            # Try to determine if this is an ingredient or vessel image
            if "ingredient" in image_path.lower():
                true_ingredient = parts[-2]  # Parent folder name
                true_vessel = None
            elif "vessel" in image_path.lower():
                true_ingredient = None
                true_vessel = parts[-2]  # Parent folder name
            else:
                # If can't determine, use parent folder name as ingredient (most common case)
                true_ingredient = parts[-2]
                true_vessel = None
        except:
            true_ingredient = None
            true_vessel = None
        
        # Predict
        top3_ingredients, top3_vessels, ingredient_pred, vessel_pred = predict_image(
            image_path, model, processor, ingredient_classes, vessel_classes
        )
        
        if top3_ingredients is None:  # Skip failed images
            continue
        
        total_images += 1
        
        # Check if predictions match ground truth (if available)
        ingredient_match = (true_ingredient is not None and true_ingredient.lower() == ingredient_pred.lower())
        vessel_match = (true_vessel is not None and true_vessel.lower() == vessel_pred.lower())
        
        if ingredient_match:
            ingredient_correct += 1
        
        if vessel_match:
            vessel_correct += 1
        
        # Count predictions
        ingredient_predictions[ingredient_pred] += 1
        vessel_predictions[vessel_pred] += 1
        
        # Store results
        result = {
            'image_path': image_path,
            'predicted_ingredient': ingredient_pred,
            'predicted_vessel': vessel_pred,
            'ingredient_confidence': top3_ingredients[0][1] * 100,
            'vessel_confidence': top3_vessels[0][1] * 100
        }
        results.append(result)
    
    # Calculate overall statistics
    if total_images > 0:
        ingredient_accuracy = (ingredient_correct / total_images) * 100 if ingredient_correct > 0 else 0
        vessel_accuracy = (vessel_correct / total_images) * 100 if vessel_correct > 0 else 0
        
        print("\n===== Folder Prediction Results =====")
        print(f"Total images processed: {total_images}")
        
        # Most common predictions
        print("\nTop 5 Ingredient Predictions:")
        for ingredient, count in ingredient_predictions.most_common(5):
            print(f"  {ingredient}: {count} images ({count/total_images*100:.1f}%)")
        
        print("\nTop 5 Vessel Predictions:")
        for vessel, count in vessel_predictions.most_common(5):
            print(f"  {vessel}: {count} images ({count/total_images*100:.1f}%)")
        
        # Print simplified results in the terminal
        print("\n===== Predictions for Each Image =====")
        for i, result in enumerate(results):
            # Get relative path from the base folder for cleaner output
            try:
                rel_path = os.path.relpath(result['image_path'], folder_path)
            except:
                rel_path = os.path.basename(result['image_path'])
                
            print(f"\nImage {i+1}/{len(results)}: {rel_path}")
            print(f"1. Ingredient Prediction: {result['predicted_ingredient']} ({result['ingredient_confidence']:.2f}%)")
            print(f"2. Vessel Prediction: {result['predicted_vessel']} ({result['vessel_confidence']:.2f}%)")
        
        # Create summary visualizations if output directory is specified
        if output_dir:
            # Create summary visualizations
            plt.figure(figsize=(15, 10))
            
            # Ingredient distribution
            plt.subplot(2, 1, 1)
            top_ingredients = dict(ingredient_predictions.most_common(10))
            plt.bar(top_ingredients.keys(), top_ingredients.values(), color='skyblue')
            plt.title('Top 10 Ingredient Predictions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Vessel distribution
            plt.subplot(2, 1, 2)
            top_vessels = dict(vessel_predictions.most_common(10))
            plt.bar(top_vessels.keys(), top_vessels.values(), color='lightgreen')
            plt.title('Top 10 Vessel Predictions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
            print(f"\nPrediction distribution visualization saved to {os.path.join(output_dir, 'prediction_distribution.png')}")

def main():
    # Hardcoded paths (similar to model_test.py)
    folder_path = "/Users/atharvabadkas/Coding /DINO/20250301"
    model_path = "food_vessel_classifier.pth"
    output_dir = "folder_test_results"  # Only used for visualization now
    
    # Normalize folder path
    folder_path = os.path.normpath(os.path.expanduser(folder_path))
    
    print(f"Processing folder: {folder_path}")
    print(f"Using model: {model_path}")
    
    # Load model
    model, processor, ingredient_classes, vessel_classes = load_trained_model(model_path)
    
    # Process folder
    start_time = time.time()
    process_folder(
        folder_path, 
        model, 
        processor, 
        ingredient_classes, 
        vessel_classes, 
        output_dir
    )
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 