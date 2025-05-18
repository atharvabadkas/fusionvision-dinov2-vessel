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
    print(f"1. Total Classes: {len(ingredient_classes) + len(vessel_classes)}")
    print(f"   - Ingredient Classes: {len(ingredient_classes)}")
    print(f"   - Vessel Classes: {len(vessel_classes)}")
    
    # Calculate total images trained (based on expected counts)
    total_ingredient_images = sum([15 for _ in ingredient_classes])  # Approximately 15 per class
    total_vessel_images = sum([10 for _ in vessel_classes])  # Approximately 10 per class
    total_images = total_ingredient_images + total_vessel_images
    print(f"2. Total Images trained: {total_images}")
    print(f"   - Ingredient Images: {total_ingredient_images}")
    print(f"   - Vessel Images: {total_vessel_images}")
    
    return model, processor, ingredient_classes, vessel_classes

# Enhanced prediction function with ensemble approach
def predict_image(image_path, model, processor, ingredient_classes, vessel_classes):
    # Check if image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    print(f"\nProcessing image: {image_path}")
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise Exception(f"Error opening image: {e}")
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    plt.savefig('input_image.png')
    
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
    
    # Print predictions
    print("\n===== Prediction Results =====")
    print("3. Predicted Ingredient (Top 3):")
    for i, (name, prob) in enumerate(top3_ingredients):
        print(f"   {i+1}. {name}, Accuracy: {prob*100:.2f}%")
    
    print("\n4. Predicted Vessel (Top 3):")
    for i, (name, prob) in enumerate(top3_vessels):
        print(f"   {i+1}. {name}, Accuracy: {prob*100:.2f}%")
    
    print("\n5. Final Prediction:")
    print(f"   Ingredient Name: {ingredient_pred}")
    print(f"   Vessel Name: {vessel_pred}")
    
    return top3_ingredients, top3_vessels, ingredient_pred, vessel_pred

def main():
    # Hardcoded image path - CHANGE THIS TO YOUR IMAGE PATH
    image_path = "/Users/atharvabadkas/Coding /DINO/verandah_prep_ingredients/lasuni palak paneer/DT20241112_TM182454_MCD83BDA89443C_WT1240_TC40_TX37_RN577.jpg"
    
    # Hardcoded model path
    model_path = "food_vessel_classifier.pth"
    
    # Normalize path (handle spaces and special characters)
    image_path = os.path.normpath(os.path.expanduser(image_path))
    
    # Load model
    model, processor, ingredient_classes, vessel_classes = load_trained_model(model_path)
    
    # Predict
    try:
        top3_ingredients, top3_vessels, ingredient_pred, vessel_pred = predict_image(
            image_path, model, processor, ingredient_classes, vessel_classes
        )
        
        # Create visualization of top predictions
        plt.figure(figsize=(12, 6))
        
        # Plot ingredient predictions
        plt.subplot(1, 2, 1)
        names = [name for name, _ in top3_ingredients]
        values = [val*100 for _, val in top3_ingredients]
        plt.barh(names, values, color='skyblue')
        plt.xlabel('Confidence (%)')
        plt.title('Top 3 Ingredient Predictions')
        plt.xlim(0, 100)
        
        # Plot vessel predictions
        plt.subplot(1, 2, 2)
        names = [name for name, _ in top3_vessels]
        values = [val*100 for _, val in top3_vessels]
        plt.barh(names, values, color='lightgreen')
        plt.xlabel('Confidence (%)')
        plt.title('Top 3 Vessel Predictions')
        plt.xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png')
        print("\nPrediction visualization saved to 'prediction_results.png'")
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main() 