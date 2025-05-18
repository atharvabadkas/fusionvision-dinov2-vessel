import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import traceback
from PIL import Image
from skimage.feature import hog
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import cv2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
print(f"Using device: {device}")

# ============================
# Feature Extraction Functions
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
# Model Loading and Prediction
# ============================

def load_trained_model():
    """Load the trained dual classifier model"""
    # Check for model checkpoint
    checkpoint_paths = ['model_checkpoint.pth', 'food_vessel_classifier_dinov2.pth', 
                       'vessel_classifier.pth', 'dual_classifier.pth']
    
    checkpoint_path = next((path for path in checkpoint_paths if os.path.exists(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found. Tried: {', '.join(checkpoint_paths)}")
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    try:
        # Load model components
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        feature_dim = model_config.get('feature_dim', checkpoint.get('feature_dim', 768))
        shape_feature_dim = model_config.get('shape_feature_dim', checkpoint.get('shape_feature_dim', 256))
        
        # Get class information
        vessel_classes = checkpoint['vessel_classes']
        ingredient_classes = checkpoint.get('ingredient_classes', [])
        
        print("\nModel Configuration:")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Shape feature dimension: {shape_feature_dim}")
        print(f"  - Vessel classes: {len(vessel_classes)}")
        print(f"  - Ingredient classes: {len(ingredient_classes)}")
        
        # Initialize model
        print("\nInitializing DINOv2 model...")
        dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base")
        model = DualClassifier(
            dinov2_model=dinov2_model,
            num_ingredient_classes=len(ingredient_classes),
            num_vessel_classes=len(vessel_classes),
            shape_feature_dim=shape_feature_dim,
            feature_dim=feature_dim
        )
        
        # Load weights and set to eval mode
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"\nModel loaded successfully!")
        return model, vessel_classes, ingredient_classes
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise

def predict_image(image_path, model, processor, ingredient_classes, vessel_classes):
    """Predict ingredient and vessel classes for an image using ensemble approach"""
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create multiple crops for ensemble prediction
    crops = [original_image.resize((224, 224))]
    width, height = original_image.size
    crop_size = min(width, height)
    crops.extend([
        original_image.crop((0, 0, crop_size, crop_size)).resize((224, 224)),
        original_image.crop((width - crop_size, 0, width, crop_size)).resize((224, 224)),
        original_image.crop((0, height - crop_size, crop_size, height)).resize((224, 224)),
        original_image.crop((width - crop_size, height - crop_size, width, height)).resize((224, 224))
    ])
    
    # Initialize prediction accumulators
    vessel_probs_avg = torch.zeros(len(vessel_classes)).to(device)
    ingredient_probs_avg = torch.zeros(len(ingredient_classes)).to(device) if ingredient_classes else None
    
    # Extract shape features
    shape_features = extract_shape_features(original_image)
    shape_features_tensor = torch.tensor(shape_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Process each crop
    for crop in crops:
        inputs = processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            ingredient_logits, vessel_logits = model(inputs['pixel_values'], shape_features_tensor)
            vessel_probs_avg += torch.softmax(vessel_logits, dim=1).squeeze(0)
            if ingredient_classes:
                ingredient_probs_avg += torch.softmax(ingredient_logits, dim=1).squeeze(0)
    
    # Average predictions
    vessel_probs_avg /= len(crops)
    if ingredient_probs_avg is not None:
        ingredient_probs_avg /= len(crops)
    
    # Get top predictions
    results = {'vessel': get_top_predictions(vessel_probs_avg, vessel_classes)}
    if ingredient_classes:
        results['ingredient'] = get_top_predictions(ingredient_probs_avg, ingredient_classes)
    
    # Display results
    print_predictions(results)
    display_analysis(original_image, shape_features, results, image_path, ingredient_classes is not None)
    
    return results

def get_top_predictions(probs, classes, top_k=5):
    """Get top-k predictions with their confidence scores"""
    values, indices = torch.topk(probs, k=min(top_k, len(classes)))
    return [(classes[idx], value.item() * 100) for idx, value in zip(indices, values)]

def print_predictions(results):
    """Print prediction results"""
    print("\nPredictions:")
    if 'ingredient' in results:
        print("Top ingredient predictions:")
        for ingredient, confidence in results['ingredient']:
            print(f"  {ingredient}: {confidence:.2f}%")
    
    print("\nTop vessel predictions:")
    for vessel, confidence in results['vessel']:
        print(f"  {vessel}: {confidence:.2f}%")

def display_analysis(image, shape_features, results, image_path, has_ingredients=False):
    """Create comprehensive visualization of analysis and predictions"""
    # Process image for visualization
    image_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Generate analysis images
    edges_canny = cv2.Canny(blurred, 30, 150)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.normalize(cv2.magnitude(sobel_x, sobel_y), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Create visualization
    fig_size = (20, 10) if has_ingredients else (20, 5)
    plt.figure(figsize=fig_size)
    
    # Plot feature analysis
    plt.subplot(2 if has_ingredients else 1, 5, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2 if has_ingredients else 1, 5, 2)
    plt.title('Edge Detection')
    plt.imshow(edges_canny, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2 if has_ingredients else 1, 5, 3)
    plt.title('Gradient Analysis')
    plt.imshow(sobel_combined, cmap='gray')
    plt.axis('off')
    
    # Plot predictions
    plot_predictions(results, has_ingredients, 2 if has_ingredients else 1, 5, 4)
    
    plt.tight_layout()
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f'{base_name}_analysis.png')
    print(f"Analysis saved as {base_name}_analysis.png")

def plot_predictions(results, has_ingredients, rows, cols, start_pos):
    """Plot prediction bar charts"""
    if has_ingredients:
        plt.subplot(rows, cols, start_pos)
        plot_class_predictions(results['ingredient'], 'Ingredient Predictions', 'lightblue')
        
        plt.subplot(rows, cols, start_pos + 1)
        plot_class_predictions(results['vessel'], 'Vessel Predictions', 'lightgreen')
    else:
        plt.subplot(rows, cols, start_pos)
        plot_class_predictions(results['vessel'], 'Vessel Predictions', 'lightgreen')

def plot_class_predictions(predictions, title, color):
    """Plot individual prediction bar chart"""
    names, values = zip(*sorted(predictions, key=lambda x: x[1], reverse=True))
    bars = plt.barh(names, values, color=color)
    plt.xlabel('Confidence (%)')
    plt.title(title)
    plt.xlim(0, 100)
    
    for bar, val in zip(bars, values):
        plt.text(min(val + 1, 95), bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center')

def main():
    """Main function to process test images"""
    image_path = "/Users/atharvabadkas/Coding /DINO/dino_feature_extraction_model/20250303/DT20250303_TM072658_MCD83BDA894D90_WT12467_TC40_TX40_RN749_DW12175.jpg"
    
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model, vessel_classes, ingredient_classes = load_trained_model()
        results = predict_image(image_path, model, processor, ingredient_classes, vessel_classes)
        print("\nPrediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 