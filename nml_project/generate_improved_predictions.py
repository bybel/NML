import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os
from sklearn.utils.class_weight import compute_class_weight

# Import components
from GATE import prepare_data_from_segments, EEGGraphAttentionNetwork, EEGFeatureExtractor
from brain_aware_graph import BrainAwareGraphConstructor

class SeizureFocusedEEGGraphConstructor:
    """Graph constructor specifically tuned for seizure detection."""
    
    def __init__(self, distances, channels):
        self.distances = distances
        self.channels = channels
        self.brain_constructor = BrainAwareGraphConstructor(distances, channels)
        
    def create_edge_index_from_distances(self):
        # Use sparser connectivity to focus on stronger connections
        return self.brain_constructor.create_neuroanatomical_edges('sparse')

def focal_loss(pred, target, alpha=1.0, gamma=3.0):
    """
    Focal loss to handle extreme class imbalance.
    Focuses learning on hard examples (seizures).
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    
    # Higher alpha for minority class (seizures)
    alpha_tensor = torch.ones(pred.size(0)).to(pred.device)
    alpha_tensor[target == 1] = alpha  # Much higher weight for seizures
    
    focal_loss = alpha_tensor * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def generate_seizure_focused_submission():
    """
    Generate submission with extreme focus on seizure detection.
    """
    print("Generating SEIZURE-FOCUSED submission...")
    print("Priority: Catch seizures even if it means more false positives!")
    
    # Load test data
    test_segments = pq.read_table('./data/test/segments.parquet').to_pandas()
    print(f"Loaded {len(test_segments)} test segments")
    
    # Load distances and setup model components
    distances_df = pd.read_csv('./data/distances_3d.csv')
    channels = sorted(list(set(distances_df['from'].unique().tolist() + distances_df['to'].unique().tolist())))
    
    # Create distance matrix
    distances = np.zeros((len(channels), len(channels)))
    channel_to_idx = {ch: idx for idx, ch in enumerate(channels)}
    
    for _, row in distances_df.iterrows():
        i = channel_to_idx[row['from']]
        j = channel_to_idx[row['to']]
        distances[i, j] = row['distance']
        distances[j, i] = row['distance']
    
    print(f"Channels: {channels}")
    print(f"Number of channels: {len(channels)}")
    
    # Use original feature extractor
    feature_extractor = EEGFeatureExtractor(sampling_rate=250)
    
    # Create seizure-focused graph constructor
    graph_constructor = SeizureFocusedEEGGraphConstructor(distances, channels)
    
    # Prepare data
    print("Preparing test data with seizure-focused approach...")
    test_data_list = prepare_data_from_segments(
        test_segments,
        np.zeros(len(test_segments)),
        channels,
        feature_extractor,
        graph_constructor,
        data_path='./data/test',
        use_correlation_edges=False
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = test_data_list[0].x.shape[1]
    print(f"Number of features per node: {num_features}")
    
    # Load original model
    model = EEGGraphAttentionNetwork(
        num_features=num_features,
        hidden_dim=256,
        num_heads=16,
        num_classes=2,
        dropout=0.2
    )
    
    model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generate predictions with AGGRESSIVE seizure detection
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)
    
    # Multiple strategies for seizure detection
    all_probabilities = []
    
    # Strategy 1: Regular prediction
    probabilities_regular = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Regular prediction"):
            batch = batch.to(device)
            
            if torch.isnan(batch.x).any():
                batch.x = torch.nan_to_num(batch.x, nan=0.0)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            probabilities_regular.extend(probs.cpu().numpy())
    
    # Strategy 2: Temperature scaling to make predictions less confident
    # This can help us catch more seizures by lowering the threshold
    temperature = 2.0  # Higher temperature = less confident predictions
    probabilities_temp = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Temperature scaled prediction"):
            batch = batch.to(device)
            
            if torch.isnan(batch.x).any():
                batch.x = torch.nan_to_num(batch.x, nan=0.0)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            # Apply temperature scaling
            out_scaled = out / temperature
            probs = F.softmax(out_scaled, dim=1)
            probabilities_temp.extend(probs.cpu().numpy())
    
    # Strategy 3: Dropout enabled for uncertainty estimation
    model.train()  # Enable dropout
    mc_predictions = []
    for mc_pass in range(5):
        probabilities_mc = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"MC Dropout pass {mc_pass+1}"):
                batch = batch.to(device)
                
                if torch.isnan(batch.x).any():
                    batch.x = torch.nan_to_num(batch.x, nan=0.0)
                
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=1)
                probabilities_mc.extend(probs.cpu().numpy())
        mc_predictions.append(np.array(probabilities_mc))
    
    # Average MC predictions
    probabilities_mc_avg = np.mean(mc_predictions, axis=0)
    
    # Combine all strategies with seizure-focused weighting
    probabilities_regular = np.array(probabilities_regular)
    probabilities_temp = np.array(probabilities_temp)
    
    print("\nCombining prediction strategies...")
    
    # Weighted combination favoring methods that might catch more seizures
    combined_probs = (
        0.3 * probabilities_regular +     # Regular prediction
        0.4 * probabilities_temp +        # Temperature scaled (less confident)
        0.3 * probabilities_mc_avg        # MC dropout (uncertainty)
    )
    
    # AGGRESSIVE thresholding for seizure detection
    print("\nApplying seizure-focused decision rules...")
    
    # Much lower threshold for predicting seizures
    seizure_threshold = 0.3  # If >30% chance of seizure, predict seizure
    
    # Additional rules to catch more seizures
    final_predictions = np.zeros(len(combined_probs), dtype=int)
    
    for i in range(len(combined_probs)):
        prob_seizure = combined_probs[i, 1]
        
        # Rule 1: Low threshold
        if prob_seizure > seizure_threshold:
            final_predictions[i] = 1
        
        # Rule 2: If any individual method is confident about seizure
        elif (probabilities_regular[i, 1] > 0.4 or 
              probabilities_temp[i, 1] > 0.4 or 
              probabilities_mc_avg[i, 1] > 0.4):
            final_predictions[i] = 1
        
        # Rule 3: High uncertainty might indicate seizure
        elif np.std([p[i, 1] for p in mc_predictions]) > 0.2:
            final_predictions[i] = 1
        
        # Rule 4: Conservative - prefer false positives over false negatives
        elif prob_seizure > 0.25:  # Even lower threshold
            final_predictions[i] = 1
        
        else:
            final_predictions[i] = 0
    
    # Calculate some stats
    confidence_scores = np.max(combined_probs, axis=1)
    seizure_confidences = combined_probs[:, 1]
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_segments.index,
        'label': final_predictions
    })
    
    # Save submission
    submission_df.to_csv('submission_seizure_focused.csv', index=False)
    
    print(f"\n🚨 SEIZURE-FOCUSED SUBMISSION CREATED! 🚨")
    print(f"Total predictions: {len(final_predictions)}")
    print(f"Class 0 (Non-seizure): {(submission_df['label'] == 0).sum()}")
    print(f"Class 1 (SEIZURE): {(submission_df['label'] == 1).sum()}")
    print(f"Seizure detection rate: {(submission_df['label'] == 1).sum() / len(submission_df):.3f}")
    print(f"Average seizure confidence: {np.mean(seizure_confidences):.3f}")
    print("Saved as 'submission_seizure_focused.csv'")
    
    # More aggressive version
    print("\n" + "="*50)
    print("Creating ULTRA-AGGRESSIVE seizure detection...")
    
    # Even more aggressive thresholding
    ultra_aggressive_predictions = np.zeros(len(combined_probs), dtype=int)
    
    for i in range(len(combined_probs)):
        prob_seizure = combined_probs[i, 1]
        
        # Ultra-low threshold
        if prob_seizure > 0.2:  # 20% threshold
            ultra_aggressive_predictions[i] = 1
        # If any uncertainty or any method suggests seizure possibility
        elif (np.max([p[i, 1] for p in mc_predictions]) > 0.3 or
              np.std([p[i, 1] for p in mc_predictions]) > 0.15):
            ultra_aggressive_predictions[i] = 1
        else:
            ultra_aggressive_predictions[i] = 0
    
    # Create ultra-aggressive submission
    submission_ultra_df = pd.DataFrame({
        'id': test_segments.index,
        'label': ultra_aggressive_predictions
    })
    
    submission_ultra_df.to_csv('submission_ultra_seizure_focused.csv', index=False)
    
    print(f"ULTRA-AGGRESSIVE SUBMISSION:")
    print(f"Class 0 (Non-seizure): {(submission_ultra_df['label'] == 0).sum()}")
    print(f"Class 1 (SEIZURE): {(submission_ultra_df['label'] == 1).sum()}")
    print(f"Seizure detection rate: {(submission_ultra_df['label'] == 1).sum() / len(submission_ultra_df):.3f}")
    print("Saved as 'submission_ultra_seizure_focused.csv'")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'id': test_segments.index,
        'regular_prediction': final_predictions,
        'ultra_prediction': ultra_aggressive_predictions,
        'prob_seizure': seizure_confidences,
        'confidence': confidence_scores,
        'regular_prob_0': probabilities_regular[:, 0],
        'regular_prob_1': probabilities_regular[:, 1],
        'temp_prob_0': probabilities_temp[:, 0],
        'temp_prob_1': probabilities_temp[:, 1],
        'mc_prob_0': probabilities_mc_avg[:, 0],
        'mc_prob_1': probabilities_mc_avg[:, 1]
    })
    analysis_df.to_csv('seizure_focused_analysis.csv', index=False)
    print("Detailed analysis saved as 'seizure_focused_analysis.csv'")
    
    return submission_df, submission_ultra_df

if __name__ == "__main__":
    submission_regular, submission_ultra = generate_seizure_focused_submission()