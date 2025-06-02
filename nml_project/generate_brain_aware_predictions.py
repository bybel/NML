import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os

# Import components
from GATE import prepare_data_from_segments, EEGGraphAttentionNetwork, EEGFeatureExtractor  # Use original feature extractor
from brain_aware_graph import BrainAwareGraphConstructor

class BrainAwareEEGGraphConstructor:
    """Wrapper to integrate brain-aware constructor with existing pipeline."""
    
    def __init__(self, distances, channels, connectivity_strength='medium'):
        self.brain_constructor = BrainAwareGraphConstructor(distances, channels)
        self.connectivity_strength = connectivity_strength
        
    def create_edge_index_from_distances(self):
        return self.brain_constructor.create_neuroanatomical_edges(self.connectivity_strength)

def generate_brain_aware_submission():
    """
    Generate submission using neuroanatomically informed graph connectivity.
    """
    print("Generating brain-aware submission...")
    
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
    
    # Use original feature extractor to match saved model
    feature_extractor = EEGFeatureExtractor(sampling_rate=250)
    
    # Test with different connectivity strengths
    predictions_ensemble = []
    connectivity_strengths = ['sparse', 'medium', 'dense']
    
    for strength in connectivity_strengths:
        print(f"\nProcessing with {strength} connectivity...")
        
        # Create brain-aware graph constructor
        graph_constructor = BrainAwareEEGGraphConstructor(
            distances, channels, connectivity_strength=strength
        )
        
        # Analyze connectivity
        edge_index = graph_constructor.create_edge_index_from_distances()
        analysis = graph_constructor.brain_constructor.analyze_connectivity(edge_index)
        
        print(f"  Connectivity analysis:")
        print(f"    Edges: {analysis['num_edges']}")
        print(f"    Connectivity ratio: {analysis['connectivity_ratio']:.3f}")
        print(f"    Average clustering: {analysis['avg_clustering']:.3f}")
        print(f"    Connected components: {analysis['connected_components']}")
        print(f"    Is connected: {analysis['is_connected']}")
        
        if not analysis['is_connected']:
            print(f"  Warning: Graph is not fully connected for {strength} connectivity")
        
        # Skip if too sparse or too dense
        if analysis['connectivity_ratio'] < 0.05:
            print(f"  Skipping {strength} - too sparse")
            continue
        if analysis['connectivity_ratio'] > 0.8:
            print(f"  Skipping {strength} - too dense")
            continue
        
        # Prepare data with original feature extractor
        try:
            test_data_list = prepare_data_from_segments(
                test_segments,
                np.zeros(len(test_segments)),
                channels,
                feature_extractor,
                graph_constructor,
                data_path='./data/test',
                use_correlation_edges=False
            )
        except Exception as e:
            print(f"  Error preparing data for {strength}: {e}")
            continue
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_features = test_data_list[0].x.shape[1]
        print(f"  Number of features per node: {num_features}")
        
        # Use original model architecture that matches the saved weights
        model = EEGGraphAttentionNetwork(
            num_features=num_features,
            hidden_dim=256,
            num_heads=16,
            num_classes=2,
            dropout=0.2
        )
        
        try:
            model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
            print(f"  Model loaded successfully for {strength} connectivity!")
        except Exception as e:
            print(f"  Error loading model for {strength}: {e}")
            continue
        
        model = model.to(device)
        model.eval()
        
        # Generate predictions
        test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)
        
        probabilities = []
        print(f"  Generating predictions for {len(test_data_list)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"{strength} connectivity"):
                batch = batch.to(device)
                
                # Clean input
                if torch.isnan(batch.x).any():
                    batch.x = torch.nan_to_num(batch.x, nan=0.0)
                
                try:
                    # Get predictions
                    out = model(batch.x, batch.edge_index, batch.batch)
                    probs = F.softmax(out, dim=1)
                    
                    probabilities.extend(probs.cpu().numpy())
                except Exception as e:
                    print(f"  Error in prediction: {e}")
                    # Add dummy predictions
                    batch_size = batch.y.size(0)
                    dummy_probs = np.ones((batch_size, 2)) * 0.5
                    probabilities.extend(dummy_probs)
        
        if len(probabilities) > 0:
            predictions_ensemble.append(np.array(probabilities))
            print(f"  Completed {strength} connectivity - shape: {np.array(probabilities).shape}")
        else:
            print(f"  No valid predictions for {strength} connectivity")
    
    if len(predictions_ensemble) == 0:
        print("No valid predictions generated! Check model and data.")
        return None
    
    # Ensemble predictions with weighted averaging
    print(f"\nCombining {len(predictions_ensemble)} predictions...")
    
    # Weight different connectivity strengths
    if len(predictions_ensemble) == 3:
        weights = [0.25, 0.5, 0.25]  # Prefer medium connectivity
    elif len(predictions_ensemble) == 2:
        weights = [0.4, 0.6]  # Prefer the second one
    else:
        weights = [1.0]  # Single prediction
    
    print(f"Using ensemble weights: {weights}")
    
    # Weighted ensemble
    weighted_probs = np.zeros_like(predictions_ensemble[0])
    for i, probs in enumerate(predictions_ensemble):
        weighted_probs += weights[i] * probs
    
    final_predictions = np.argmax(weighted_probs, axis=1)
    confidence_scores = np.max(weighted_probs, axis=1)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_segments.index,
        'label': final_predictions
    })
    
    # Save submission
    submission_df.to_csv('submission_brain_aware.csv', index=False)
    
    print(f"\nBrain-aware submission created!")
    print(f"Total predictions: {len(final_predictions)}")
    print(f"Class 0: {(submission_df['label'] == 0).sum()}")
    print(f"Class 1: {(submission_df['label'] == 1).sum()}")
    print(f"Class distribution: {(submission_df['label'] == 1).sum() / len(submission_df):.3f}")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"Min confidence: {np.min(confidence_scores):.3f}")
    print(f"Max confidence: {np.max(confidence_scores):.3f}")
    print("Saved as 'submission_brain_aware.csv'")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'id': test_segments.index,
        'prediction': final_predictions,
        'confidence': confidence_scores,
        'prob_class_0': weighted_probs[:, 0],
        'prob_class_1': weighted_probs[:, 1]
    })
    analysis_df.to_csv('brain_aware_analysis.csv', index=False)
    print("Detailed analysis saved as 'brain_aware_analysis.csv'")
    
    return submission_df

if __name__ == "__main__":
    submission = generate_brain_aware_submission()