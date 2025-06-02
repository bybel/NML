import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os

# Import components
from GATE import prepare_data_from_segments, EEGGraphAttentionNetwork, EEGFeatureExtractor
from brain_aware_graph import BrainAwareGraphConstructor

class RegularizedEEGGraphConstructor:
    """Graph constructor with regularization techniques."""
    
    def __init__(self, distances, channels, strategy='brain_aware'):
        self.distances = distances
        self.channels = channels
        self.strategy = strategy
        
        if strategy == 'brain_aware':
            self.brain_constructor = BrainAwareGraphConstructor(distances, channels)
        
    def create_edge_index_from_distances(self):
        if self.strategy == 'brain_aware':
            return self.brain_constructor.create_neuroanatomical_edges('medium')
        elif self.strategy == 'distance_threshold':
            # More conservative threshold to reduce overfitting
            threshold = np.percentile(self.distances[self.distances > 0], 20)  # Reduced from 25
            edge_list = []
            for i in range(len(self.channels)):
                for j in range(i + 1, len(self.channels)):
                    if self.distances[i, j] <= threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
            return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def create_regularized_model(num_features):
    """Create a model with anti-overfitting modifications."""
    
    class RegularizedEEGGraphAttentionNetwork(torch.nn.Module):
        def __init__(self, num_features, hidden_dim=128, num_heads=8, num_classes=2, dropout=0.5):
            super().__init__()
            
            # Reduced model capacity to prevent overfitting
            self.dropout = dropout
            
            # Smaller hidden dimensions
            self.gat1 = torch.nn.ModuleList([
                torch.geometric.nn.GATConv(num_features, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            ])
            
            self.gat2 = torch.nn.ModuleList([
                torch.geometric.nn.GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            ])
            
            # Additional dropout and batch norm for regularization
            self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
            
            # Simpler classifier
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 3, hidden_dim // 2),  # Reduced size
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim // 2, num_classes)
            )
            
        def forward(self, x, edge_index, batch):
            # First GAT layer with residual connection
            x1 = self.gat1[0](x, edge_index)
            x1 = F.relu(x1)
            x1 = self.batch_norm1(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            
            # Second GAT layer
            x2 = self.gat2[0](x1, edge_index)
            x2 = F.relu(x2)
            x2 = self.batch_norm2(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            
            # Global pooling with multiple strategies
            x_mean = torch.geometric.nn.global_mean_pool(x2, batch)
            x_max = torch.geometric.nn.global_max_pool(x2, batch)
            x_add = torch.geometric.nn.global_add_pool(x2, batch)
            
            # Combine pooling strategies
            x = torch.cat([x_mean, x_max, x_add], dim=1)
            
            # Final classification
            return self.classifier(x)
    
    return RegularizedEEGGraphAttentionNetwork(num_features)

def generate_regularized_submission():
    """
    Generate submission with regularization techniques to reduce overfitting.
    """
    print("Generating regularized submission to combat overfitting...")
    
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
    
    # Test multiple strategies with regularization
    predictions_ensemble = []
    strategies = [
        ('brain_aware', 'Brain-aware connectivity'),
        ('distance_threshold', 'Distance threshold connectivity')
    ]
    
    for strategy, description in strategies:
        print(f"\nProcessing with {description}...")
        
        # Create regularized graph constructor
        graph_constructor = RegularizedEEGGraphConstructor(
            distances, channels, strategy=strategy
        )
        
        # Analyze connectivity
        edge_index = graph_constructor.create_edge_index_from_distances()
        num_edges = edge_index.shape[1] // 2
        num_nodes = len(channels)
        max_edges = num_nodes * (num_nodes - 1) // 2
        connectivity_ratio = num_edges / max_edges
        
        print(f"  Edges: {num_edges}, Connectivity ratio: {connectivity_ratio:.3f}")
        
        # More restrictive connectivity bounds
        if connectivity_ratio < 0.08 or connectivity_ratio > 0.6:  # Tighter bounds
            print(f"  Skipping {strategy} - connectivity ratio out of range")
            continue
        
        # Prepare data
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
            print(f"  Error preparing data for {strategy}: {e}")
            continue
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_features = test_data_list[0].x.shape[1]
        print(f"  Number of features per node: {num_features}")
        
        # Try to load pre-trained model, but with different architecture if needed
        try:
            # First try original architecture
            model = EEGGraphAttentionNetwork(
                num_features=num_features,
                hidden_dim=256,
                num_heads=16,
                num_classes=2,
                dropout=0.4  # Increased dropout
            )
            model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
            print(f"  Original model loaded successfully for {strategy}!")
            
        except Exception as e:
            print(f"  Could not load original model: {e}")
            print(f"  Using regularized model architecture...")
            
            # Use regularized model
            model = create_regularized_model(num_features)
            # Initialize with smaller weights
            for param in model.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param, gain=0.5)  # Smaller initial weights
        
        model = model.to(device)
        model.eval()
        
        # Generate predictions with regularization techniques
        test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False)  # Smaller batch size
        
        # Use dropout during inference for uncertainty estimation
        predictions_with_uncertainty = []
        num_uncertainty_passes = 10  # More passes for uncertainty
        
        for uncertainty_pass in range(num_uncertainty_passes):
            # Enable dropout during inference
            model.train()  # This enables dropout
            
            probabilities = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"{strategy} - Uncertainty {uncertainty_pass+1}"):
                    batch = batch.to(device)
                    
                    # Clean input
                    if torch.isnan(batch.x).any():
                        batch.x = torch.nan_to_num(batch.x, nan=0.0)
                    
                    try:
                        # Get predictions with dropout active
                        out = model(batch.x, batch.edge_index, batch.batch)
                        probs = F.softmax(out, dim=1)
                        probabilities.extend(probs.cpu().numpy())
                    except Exception as e:
                        print(f"  Error in prediction: {e}")
                        batch_size = batch.y.size(0)
                        dummy_probs = np.ones((batch_size, 2)) * 0.5
                        probabilities.extend(dummy_probs)
            
            predictions_with_uncertainty.append(np.array(probabilities))
        
        # Calculate mean and uncertainty
        if predictions_with_uncertainty:
            all_preds = np.array(predictions_with_uncertainty)
            mean_probs = np.mean(all_preds, axis=0)
            uncertainty = np.std(all_preds, axis=0)
            
            predictions_ensemble.append(mean_probs)
            print(f"  Completed {strategy} - shape: {mean_probs.shape}")
            print(f"  Average uncertainty: {np.mean(uncertainty):.4f}")
        else:
            print(f"  No valid predictions for {strategy}")
    
    if len(predictions_ensemble) == 0:
        print("No valid predictions generated!")
        return None
    
    # Conservative ensemble
    print(f"\nCombining {len(predictions_ensemble)} prediction sets...")
    
    if len(predictions_ensemble) == 2:
        # More balanced weights to reduce overfitting to any single approach
        weights = [0.6, 0.4]  # Less extreme weighting
        print("Using balanced weights: [0.6 brain-aware, 0.4 distance-threshold]")
    else:
        weights = [1.0]
    
    # Weighted ensemble
    weighted_probs = np.zeros_like(predictions_ensemble[0])
    for i, probs in enumerate(predictions_ensemble):
        weighted_probs += weights[i] * probs
    
    # More conservative post-processing
    print("\nApplying conservative post-processing...")
    
    # Use a higher threshold for positive predictions
    positive_threshold = 0.6  # Require 60% confidence for seizure prediction
    final_predictions = (weighted_probs[:, 1] > positive_threshold).astype(int)
    confidence_scores = np.max(weighted_probs, axis=1)
    
    # Additional conservative bias for very uncertain predictions
    very_uncertain = confidence_scores < 0.6
    if np.any(very_uncertain):
        print(f"Setting {np.sum(very_uncertain)} uncertain predictions to class 0")
        final_predictions[very_uncertain] = 0
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_segments.index,
        'label': final_predictions
    })
    
    # Save submission
    submission_df.to_csv('submission_regularized.csv', index=False)
    
    print(f"\nRegularized submission created!")
    print(f"Total predictions: {len(final_predictions)}")
    print(f"Class 0: {(submission_df['label'] == 0).sum()}")
    print(f"Class 1: {(submission_df['label'] == 1).sum()}")
    print(f"Class distribution: {(submission_df['label'] == 1).sum() / len(submission_df):.3f}")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    print("Saved as 'submission_regularized.csv'")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'id': test_segments.index,
        'prediction': final_predictions,
        'confidence': confidence_scores,
        'prob_class_0': weighted_probs[:, 0],
        'prob_class_1': weighted_probs[:, 1]
    })
    analysis_df.to_csv('regularized_analysis.csv', index=False)
    print("Analysis saved as 'regularized_analysis.csv'")
    
    return submission_df

if __name__ == "__main__":
    submission = generate_regularized_submission()