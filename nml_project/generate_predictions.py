import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os

# Import from your GATE.py
from GATE import (
    EEGGraphAttentionNetwork, 
    EEGFeatureExtractor, 
    EEGGraphConstructor,
    prepare_data_from_segments
)

def generate_submission():
    """
    Generate submission file in the correct format.
    """
    print("Generating submission for GAT model...")
    
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
    
    # Initialize components
    feature_extractor = EEGFeatureExtractor(sampling_rate=250)
    graph_constructor = EEGGraphConstructor(distances, threshold_percentile=30)
    
    # Prepare test data
    print("Preparing test data...")
    dummy_labels = np.zeros(len(test_segments))
    
    # Fit feature extractor on small sample
    sample_data = prepare_data_from_segments(
        test_segments.head(20),
        dummy_labels[:20],
        channels,
        feature_extractor,
        graph_constructor,
        data_path='./data/test',
        use_correlation_edges=False
    )
    
    # Prepare all test data
    test_data_list = prepare_data_from_segments(
        test_segments,
        dummy_labels,
        channels,
        feature_extractor,
        graph_constructor,
        data_path='./data/test',
        use_correlation_edges=False
    )
    
    print(f"Prepared {len(test_data_list)} test samples")
    
    # Load trained model - MATCH THE ARCHITECTURE FROM TRAINING
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = test_data_list[0].x.shape[1]
    
    # Use the SAME architecture as the full model training
    model = EEGGraphAttentionNetwork(
        num_features=num_features,
        hidden_dim=256,  # Changed from 128 to 256
        num_heads=16,    # Changed from 8 to 16
        num_classes=2,
        dropout=0.2      # Changed from 0.3 to 0.2
    )
    
    # Load weights
    model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Generate predictions
    print("Generating predictions...")
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            
            # Clean input
            if torch.isnan(batch.x).any():
                batch.x = torch.nan_to_num(batch.x, nan=0.0)
            
            # Get predictions
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    # Create submission DataFrame
    # Use the index as ID (matching the expected format)
    submission_df = pd.DataFrame({
        'id': test_segments.index,
        'label': predictions
    })
    
    # Save submission
    submission_df.to_csv('submission_gat.csv', index=False)
    
    print(f"\nSubmission created!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Class 0: {(submission_df['label'] == 0).sum()}")
    print(f"Class 1: {(submission_df['label'] == 1).sum()}")
    print("Saved as 'submission_gat.csv'")
    
    # Show sample
    print("\nFirst 10 predictions:")
    print(submission_df.head(10))
    
    return submission_df

if __name__ == "__main__":
    if not os.path.exists('best_gat_model.pth'):
        print("Error: Model file 'best_gat_model.pth' not found!")
        print("Please train the model first by running GATE.py")
    else:
        submission = generate_submission()