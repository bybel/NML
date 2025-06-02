import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

# Import components
from GATE import prepare_data_from_segments, EEGGraphAttentionNetwork, EEGFeatureExtractor
from brain_aware_graph import BrainAwareGraphConstructor

class UltraSeizureFocusedConstructor:
    """Ultra-focused seizure detection graph constructor."""
    
    def __init__(self, distances, channels):
        self.distances = distances
        self.channels = channels
        self.brain_constructor = BrainAwareGraphConstructor(distances, channels)
        
    def create_edge_index_from_distances(self):
        # Try multiple connectivity patterns and use the most seizure-sensitive
        sparse_edges = self.brain_constructor.create_neuroanatomical_edges('sparse')
        medium_edges = self.brain_constructor.create_neuroanatomical_edges('medium')
        
        # For now, return medium (but we'll use both in ensemble)
        return medium_edges

def advanced_seizure_submission():
    """
    Advanced seizure-focused submission with multiple sophisticated techniques.
    """
    print("🧠 ADVANCED SEIZURE DETECTION SYSTEM 🧠")
    print("Target: >90% seizure recall!")
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Strategy 1: Multiple Graph Topologies
    ensemble_predictions = []
    graph_strategies = ['sparse', 'medium', 'dense']
    
    for strategy in graph_strategies:
        print(f"\n🔄 Processing with {strategy} connectivity...")
        
        # Create graph constructor
        class StrategyConstructor:
            def __init__(self, distances, channels, strategy):
                self.brain_constructor = BrainAwareGraphConstructor(distances, channels)
                self.strategy = strategy
            
            def create_edge_index_from_distances(self):
                return self.brain_constructor.create_neuroanatomical_edges(self.strategy)
        
        graph_constructor = StrategyConstructor(distances, channels, strategy)
        
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
            print(f"Error with {strategy}: {e}")
            continue
        
        # Load model
        num_features = test_data_list[0].x.shape[1]
        model = EEGGraphAttentionNetwork(
            num_features=num_features,
            hidden_dim=256,
            num_heads=16,
            num_classes=2,
            dropout=0.2
        )
        
        model.load_state_dict(torch.load('best_gat_model.pth', map_location=device))
        model = model.to(device)
        
        # Generate predictions with multiple techniques
        strategy_predictions = []
        test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)
        
        # Technique 1: Regular inference
        model.eval()
        probs_regular = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"{strategy} - Regular"):
                batch = batch.to(device)
                if torch.isnan(batch.x).any():
                    batch.x = torch.nan_to_num(batch.x, nan=0.0)
                
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=1)
                probs_regular.extend(probs.cpu().numpy())
        
        # Technique 2: Multiple temperature scaling
        temperatures = [0.8, 1.0, 1.5, 2.0, 3.0]
        temp_predictions = []
        
        for temp in temperatures:
            probs_temp = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"{strategy} - Temp {temp}"):
                    batch = batch.to(device)
                    if torch.isnan(batch.x).any():
                        batch.x = torch.nan_to_num(batch.x, nan=0.0)
                    
                    out = model(batch.x, batch.edge_index, batch.batch)
                    out_scaled = out / temp
                    probs = F.softmax(out_scaled, dim=1)
                    probs_temp.extend(probs.cpu().numpy())
            temp_predictions.append(np.array(probs_temp))
        
        # Technique 3: Stochastic inference (multiple runs with dropout)
        model.train()  # Enable dropout
        stochastic_predictions = []
        
        for run in range(10):  # More runs
            probs_stoch = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"{strategy} - Stochastic {run+1}"):
                    batch = batch.to(device)
                    if torch.isnan(batch.x).any():
                        batch.x = torch.nan_to_num(batch.x, nan=0.0)
                    
                    out = model(batch.x, batch.edge_index, batch.batch)
                    probs = F.softmax(out, dim=1)
                    probs_stoch.extend(probs.cpu().numpy())
            stochastic_predictions.append(np.array(probs_stoch))
        
        # Combine all techniques for this strategy
        probs_regular = np.array(probs_regular)
        avg_temp_probs = np.mean(temp_predictions, axis=0)
        avg_stoch_probs = np.mean(stochastic_predictions, axis=0)
        uncertainty_stoch = np.std(stochastic_predictions, axis=0)
        
        # Weighted combination favoring seizure detection
        strategy_combined = (
            0.25 * probs_regular +
            0.35 * avg_temp_probs +
            0.4 * avg_stoch_probs
        )
        
        ensemble_predictions.append({
            'probs': strategy_combined,
            'uncertainty': uncertainty_stoch,
            'strategy': strategy
        })
        
        print(f"✅ Completed {strategy} strategy")
    
    # Strategy 2: Advanced Ensemble
    print("\n🎯 Creating advanced ensemble...")
    
    if len(ensemble_predictions) == 0:
        print("❌ No predictions generated!")
        return None
    
    # Combine all strategies with seizure-focused weighting
    all_probs = [pred['probs'] for pred in ensemble_predictions]
    all_uncertainties = [pred['uncertainty'] for pred in ensemble_predictions]
    
    # Weight strategies: sparse gets less weight, medium and dense get more
    strategy_weights = []
    for pred in ensemble_predictions:
        if pred['strategy'] == 'sparse':
            strategy_weights.append(0.2)
        elif pred['strategy'] == 'medium':
            strategy_weights.append(0.4)
        else:  # dense
            strategy_weights.append(0.4)
    
    # Normalize weights
    strategy_weights = np.array(strategy_weights)
    strategy_weights = strategy_weights / np.sum(strategy_weights)
    
    print(f"Strategy weights: {dict(zip([p['strategy'] for p in ensemble_predictions], strategy_weights))}")
    
    # Weighted ensemble
    final_probs = np.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        final_probs += strategy_weights[i] * probs
    
    # Calculate ensemble uncertainty
    ensemble_uncertainty = np.mean([unc[:, 1] for unc in all_uncertainties], axis=0)
    
    # Strategy 3: Sophisticated Decision Rules
    print("\n🧮 Applying sophisticated decision rules...")
    
    seizure_probs = final_probs[:, 1]
    predictions = np.zeros(len(seizure_probs), dtype=int)
    
    # Multi-tier thresholding system
    for i in range(len(seizure_probs)):
        prob = seizure_probs[i]
        uncertainty = ensemble_uncertainty[i]
        
        # Rule 1: High confidence seizure
        if prob > 0.35:
            predictions[i] = 1
        
        # Rule 2: Medium confidence with high uncertainty (might be seizure)
        elif prob > 0.25 and uncertainty > 0.1:
            predictions[i] = 1
        
        # Rule 3: Any individual strategy strongly suggests seizure
        elif any(pred['probs'][i, 1] > 0.4 for pred in ensemble_predictions):
            predictions[i] = 1
        
        # Rule 4: Consensus among strategies (even if low confidence)
        elif sum(pred['probs'][i, 1] > 0.3 for pred in ensemble_predictions) >= 2:
            predictions[i] = 1
        
        # Rule 5: Very low threshold for final safety net
        elif prob > 0.2:
            predictions[i] = 1
        
        else:
            predictions[i] = 0
    
    # Strategy 4: Create multiple submission variants
    print("\n📊 Creating multiple submission variants...")
    
    # Variant 1: Current predictions
    submission_v1 = pd.DataFrame({
        'id': test_segments.index,
        'label': predictions
    })
    
    # Variant 2: Even more aggressive
    predictions_v2 = np.zeros(len(seizure_probs), dtype=int)
    for i in range(len(seizure_probs)):
        prob = seizure_probs[i]
        uncertainty = ensemble_uncertainty[i]
        
        # Much lower thresholds
        if (prob > 0.15 or  # Very low threshold
            uncertainty > 0.08 or  # High uncertainty = potential seizure
            any(pred['probs'][i, 1] > 0.25 for pred in ensemble_predictions) or  # Any strategy suggests
            max(pred['probs'][i, 1] for pred in ensemble_predictions) > 0.3):  # Max probability
            predictions_v2[i] = 1
        else:
            predictions_v2[i] = 0
    
    submission_v2 = pd.DataFrame({
        'id': test_segments.index,
        'label': predictions_v2
    })
    
    # Variant 3: Balanced approach
    predictions_v3 = (seizure_probs > 0.3).astype(int)
    submission_v3 = pd.DataFrame({
        'id': test_segments.index,
        'label': predictions_v3
    })
    
    # Save all variants
    submission_v1.to_csv('submission_advanced_v1.csv', index=False)
    submission_v2.to_csv('submission_advanced_v2_aggressive.csv', index=False)
    submission_v3.to_csv('submission_advanced_v3_balanced.csv', index=False)
    
    # Print statistics
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"V1 (Sophisticated): {(submission_v1['label'] == 1).sum()}/{len(submission_v1)} seizures ({(submission_v1['label'] == 1).mean():.3f})")
    print(f"V2 (Aggressive): {(submission_v2['label'] == 1).sum()}/{len(submission_v2)} seizures ({(submission_v2['label'] == 1).mean():.3f})")
    print(f"V3 (Balanced): {(submission_v3['label'] == 1).sum()}/{len(submission_v3)} seizures ({(submission_v3['label'] == 1).mean():.3f})")
    
    print(f"\nAverage seizure probability: {np.mean(seizure_probs):.3f}")
    print(f"Average uncertainty: {np.mean(ensemble_uncertainty):.3f}")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'id': test_segments.index,
        'pred_v1': predictions,
        'pred_v2': predictions_v2,
        'pred_v3': predictions_v3,
        'seizure_prob': seizure_probs,
        'uncertainty': ensemble_uncertainty,
        'confidence': np.max(final_probs, axis=1)
    })
    
    # Add individual strategy probabilities
    for i, pred in enumerate(ensemble_predictions):
        analysis_df[f'{pred["strategy"]}_prob'] = pred['probs'][:, 1]
    
    analysis_df.to_csv('advanced_seizure_analysis.csv', index=False)
    print("Detailed analysis saved as 'advanced_seizure_analysis.csv'")
    
    return submission_v1, submission_v2, submission_v3

if __name__ == "__main__":
    v1, v2, v3 = advanced_seizure_submission()