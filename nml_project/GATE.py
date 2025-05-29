# GAT Model for EEG-based Epilepsy Classification
# Adapted for your data format

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
data_path = './data'
distances_df = pd.read_csv(f'{data_path}/distances_3d.csv')
train_segments = pq.read_table(f'{data_path}/train/segments.parquet').to_pandas()

print(f"Train segments shape: {train_segments.shape}")
print(f"Columns: {train_segments.columns.tolist()}")
print(f"\nFirst few rows:")
print(train_segments.head())

# Labels are in the segments dataframe
train_labels = train_segments['label'].values

# Convert distances from long format to matrix format
print(f"\nDistances shape: {distances_df.shape}")
print(distances_df.head())

# Get unique channels
channels = sorted(list(set(distances_df['from'].unique().tolist() + distances_df['to'].unique().tolist())))
n_channels = len(channels)
print(f"\nNumber of channels: {n_channels}")
print(f"Channels: {channels[:10]}...")  # Show first 10

# Create distance matrix
distances = np.zeros((n_channels, n_channels))
channel_to_idx = {ch: idx for idx, ch in enumerate(channels)}

for _, row in distances_df.iterrows():
    i = channel_to_idx[row['from']]
    j = channel_to_idx[row['to']]
    distances[i, j] = row['distance']
    distances[j, i] = row['distance']  # Symmetric matrix

print(f"\nDistance matrix shape: {distances.shape}")


class EEGGraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for EEG-based epilepsy classification.
    """
    def __init__(self, num_features, hidden_dim=64, num_heads=8, num_classes=2, dropout=0.2):
        super(EEGGraphAttentionNetwork, self).__init__()
        
        # GAT layers
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch):
        # GAT layers with batch norm and residual connections
        h1 = self.gat1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)
        
        h2 = self.gat2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        
        h3 = self.gat3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.elu(h3)
        h3 = self.dropout(h3)
        
        # Global pooling
        h_graph = global_mean_pool(h3, batch)
        
        # Classification
        out = F.relu(self.fc1(h_graph))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class EEGFeatureExtractor:
    """
    Extract features from EEG signals for graph construction.
    """
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
    def extract_features(self, eeg_data):
        """
        Extract features from EEG data with NaN handling.
        
        Args:
            eeg_data: numpy array of shape (n_channels, n_samples) or flattened array
        
        Returns:
            features: numpy array of shape (n_channels, n_features)
        """
        # If data is flattened, reshape based on expected channels
        if len(eeg_data.shape) == 1:
            n_channels = len(self.channels) if hasattr(self, 'channels') else 19
            n_samples = len(eeg_data) // n_channels
            eeg_data = eeg_data.reshape(n_channels, n_samples)
        
        n_channels, n_samples = eeg_data.shape
        features = []
        
        for ch in range(n_channels):
            ch_signal = eeg_data[ch, :]
            
            # Handle NaN/inf values
            if np.any(np.isnan(ch_signal)) or np.any(np.isinf(ch_signal)):
                ch_signal = np.nan_to_num(ch_signal, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Skip channels with all zeros or constant values
            if np.std(ch_signal) == 0:
                ch_signal = ch_signal + np.random.normal(0, 1e-6, len(ch_signal))
            
            ch_features = []
            
            # Time domain features with safety checks
            ch_features.append(np.mean(ch_signal))
            ch_features.append(max(np.std(ch_signal), 1e-8))  # Avoid zero std
            ch_features.append(np.max(ch_signal) - np.min(ch_signal))
            ch_features.append(np.percentile(ch_signal, 75) - np.percentile(ch_signal, 25))
            
            # Statistical features
            ch_features.append(np.median(ch_signal))
            ch_features.append(max(np.var(ch_signal), 1e-8))  # Avoid zero variance
            
            from scipy.stats import skew, kurtosis
            try:
                skew_val = skew(ch_signal)
                kurt_val = kurtosis(ch_signal)
                ch_features.append(skew_val if not np.isnan(skew_val) else 0.0)
                ch_features.append(kurt_val if not np.isnan(kurt_val) else 0.0)
            except:
                ch_features.extend([0.0, 0.0])
            
            # Frequency domain features with better error handling
            try:
                nperseg = min(256, max(64, n_samples // 4))  # Adaptive window size
                freqs, psd = signal.welch(ch_signal, fs=self.sampling_rate, nperseg=nperseg)
                
                # Handle edge cases
                if len(psd) == 0 or np.sum(psd) == 0:
                    ch_features.extend([0.2, 0.2, 0.2, 0.2, 0.2, 10.0])  # Default values
                else:
                    # Band power features with fixed frequency ranges
                    delta_idx = np.where((freqs >= 0.5) & (freqs <= 4))[0]
                    theta_idx = np.where((freqs >= 4) & (freqs <= 8))[0]
                    alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
                    beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
                    gamma_idx = np.where((freqs >= 30) & (freqs <= min(100, self.sampling_rate/2)))[0]
                    
                    total_power = max(np.sum(psd), 1e-8)  # Avoid division by zero
                    
                    # Calculate relative powers with safety checks
                    delta_power = np.sum(psd[delta_idx]) / total_power if len(delta_idx) > 0 else 0.0
                    theta_power = np.sum(psd[theta_idx]) / total_power if len(theta_idx) > 0 else 0.0
                    alpha_power = np.sum(psd[alpha_idx]) / total_power if len(alpha_idx) > 0 else 0.0
                    beta_power = np.sum(psd[beta_idx]) / total_power if len(beta_idx) > 0 else 0.0
                    gamma_power = np.sum(psd[gamma_idx]) / total_power if len(gamma_idx) > 0 else 0.0
                    
                    ch_features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])
                    
                    # Spectral centroid with safety check
                    spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else np.mean(freqs)
                    ch_features.append(spectral_centroid if not np.isnan(spectral_centroid) else 10.0)
            
            except Exception as e:
                print(f"Error in frequency analysis for channel {ch}: {e}")
                # Add default frequency features
                ch_features.extend([0.2, 0.2, 0.2, 0.2, 0.2, 10.0])
            
            # Final NaN check
            ch_features = [f if not np.isnan(f) and not np.isinf(f) else 0.0 for f in ch_features]
            features.append(ch_features)
        
        return np.array(features)
    
    def fit(self, features_list):
        """Fit the scaler on training data"""
        all_features = np.vstack(features_list)
        self.scaler.fit(all_features)
        
    def transform(self, features):
        """Transform features using fitted scaler"""
        return self.scaler.transform(features)


class EEGGraphConstructor:
    """
    Construct graphs from EEG data using the provided distance matrix.
    """
    def __init__(self, distances_matrix, threshold_percentile=30):
        self.distances = distances_matrix
        self.threshold_percentile = threshold_percentile
        self.n_channels = len(self.distances)
        
    def create_edge_index_from_distances(self):
        """
        Create edge indices based on 3D electrode distances with fallback.
        """
        try:
            # Get non-zero distances for threshold calculation
            non_zero_distances = self.distances[self.distances > 0]
            
            if len(non_zero_distances) == 0:
                # Fallback: create a simple chain connectivity
                edges = []
                for i in range(self.n_channels - 1):
                    edges.append([i, i + 1])
                    edges.append([i + 1, i])
                return torch.tensor(edges, dtype=torch.long).t()
            
            # Calculate threshold based on percentile of distances
            threshold = np.percentile(non_zero_distances, self.threshold_percentile)
            
            edges = []
            for i in range(self.n_channels):
                for j in range(i + 1, self.n_channels):
                    if 0 < self.distances[i, j] <= threshold:
                        edges.append([i, j])
                        edges.append([j, i])  # Undirected graph
            
            # Ensure we have at least some edges
            if len(edges) == 0:
                # Fallback: connect each node to its nearest neighbors
                for i in range(self.n_channels):
                    row = self.distances[i, :]
                    row[i] = np.inf  # Exclude self
                    nearest = np.argmin(row)
                    edges.append([i, nearest])
                    edges.append([nearest, i])
                    
            return torch.tensor(edges, dtype=torch.long).t()
            
        except Exception as e:
            print(f"Error in edge creation: {e}")
            # Ultimate fallback: create a simple connectivity
            edges = [[i, (i+1) % self.n_channels] for i in range(self.n_channels)]
            edges.extend([[(i+1) % self.n_channels, i] for i in range(self.n_channels)])
            return torch.tensor(edges, dtype=torch.long).t()
    


def load_eeg_signals(signals_path, start_time, end_time, sampling_rate, data_path='./data/train'):
    """
    Load EEG signals from parquet file for a specific time segment.
    """
    try:
        full_path = f"{data_path}/{signals_path}"
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            return None
            
        signals_df = pq.read_table(full_path).to_pandas()
        
        # Calculate sample indices
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        
        # Validate indices
        if start_sample >= len(signals_df) or end_sample > len(signals_df):
            print(f"Invalid sample range: {start_sample}-{end_sample} for file with {len(signals_df)} samples")
            return None
        
        # Extract the segment
        segment = signals_df.iloc[start_sample:end_sample]
        
        # Convert to numpy array and ensure proper shape
        eeg_data = segment.values.T  # Transpose to get (n_channels, n_samples)
        
        return eeg_data
        
    except Exception as e:
        print(f"Error loading segment from {signals_path}: {e}")
        return None


def prepare_data_from_segments(segments_df, labels, channels, feature_extractor, 
                              graph_constructor, data_path='./data/train',
                              use_correlation_edges=False):
    """
    Prepare data from segments DataFrame for GAT processing.
    """
    data_list = []
    
    # Group by signals_path to avoid loading the same file multiple times
    grouped = segments_df.groupby('signals_path')
    
    print("Loading EEG signals and extracting features...")
    all_features = []
    
    # Cache for loaded signal files
    signal_cache = {}
    
    for signal_path, group in tqdm(grouped):
        # Load the signal file once
        if signal_path not in signal_cache:
            full_path = f"{data_path}/{signal_path}"
            signal_cache[signal_path] = pq.read_table(full_path).to_pandas()
        
        signals_df = signal_cache[signal_path]
        
        # Process each segment in this file
        for idx, row in group.iterrows():
            # Calculate sample indices
            start_sample = int(row['start_time'] * row['sampling_rate'])
            end_sample = int(row['end_time'] * row['sampling_rate'])
            
            # Extract the segment
            segment = signals_df.iloc[start_sample:end_sample]
            
            # Convert to numpy array (channels x samples)
            eeg_data = segment.values.T  # Transpose to get (n_channels, n_samples)
            
            # Ensure we have the right number of channels
            if eeg_data.shape[0] != len(channels):
                print(f"Warning: Expected {len(channels)} channels, got {eeg_data.shape[0]}")
                continue
            
            # Extract features
            features = feature_extractor.extract_features(eeg_data)
            all_features.append(features)
    
    # Fit scaler on all features
    print("Fitting feature scaler...")
    feature_extractor.fit(all_features)
    
    # Create graph data objects
    print("Creating graph data objects...")
    feature_idx = 0
    
    for signal_path, group in tqdm(grouped):
        signals_df = signal_cache[signal_path]
        
        for idx, row in group.iterrows():
            # Transform features
            features = feature_extractor.transform(all_features[feature_idx])
            feature_idx += 1
            
            # Create edges
            if use_correlation_edges:
                # Reload the EEG data for correlation calculation
                start_sample = int(row['start_time'] * row['sampling_rate'])
                end_sample = int(row['end_time'] * row['sampling_rate'])
                segment = signals_df.iloc[start_sample:end_sample]
                eeg_data = segment.values.T
                edge_index = graph_constructor.create_correlation_based_edges(eeg_data)
            else:
                edge_index = graph_constructor.create_edge_index_from_distances()
            
            # Get label
            label = labels[segments_df.index.get_loc(idx)]
            
            # Create Data object
            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long)
            )
            
            data_list.append(data)
    
    return data_list


# Example: Prepare your data
print("\nPreparing data for GAT...")

# Initialize components
feature_extractor = EEGFeatureExtractor(sampling_rate=250)  # Using sampling rate from your data
graph_constructor = EEGGraphConstructor(distances, threshold_percentile=30)

# Check the edge structure
sample_edges = graph_constructor.create_edge_index_from_distances()
print(f"\nNumber of edges in graph: {sample_edges.shape[1]}")
print(f"Graph density: {sample_edges.shape[1] / (n_channels * (n_channels - 1)):.3f}")

# Visualize the graph structure
plt.figure(figsize=(10, 8))
adj_matrix = np.zeros((n_channels, n_channels))
for i in range(sample_edges.shape[1]):
    adj_matrix[sample_edges[0, i], sample_edges[1, i]] = 1
    
plt.imshow(adj_matrix, cmap='binary')
plt.colorbar()
plt.title('EEG Channel Connectivity Matrix')
plt.xlabel('Channel')
plt.ylabel('Channel')
plt.show()

# Prepare the data - let's do a small subset first to test
print("\nPreparing a subset of data to test...")

# Check class distribution in full dataset
print(f"Full dataset class distribution:")
print(f"Class 0: {(train_labels == 0).sum()}, Class 1: {(train_labels == 1).sum()}")

# Create a balanced subset
subset_size = 100  # Start with 100 samples to test
class_0_indices = train_segments[train_labels == 0].index[:subset_size//2]
class_1_indices = train_segments[train_labels == 1].index[:subset_size//2]

# Combine indices and create subset
subset_indices = list(class_0_indices) + list(class_1_indices)
train_segments_subset = train_segments.loc[subset_indices]
train_labels_subset = train_labels[train_segments.index.isin(subset_indices)]

print(f"Subset class distribution: Class 0: {(train_labels_subset == 0).sum()}, Class 1: {(train_labels_subset == 1).sum()}")

data_list = prepare_data_from_segments(
    train_segments_subset, 
    train_labels_subset,
    channels,
    feature_extractor, 
    graph_constructor,
    data_path='./data/train',
    use_correlation_edges=False  # Set to True to use correlation-based edges
)

print(f"\nSuccessfully prepared {len(data_list)} graph samples")

# Split data
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42, stratify=train_labels_subset)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=[d.y.item() for d in train_data])

print(f"\nDataset sizes:")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_data)}")

# Check class distribution
train_labels_list = [d.y.item() for d in train_data]
val_labels_list = [d.y.item() for d in val_data]
test_labels_list = [d.y.item() for d in test_data]

print(f"\nClass distribution:")
print(f"Train - Class 0: {train_labels_list.count(0)}, Class 1: {train_labels_list.count(1)}")
print(f"Val - Class 0: {val_labels_list.count(0)}, Class 1: {val_labels_list.count(1)}")
print(f"Test - Class 0: {test_labels_list.count(0)}, Class 1: {test_labels_list.count(1)}")


def train_gat_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10):
    """
    Train the GAT model with comprehensive error handling.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Use class weights for imbalanced dataset
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch.y.numpy())
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            try:
                batch = batch.to(device)
                
                # Check for NaN in input
                if torch.isnan(batch.x).any():
                    print(f"NaN detected in batch {batch_idx} features")
                    batch.x = torch.nan_to_num(batch.x, nan=0.0)
                
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                
                # Check for NaN in output
                if torch.isnan(out).any():
                    print(f"NaN detected in model output at batch {batch_idx}")
                    continue
                
                loss = criterion(out, batch.y)
                
                if torch.isnan(loss):
                    print(f"NaN loss at batch {batch_idx}")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = out.argmax(dim=1)
                train_correct += (pred == batch.y).sum().item()
                train_total += batch.y.size(0)
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation (similar error handling)
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = batch.to(device)
                    
                    if torch.isnan(batch.x).any():
                        batch.x = torch.nan_to_num(batch.x, nan=0.0)
                    
                    out = model(batch.x, batch.edge_index, batch.batch)
                    
                    if torch.isnan(out).any():
                        continue
                    
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()
                    
                    probs = F.softmax(out, dim=1)
                    pred = out.argmax(dim=1)
                    
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())
                    val_probs.extend(probs[:, 1].cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        if len(val_preds) == 0:
            print("No valid validation predictions")
            continue
            
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # Calculate AUC safely
        try:
            if len(np.unique(val_labels)) > 1:
                val_auc = roc_auc_score(val_labels, val_probs)
            else:
                val_auc = np.nan
        except:
            val_auc = np.nan
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_gat_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            if not np.isnan(val_auc):
                print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}')
            else:
                print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            print(f'  Best Val Acc: {best_val_acc:.4f}')
    
    # Load best model
    if os.path.exists('best_gat_model.pth'):
        model.load_state_dict(torch.load('best_gat_model.pth'))
    
    return train_losses, val_losses, val_accuracies, model


# Example training code
# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize model
num_features = train_data[0].x.shape[1]
model = EEGGraphAttentionNetwork(
    num_features=num_features,
    hidden_dim=64,
    num_heads=4,
    num_classes=2,
    dropout=0.2
)

print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Number of features per node: {num_features}")

# Train model
print("\nTraining GAT model...")
train_losses, val_losses, val_accuracies, model = train_gat_model(
    model, train_loader, val_loader, num_epochs=50, lr=0.001, patience=10
)

# Evaluate on test set
print("\nEvaluating on test set...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)
model.eval()

test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        
        test_preds.extend(pred.cpu().numpy())
        test_labels.extend(batch.y.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

# Calculate final metrics
accuracy = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
cm = confusion_matrix(test_labels, test_preds)

# Calculate AUC only if both classes are present
if len(np.unique(test_labels)) > 1:
    auc = roc_auc_score(test_labels, test_probs)
else:
    auc = np.nan

print("\nFinal Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
if not np.isnan(auc):
    print(f"AUC: {auc:.4f}")
else:
    print(f"AUC: N/A (single class in test set)")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training history
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()

axes[0, 1].plot(val_accuracies)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Validation Accuracy')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('True')
axes[1, 0].set_title('Confusion Matrix')

# ROC Curve
from sklearn.metrics import roc_curve
if len(np.unique(test_labels)) > 1:
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
else:
    axes[1, 1].text(0.5, 0.5, 'ROC Curve not available\n(single class in test set)', 
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.show()

# To process the full dataset, uncomment below:
print("\n" + "="*50)
print("Processing full dataset...")
print("="*50)

full_data_list = prepare_data_from_segments(
    train_segments, 
    train_labels,
    channels,
    feature_extractor, 
    graph_constructor,
    data_path='./data/train',
    use_correlation_edges=False
)

print(f"\nTotal samples prepared: {len(full_data_list)}")

# Split and train on full dataset with proper stratification
all_labels_full = [d.y.item() for d in full_data_list]
train_data_full, test_data_full = train_test_split(
    full_data_list, 
    test_size=0.2, 
    random_state=42, 
    stratify=all_labels_full
)

train_labels_for_val_split = [d.y.item() for d in train_data_full]
train_data_full, val_data_full = train_test_split(
    train_data_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_labels_for_val_split
)

print(f"\nFull dataset splits:")
print(f"Train: {len(train_data_full)}")
print(f"Validation: {len(val_data_full)}")
print(f"Test: {len(test_data_full)}")

# Create new data loaders
train_loader_full = DataLoader(train_data_full, batch_size=64, shuffle=True)
val_loader_full = DataLoader(val_data_full, batch_size=64, shuffle=False)
test_loader_full = DataLoader(test_data_full, batch_size=64, shuffle=False)

# Initialize new model for full training
model_full = EEGGraphAttentionNetwork(
    num_features=num_features,
    hidden_dim=256,
    num_heads=16,
    num_classes=2,
    dropout=0.2
)

print(f"\nFull model parameters: {sum(p.numel() for p in model_full.parameters()):,}")

# Train on full dataset
print("\nTraining on full dataset...")
train_losses_full, val_losses_full, val_accuracies_full, model_full = train_gat_model(
    model_full, train_loader_full, val_loader_full, num_epochs=200, lr=0.002, patience=25
)

# PROPER EVALUATION ON FULL TEST SET
print("\n" + "="*50)
print("FINAL EVALUATION ON FULL TEST SET")
print("="*50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_full = model_full.to(device)
model_full.eval()

test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader_full, desc="Final Testing"):
        batch = batch.to(device)
        out = model_full(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        
        test_preds.extend(pred.cpu().numpy())
        test_labels.extend(batch.y.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

# Calculate final metrics
accuracy = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
cm = confusion_matrix(test_labels, test_preds)

# Calculate AUC only if both classes are present
if len(np.unique(test_labels)) > 1:
    auc = roc_auc_score(test_labels, test_probs)
else:
    auc = np.nan

print(f"\nFinal Test Results (Full Dataset):")
print(f"Total test samples: {len(test_labels)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
if not np.isnan(auc):
    print(f"AUC: {auc:.4f}")
else:
    print(f"AUC: N/A (single class in test set)")

# Detailed breakdown
print(f"\nDetailed Results:")
print(f"True Positives: {cm[1,1]}")
print(f"False Positives: {cm[0,1]}")
print(f"True Negatives: {cm[0,0]}")
print(f"False Negatives: {cm[1,0]}")

# Class distribution
test_class_dist = np.bincount(test_labels)
pred_class_dist = np.bincount(test_preds)
print(f"\nClass distribution in test set:")
print(f"Actual - Class 0: {test_class_dist[0]}, Class 1: {test_class_dist[1]}")
print(f"Predicted - Class 0: {pred_class_dist[0]}, Class 1: {pred_class_dist[1]}")

# Use the training history from full training for visualizations
train_losses = train_losses_full
val_losses = val_losses_full
val_accuracies = val_accuracies_full

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training history
axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss (Full Dataset)')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(val_accuracies, linewidth=2, color='green')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Validation Accuracy (Full Dataset)')
axes[0, 1].grid(True)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('True')
axes[1, 0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}')

# ROC Curve
from sklearn.metrics import roc_curve
if len(np.unique(test_labels)) > 1:
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
else:
    axes[1, 1].text(0.5, 0.5, 'ROC Curve not available\n(single class in test set)', 
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('gat_full_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining complete! Results saved to 'gat_full_results.png'")
print("Model weights saved to 'best_gat_model.pth'")
