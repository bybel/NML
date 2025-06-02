import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

class EnhancedEEGFeatureExtractor:
    """
    Enhanced feature extractor with more sophisticated features.
    """
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.scaler = RobustScaler()  # More robust to outliers
        
    def extract_features(self, eeg_data):
        """Extract comprehensive features from EEG signal."""
        features = []
        
        for channel_idx in range(eeg_data.shape[0]):
            channel_data = eeg_data[channel_idx, :]
            channel_features = []
            
            # Time domain features
            channel_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                skew(channel_data),
                kurtosis(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.max(channel_data) - np.min(channel_data),
                np.mean(np.abs(np.diff(channel_data))),  # Mean absolute deviation
                np.sqrt(np.mean(channel_data**2))  # RMS
            ])
            
            # Frequency domain features
            freqs, psd = signal.welch(channel_data, self.sampling_rate, nperseg=min(256, len(channel_data)))
            
            # Frequency bands
            delta_band = (0.5, 4)
            theta_band = (4, 8)
            alpha_band = (8, 13)
            beta_band = (13, 30)
            gamma_band = (30, 100)
            
            bands = [delta_band, theta_band, alpha_band, beta_band, gamma_band]
            
            for low, high in bands:
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_power = np.sum(psd[band_mask])
                    channel_features.append(band_power)
                else:
                    channel_features.append(0.0)
            
            # Spectral features
            channel_features.extend([
                np.sum(psd),  # Total power
                freqs[np.argmax(psd)],  # Peak frequency
                np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0,  # Spectral centroid
            ])
            
            # Hjorth parameters
            def hjorth_params(signal_data):
                first_deriv = np.diff(signal_data)
                second_deriv = np.diff(first_deriv)
                
                var_zero = np.var(signal_data)
                var_d1 = np.var(first_deriv)
                var_d2 = np.var(second_deriv)
                
                activity = var_zero
                mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
                complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 > 0 and mobility > 0 else 0
                
                return activity, mobility, complexity
            
            activity, mobility, complexity = hjorth_params(channel_data)
            channel_features.extend([activity, mobility, complexity])
            
            features.append(channel_features)
        
        return np.array(features)
    
    def fit(self, features_list):
        all_features = np.vstack(features_list)
        self.scaler.fit(all_features)
        
    def transform(self, features):
        return self.scaler.transform(features)

class ImprovedEEGGraphAttentionNetwork(nn.Module):
    """
    Enhanced GAT with more sophisticated architecture.
    """
    def __init__(self, num_features, hidden_dim=256, num_heads=16, num_classes=2, dropout=0.8):
        super().__init__()
        
        # Multi-layer GAT with residual connections
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads//2, dropout=dropout)
        self.gat4 = GATConv(hidden_dim * num_heads//2, hidden_dim//2, heads=1, dropout=dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_dim * num_heads//2)
        self.bn4 = nn.BatchNorm1d(hidden_dim//2)
        
        # Multiple pooling strategies
        self.pool_dim = hidden_dim//2 * 3  # mean + max + add pooling
        
        # Enhanced classifier with more layers
        self.fc1 = nn.Linear(self.pool_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim//2)
        self.bn_fc3 = nn.BatchNorm1d(hidden_dim//4)
        
    def forward(self, x, edge_index, batch):
        # Input validation
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # GAT layers with residual connections
        x1 = F.elu(self.bn1(self.gat1(x, edge_index)))
        x1 = self.dropout(x1)
        
        x2 = F.elu(self.bn2(self.gat2(x1, edge_index)))
        x2 = self.dropout(x2)
        
        x3 = F.elu(self.bn3(self.gat3(x2, edge_index)))
        x3 = self.dropout(x3)
        
        x4 = F.elu(self.bn4(self.gat4(x3, edge_index)))
        x4 = self.dropout(x4)
        
        # Multiple pooling strategies
        mean_pool = global_mean_pool(x4, batch)
        max_pool = global_max_pool(x4, batch)
        add_pool = global_add_pool(x4, batch)
        
        # Concatenate different pooling results
        x = torch.cat([mean_pool, max_pool, add_pool], dim=1)
        
        # Enhanced classifier
        x = F.elu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.elu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.elu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x

def improved_train_gat_model(model, train_loader, val_loader, num_epochs=300, lr=0.1, patience=30):
    """
    Enhanced training with better optimization strategies.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Better optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Use class weights for imbalanced dataset
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch.y.cpu().numpy())
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Focal loss for better handling of class imbalance
    def focal_loss(pred, target, alpha=class_weights, gamma=2.0):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha[target] * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
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
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            try:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = focal_loss(out, batch.y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                train_loss += loss.item()
                pred = out.argmax(dim=1)
                train_correct += (pred == batch.y).sum().item()
                train_total += batch.y.size(0)
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                try:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = focal_loss(out, batch.y)
                    
                    val_loss += loss.item()
                    pred = out.argmax(dim=1)
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        if len(val_preds) == 0:
            continue
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping with model saving
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_improved_gat_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    if os.path.exists('best_improved_gat_model.pth'):
        model.load_state_dict(torch.load('best_improved_gat_model.pth'))
    
    return train_losses, val_losses, val_accuracies, model