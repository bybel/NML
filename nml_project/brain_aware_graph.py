import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import networkx as nx

class BrainAwareGraphConstructor:
    """
    Graph constructor that respects neuroanatomical connectivity patterns.
    """
    def __init__(self, distances, channels):
        self.distances = distances
        self.channels = channels
        self.n_channels = len(channels)
        
        # Define brain regions based on standard 10-20 system
        self.brain_regions = self._define_brain_regions()
        self.channel_to_region = self._map_channels_to_regions()
        
        print(f"Channel mapping: {self.channel_to_region}")
        
    def _define_brain_regions(self):
        """Define anatomical brain regions."""
        regions = {
            'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6'],
            'central': ['C3', 'C4', 'Cz', 'CP1', 'CP2', 'CP5', 'CP6'],
            'parietal': ['P3', 'P4', 'Pz', 'P7', 'P8'],
            'occipital': ['O1', 'O2', 'Oz'],
            'temporal': ['T3', 'T4', 'T5', 'T6', 'TP9', 'TP10', 'T7', 'T8'],
            'midline': ['Fz', 'Cz', 'Pz', 'Oz']
        }
        return regions
    
    def _map_channels_to_regions(self):
        """Map each channel to its brain region."""
        channel_to_region = {}
        
        for channel in self.channels:
            # Find which region this channel belongs to
            assigned = False
            for region, region_channels in self.brain_regions.items():
                if any(ch.lower() in channel.lower() for ch in region_channels):
                    channel_to_region[channel] = region
                    assigned = True
                    break
            
            if not assigned:
                # Default assignment based on naming pattern
                channel_upper = channel.upper()
                if any(x in channel_upper for x in ['FP', 'F']):
                    channel_to_region[channel] = 'frontal'
                elif 'C' in channel_upper:
                    channel_to_region[channel] = 'central'
                elif 'P' in channel_upper:
                    channel_to_region[channel] = 'parietal'
                elif 'O' in channel_upper:
                    channel_to_region[channel] = 'occipital'
                elif 'T' in channel_upper:
                    channel_to_region[channel] = 'temporal'
                else:
                    channel_to_region[channel] = 'other'
        
        return channel_to_region
    
    def create_neuroanatomical_edges(self, connectivity_strength='medium'):
        """
        Create edges based on neuroanatomical connectivity patterns.
        
        Args:
            connectivity_strength: 'sparse', 'medium', or 'dense'
        """
        edge_list = []
        
        # Define connectivity rules based on strength
        if connectivity_strength == 'sparse':
            distance_threshold = np.percentile(self.distances[self.distances > 0], 15)
            inter_region_factor = 0.4
            max_connections_per_node = 3
        elif connectivity_strength == 'medium':
            distance_threshold = np.percentile(self.distances[self.distances > 0], 25)
            inter_region_factor = 0.6
            max_connections_per_node = 5
        else:  # dense
            distance_threshold = np.percentile(self.distances[self.distances > 0], 35)
            inter_region_factor = 0.8
            max_connections_per_node = 8
        
        print(f"Using distance threshold: {distance_threshold:.3f} for {connectivity_strength} connectivity")
        
        # Track connections per node to avoid over-connectivity
        node_connections = {i: 0 for i in range(self.n_channels)}
        
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                # Skip if either node already has too many connections
                if (node_connections[i] >= max_connections_per_node or 
                    node_connections[j] >= max_connections_per_node):
                    continue
                
                channel_i = self.channels[i]
                channel_j = self.channels[j]
                
                region_i = self.channel_to_region.get(channel_i, 'other')
                region_j = self.channel_to_region.get(channel_j, 'other')
                
                distance = self.distances[i, j]
                
                # Connection probability based on neuroanatomical rules
                connect = False
                
                # Rule 1: Always connect very close channels (local connectivity)
                if distance <= np.percentile(self.distances[self.distances > 0], 10):
                    connect = True
                
                # Rule 2: Intra-region connectivity (within same brain region)
                elif region_i == region_j and distance <= distance_threshold * 1.2:
                    connect = True
                
                # Rule 3: Inter-hemispheric connectivity (symmetric channels)
                elif self._are_symmetric_channels(channel_i, channel_j):
                    if distance <= distance_threshold * 1.1:
                        connect = True
                
                # Rule 4: Frontal-central-parietal pathway (main brain highway)
                elif self._is_anterior_posterior_pathway(region_i, region_j):
                    if distance <= distance_threshold * inter_region_factor:
                        connect = True
                
                # Rule 5: Temporal connections to nearby regions
                elif 'temporal' in [region_i, region_j]:
                    other_region = region_j if region_i == 'temporal' else region_i
                    if other_region in ['frontal', 'central', 'parietal'] and distance <= distance_threshold * 0.7:
                        connect = True
                
                # Rule 6: Midline connections (important for bilateral coordination)
                elif self._involves_midline(channel_i, channel_j):
                    if distance <= distance_threshold * 0.9:
                        connect = True
                
                if connect:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected graph
                    node_connections[i] += 1
                    node_connections[j] += 1
        
        if len(edge_list) == 0:
            # Fallback: create a minimal connected graph
            print("Warning: No edges created, using fallback connectivity")
            edge_list = self._create_minimal_connected_graph()
        
        print(f"Created {len(edge_list)//2} edges for {connectivity_strength} connectivity")
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _are_symmetric_channels(self, ch1, ch2):
        """Check if two channels are symmetric (left-right hemisphere)."""
        # Common symmetric pairs in EEG
        symmetric_pairs = [
            ('1', '2'), ('3', '4'), ('5', '6'), ('7', '8'), ('9', '10'),
            ('left', 'right'), ('L', 'R')
        ]
        
        for pair in symmetric_pairs:
            if (pair[0] in ch1 and pair[1] in ch2) or (pair[1] in ch1 and pair[0] in ch2):
                # Check if the base name is similar (e.g., F3 and F4)
                base1 = ch1.replace(pair[0], '').replace(pair[1], '')
                base2 = ch2.replace(pair[0], '').replace(pair[1], '')
                if base1 == base2:
                    return True
        
        return False
    
    def _is_anterior_posterior_pathway(self, region1, region2):
        """Check if connection follows anterior-posterior pathway."""
        pathway_connections = [
            ('frontal', 'central'),
            ('central', 'parietal'),
            ('frontal', 'parietal'),
            ('parietal', 'occipital'),
            ('central', 'occipital')  # Added direct central-occipital
        ]
        
        pair = (region1, region2) if region1 < region2 else (region2, region1)
        return pair in pathway_connections
    
    def _involves_midline(self, ch1, ch2):
        """Check if connection involves midline channels."""
        midline_indicators = ['z', 'Z', 'midline']
        
        ch1_midline = any(indicator in ch1 for indicator in midline_indicators)
        ch2_midline = any(indicator in ch2 for indicator in midline_indicators)
        
        return ch1_midline or ch2_midline
    
    def _create_minimal_connected_graph(self):
        """Create a minimal connected graph as fallback."""
        edge_list = []
        
        # Connect each channel to its 2-3 nearest neighbors
        for i in range(self.n_channels):
            distances_from_i = self.distances[i, :]
            # Find 3 nearest neighbors
            nearest_indices = np.argsort(distances_from_i)[1:4]  # Skip self (index 0)
            
            for j in nearest_indices:
                if distances_from_i[j] > 0:  # Valid connection
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        print(f"Fallback: Created {len(edge_list)//2} edges")
        return edge_list
    
    def analyze_connectivity(self, edge_index):
        """Analyze the connectivity properties of the graph."""
        num_edges = edge_index.shape[1] // 2  # Undirected edges
        max_possible_edges = self.n_channels * (self.n_channels - 1) // 2
        
        connectivity_ratio = num_edges / max_possible_edges
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        G.add_nodes_from(range(self.n_channels))
        edge_list = edge_index.t().cpu().numpy()
        G.add_edges_from(edge_list)
        
        # Calculate graph properties
        try:
            avg_clustering = nx.average_clustering(G)
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                # Calculate for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
            components = nx.number_connected_components(G)
        except:
            avg_clustering = 0
            avg_path_length = float('inf')
            components = self.n_channels
        
        analysis = {
            'num_edges': num_edges,
            'connectivity_ratio': connectivity_ratio,
            'avg_clustering': avg_clustering,
            'avg_path_length': avg_path_length,
            'connected_components': components,
            'is_connected': components == 1
        }
        
        return analysis