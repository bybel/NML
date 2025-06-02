import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_brain_aware_results():
    """
    Analyze the brain-aware prediction results.
    """
    print("Analyzing brain-aware prediction results...")
    
    # Load the detailed analysis
    analysis_df = pd.read_csv('brain_aware_analysis.csv')
    
    print(f"Total predictions: {len(analysis_df)}")
    print(f"Class distribution:")
    print(analysis_df['prediction'].value_counts())
    print(f"\nClass proportions:")
    print(analysis_df['prediction'].value_counts(normalize=True))
    
    # Confidence analysis
    print(f"\nConfidence statistics:")
    print(f"Mean confidence: {analysis_df['confidence'].mean():.4f}")
    print(f"Std confidence: {analysis_df['confidence'].std():.4f}")
    print(f"Min confidence: {analysis_df['confidence'].min():.4f}")
    print(f"Max confidence: {analysis_df['confidence'].max():.4f}")
    
    # Low confidence predictions
    low_confidence = analysis_df[analysis_df['confidence'] < 0.6]
    print(f"\nLow confidence predictions (< 0.6): {len(low_confidence)}")
    if len(low_confidence) > 0:
        print(f"Low confidence class distribution:")
        print(low_confidence['prediction'].value_counts(normalize=True))
    
    # High confidence predictions
    high_confidence = analysis_df[analysis_df['confidence'] > 0.8]
    print(f"\nHigh confidence predictions (> 0.8): {len(high_confidence)}")
    if len(high_confidence) > 0:
        print(f"High confidence class distribution:")
        print(high_confidence['prediction'].value_counts(normalize=True))
    
    # Compare with original submission if available
    try:
        original_df = pd.read_csv('submission_gat.csv')
        print(f"\nComparison with original submission:")
        print(f"Original class distribution: {(original_df['label'] == 1).mean():.3f}")
        print(f"Brain-aware class distribution: {(analysis_df['prediction'] == 1).mean():.3f}")
        
        # Agreement between models
        agreement = (original_df['label'] == analysis_df['prediction']).mean()
        print(f"Agreement between models: {agreement:.3f}")
        
    except FileNotFoundError:
        print("Original submission file not found for comparison")
    
    return analysis_df

if __name__ == "__main__":
    results = analyze_brain_aware_results()