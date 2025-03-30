import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_current_season():
    """Load the 2025 season data"""
    try:
        df = pd.read_csv("data/school_stats_2025.csv")
        print(f"Loaded data for {len(df)} teams from 2025 season")
        return df
    except Exception as e:
        print(f"Error loading 2025 season data: {e}")
        return None

def pca_based_ranking(df):
    """
    Rank teams based on factors identified as important by PCA
    """
    # Based on your PCA analysis, these are the most important factors
    # with their approximate weights from the visualization
    weights = {
        'SRS': 0.30,       # Strong loading on PC1
        'SOS': 0.20,       # Strong loading on PC1
        'Win_Pct': 0.15,   # Strong loading on PC1
        'Points_For': 0.10, # Moderate loading on PC1
        'FG_Pct': 0.08,    # Moderate loading on PC1
        'BLK': 0.05,      # Moderate loading on PC2
        'AST': 0.05,      # Moderate loading on PC2
        'STL': 0.04,      # Moderate loading on PC2
        'TRB': 0.03       # Minor loading on PC2
    }
    
    # Create a weighted score
    df['PCA_Score'] = 0
    
    # Apply weights
    for feature, weight in weights.items():
        if feature in df.columns:
            # Convert to numeric
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
            # Handle missing values
            df[feature] = df[feature].fillna(df[feature].mean())
            
            # Standardize the feature (z-score)
            z_score = (df[feature] - df[feature].mean()) / df[feature].std()
            
            # Add weighted contribution
            df['PCA_Score'] += z_score * weight
    
    # Apply softmax to get probabilities that sum to 1
    # First, normalize the scores to avoid numerical issues
    scores = df['PCA_Score'].values
    scores = scores - np.max(scores)  # Prevent overflow
    
    # Apply softmax
    exp_scores = np.exp(scores)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    # Scale so the top team has 20% probability (realistic for NCAA tournament)
    df['Tournament_Probability'] = softmax_scores
    max_prob = df['Tournament_Probability'].max()
    scaling_factor = 0.20 / max_prob  # Scale so top team has 20%
    df['Tournament_Probability'] = df['Tournament_Probability'] * scaling_factor
    
    # Ensure probabilities sum to 100%
    remaining_prob = 1.0 - df['Tournament_Probability'].sum()
    df['Tournament_Probability'] = df['Tournament_Probability'] + remaining_prob / len(df)
    
    # Sort by probability
    df = df.sort_values('Tournament_Probability', ascending=False)
    
    return df

def visualize_top_contenders(df, top_n=20):
    """Create visualizations for the top contenders"""
    
    # Select top contenders
    top_teams = df.head(top_n).copy()
    
    # Bar chart of probabilities
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar chart
    ax = sns.barplot(x='Tournament_Probability', y='School', 
                    data=top_teams)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 0.001, p.get_y() + p.get_height()/2, 
                f'{width:.1%}', ha='left', va='center')
    
    # Set title and labels
    plt.title('Top 2025 NCAA Tournament Championship Contenders\n(Based on PCA Analysis)', 
              fontsize=16)
    plt.xlabel('Championship Probability', fontsize=12)
    plt.text(0, -0.05, 
             'Note: Probabilities reflect historical tournament unpredictability. Even the strongest team typically has only a ~20% chance.',
             transform=ax.transAxes, fontsize=9)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('2025_pca_predictions.png')
    print("Saved predictions to 2025_pca_predictions.png")
    
    # Create scatter plot if relevant columns exist
    if 'SRS' in df.columns and 'Win_Pct' in df.columns:
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot for all teams
        plt.scatter(df['SRS'], df['Win_Pct'], 
                   alpha=0.3, c='lightblue', s=30)
        
        # Highlight top teams
        top_teams_scatter = plt.scatter(top_teams['SRS'], top_teams['Win_Pct'],
                                       s=top_teams['Tournament_Probability']*1000, 
                                       c=top_teams['Tournament_Probability'],
                                       alpha=0.8, cmap='viridis')
        
        # Add labels for top 10 teams
        for i, row in df.head(10).iterrows():
            plt.annotate(row['School'], 
                        (row['SRS'], row['Win_Pct']),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(top_teams_scatter, label='Championship Probability')
        
        # Set labels and title
        plt.xlabel('SRS (Simple Rating System)', fontsize=12)
        plt.ylabel('Win Percentage', fontsize=12)
        plt.title('2025 Championship Contenders by SRS and Win Percentage', fontsize=14)
        plt.grid(alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('2025_pca_scatter.png')
        print("Saved scatter plot to 2025_pca_scatter.png")

def main():
    # Load 2025 season data
    current_data = load_current_season()
    
    if current_data is None:
        print("Cannot continue without 2025 season data")
        return
        
    # Apply PCA-based ranking
    print("\nRanking teams based on PCA factors...")
    ranked_teams = pca_based_ranking(current_data)
    
    # Display top contenders
    print("\nTop Championship Contenders for 2025:")
    display_cols = ['School', 'Tournament_Probability']
    
    # Add additional informative columns if available
    for col in ['Wins', 'Losses', 'Win_Pct', 'SRS', 'SOS']:
        if col in ranked_teams.columns:
            display_cols.append(col)
    
    # Get top 10 teams
    top_10 = ranked_teams[display_cols].head(10)
    
    # Format probabilities for display
    formatted_top_10 = top_10.copy()
    formatted_top_10['Tournament_Probability'] = formatted_top_10['Tournament_Probability'].map("{:.1%}".format)
    
    # Display results
    print(formatted_top_10)
    
    # Calculate cumulative probabilities
    top_10_prob = ranked_teams.head(10)['Tournament_Probability'].sum()
    top_20_prob = ranked_teams.head(20)['Tournament_Probability'].sum()
    print(f"\nThe top 10 teams have a combined {top_10_prob:.1%} chance of winning")
    print(f"The top 20 teams have a combined {top_20_prob:.1%} chance of winning")
    
    # Create visualizations
    visualize_top_contenders(ranked_teams)
    
    # Save results
    ranked_teams.to_csv("2025_pca_predictions.csv", index=False)
    print("\nSaved complete predictions to 2025_pca_predictions.csv")

if __name__ == "__main__":
    main()