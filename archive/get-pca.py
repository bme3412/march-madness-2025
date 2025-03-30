import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data from a single season
def load_season_data(file_path):
    """Load data from a single season file"""
    try:
        df = pd.read_csv(file_path)
        # Extract year from filename
        year_match = re.search(r'(\d{4})\.csv', file_path)
        year = year_match.group(1) if year_match else "Unknown"
        df['Season'] = year
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load all seasons data
def load_all_seasons(data_dir="data"):
    """Load data from all available seasons"""
    all_data = []
    
    for file in os.listdir(data_dir):
        if file.startswith("school_stats_") and file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            print(f"Loading {file_path}")
            season_data = load_season_data(file_path)
            if season_data is not None:
                all_data.append(season_data)
    
    if not all_data:
        print("No data found")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Identify champion teams
def mark_champions(df):
    """Mark champion teams in the dataset"""
    # Define champions by year
    champions = {
        '2000': 'Michigan State',
        '2001': 'Duke',
        '2002': 'Maryland',
        '2003': 'Syracuse',
        '2004': 'Connecticut',
        '2005': 'North Carolina',
        '2006': 'Florida',
        '2007': 'Florida',
        '2008': 'Kansas',
        '2009': 'North Carolina',
        '2010': 'Duke',
        '2011': 'Connecticut',
        '2012': 'Kentucky',
        '2013': 'Louisville',
        '2014': 'Connecticut',
        '2015': 'Duke',
        '2016': 'Villanova',
        '2017': 'North Carolina',
        '2018': 'Villanova',
        '2019': 'Virginia',
        '2020': None,  # No champion (COVID)
        '2021': 'Baylor',
        '2022': 'Kansas',
        '2023': 'Connecticut',
        '2024': 'Connecticut',
        '2025': None
    }
    
    # Add champion column
    df['Is_Champion'] = False
    
    for season, champion in champions.items():
        if not champion:
            continue
            
        # Try different methods to match champion teams
        for index, row in df[df['Season'] == season].iterrows():
            school_name = row['School']
            
            # Try different name variations
            if champion.lower() in school_name.lower() or school_name.lower() in champion.lower():
                df.at[index, 'Is_Champion'] = True
                print(f"Identified champion: {school_name} ({season})")
                break
            # Special case for Connecticut/UConn
            elif (champion.lower() == 'connecticut' and 'uconn' in school_name.lower()) or \
                 (champion.lower() == 'uconn' and 'connecticut' in school_name.lower()):
                df.at[index, 'Is_Champion'] = True
                print(f"Identified champion: {school_name} ({season})")
                break
    
    # Print summary
    champion_count = df['Is_Champion'].sum()
    print(f"Total champions identified: {champion_count}")
    return df

# Clean and prepare data for analysis
def prepare_data(df):
    """Clean and prepare data for analysis"""
    # Print original columns
    print("\nOriginal columns:", df.columns.tolist())
    
    # List of potential numeric columns by various names
    stat_columns = {
        'SRS': ['SRS'],
        'SOS': ['SOS'],
        'Win_Pct': ['Win_Pct', 'W-L%', 'Win%', 'WinPct'],
        'Points_For': ['Points_For', 'Tm.', 'PTS', 'Points'],
        'Points_Against': ['Points_Against', 'Opp.', 'PA', 'PTS_A'],
        'FG_Pct': ['FG_Pct', 'FG%'],
        '3P_Pct': ['3P_Pct', '3P%'],
        'FT_Pct': ['FT_Pct', 'FT%'],
        'Wins': ['Wins', 'W'],
        'Losses': ['Losses', 'L'],
        'Games': ['Games', 'G'],
        'FG': ['FG'],
        'FGA': ['FGA'],
        '3P': ['3P'],
        '3PA': ['3PA'],
        'FT': ['FT'],
        'FTA': ['FTA'],
        'ORB': ['ORB'],
        'TRB': ['TRB'],
        'AST': ['AST'],
        'STL': ['STL'],
        'BLK': ['BLK'],
        'TOV': ['TOV'],
        'PF': ['PF']
    }
    
    # Create standardized column names
    for std_name, variations in stat_columns.items():
        for var in variations:
            if var in df.columns and std_name not in df.columns:
                df[std_name] = df[var]
                break
    
    # Calculate percentages if missing
    if 'FG_Pct' not in df.columns and 'FG' in df.columns and 'FGA' in df.columns:
        df['FG_Pct'] = df['FG'] / df['FGA']
    
    if '3P_Pct' not in df.columns and '3P' in df.columns and '3PA' in df.columns:
        df['3P_Pct'] = df['3P'] / df['3PA']
    
    if 'FT_Pct' not in df.columns and 'FT' in df.columns and 'FTA' in df.columns:
        df['FT_Pct'] = df['FT'] / df['FTA']
    
    # Select columns for analysis (only use ones that exist in the dataframe)
    analysis_cols = [col for col in stat_columns.keys() if col in df.columns]
    
    print(f"\nAnalysis columns ({len(analysis_cols)}):", analysis_cols)
    
    # Handle missing values
    for col in analysis_cols:
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create a copy with only the needed columns
    analysis_df = df[analysis_cols + ['School', 'Season', 'Is_Champion']].copy()
    
    # Drop rows with missing values
    orig_count = len(analysis_df)
    analysis_df = analysis_df.dropna(subset=analysis_cols)
    dropped_count = orig_count - len(analysis_df)
    print(f"Dropped {dropped_count} rows with missing values out of {orig_count} total rows")
    
    # Ensure we still have champions
    champ_count = analysis_df['Is_Champion'].sum()
    print(f"Champions remaining after cleaning: {champ_count}")
    
    return analysis_df, analysis_cols

# Perform PCA
def run_pca(df, features):
    """Run Principal Component Analysis"""
    print(f"\nRunning PCA with {len(features)} features on {len(df)} rows")
    
    # Ensure we have data
    if len(df) == 0:
        print("Error: No data available for PCA")
        return None, None, None
    
    # Extract features
    X = df[features].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(
        data=X_pca[:, :2],
        columns=['PC1', 'PC2']
    )
    pca_df['School'] = df['School'].values
    pca_df['Season'] = df['Season'].values
    pca_df['Is_Champion'] = df['Is_Champion'].values
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    
    return pca_df, pca, explained_variance

# Analyze feature importance
def analyze_features(pca, features, explained_variance):
    """Analyze which features are most important"""
    # Get loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create loadings dataframe
    loadings_df = pd.DataFrame(
        loadings[:, :2],
        columns=['PC1', 'PC2'],
        index=features
    )
    
    # Calculate importance
    loadings_df['PC1_abs'] = abs(loadings_df['PC1'])
    loadings_df['PC2_abs'] = abs(loadings_df['PC2'])
    loadings_df['Combined'] = loadings_df['PC1_abs'] + loadings_df['PC2_abs']
    
    # Sort by importance
    loadings_df = loadings_df.sort_values('Combined', ascending=False)
    
    print("\nMost important features:")
    print(loadings_df[['PC1', 'PC2', 'Combined']].head(10))
    
    return loadings_df

# Create visualizations
def create_visualizations(pca_df, loadings_df, explained_variance):
    """Create visualization plots"""
    
    # 1. PCA scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot regular teams
    plt.scatter(
        pca_df.loc[~pca_df['Is_Champion'], 'PC1'],
        pca_df.loc[~pca_df['Is_Champion'], 'PC2'],
        c='blue', alpha=0.3, label='Regular Teams'
    )
    
    # Plot champions
    champion_df = pca_df[pca_df['Is_Champion']]
    if not champion_df.empty:
        plt.scatter(
            champion_df['PC1'],
            champion_df['PC2'],
            c='red', s=100, marker='*', label='Champions'
        )
        
        # Add labels for champions
        for _, row in champion_df.iterrows():
            plt.annotate(
                f"{row['School']} ({row['Season']})",
                (row['PC1'], row['PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    # Customize plot
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.title('PCA of College Basketball Teams (2000-Present)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('pca_teams_plot.png')
    plt.close()
    
    # 2. Loadings plot
    plt.figure(figsize=(12, 10))
    
    # Plot arrows for each feature
    for feature in loadings_df.index:
        plt.arrow(
            0, 0,
            loadings_df.loc[feature, 'PC1'],
            loadings_df.loc[feature, 'PC2'],
            head_width=0.02, head_length=0.02,
            fc='blue', ec='blue'
        )
        plt.text(
            loadings_df.loc[feature, 'PC1'] * 1.1,
            loadings_df.loc[feature, 'PC2'] * 1.1,
            feature,
            fontsize=9
        )
    
    # Add reference circles
    circle1 = plt.Circle((0, 0), 0.5, color='r', fill=False)
    circle2 = plt.Circle((0, 0), 1.0, color='r', fill=False)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    
    # Customize plot
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.title('Feature Loadings: What Explains Basketball Success')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('pca_loadings_plot.png')
    plt.close()
    
    print("\nSaved visualizations:")
    print("- pca_teams_plot.png")
    print("- pca_loadings_plot.png")

# Compare champions vs non-champions
def compare_champions(df, features):
    """Compare statistics between champions and non-champions"""
    if df['Is_Champion'].sum() == 0:
        print("No champions identified for comparison")
        return
    
    # Calculate means
    champ_stats = df[df['Is_Champion']][features].mean()
    regular_stats = df[~df['Is_Champion']][features].mean()
    
    # Calculate percentage differences
    pct_diff = (champ_stats - regular_stats) / regular_stats * 100
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Champions': champ_stats,
        'Regular Teams': regular_stats,
        'Pct Difference': pct_diff
    })
    
    # Sort by absolute difference
    comparison = comparison.sort_values('Pct Difference', key=abs, ascending=False)
    
    print("\nChampion vs Regular Team Comparison:")
    print(comparison)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    # Get top differences (limit to 10 for clarity)
    top_diff = comparison.head(10)
    
    # Plot differences
    sns.barplot(x=top_diff['Pct Difference'], y=top_diff.index)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel('Percentage Difference (%)')
    plt.ylabel('Features')
    plt.title('How Champions Differ from Regular Teams')
    plt.grid(axis='x', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('champion_differences.png')
    plt.close()
    
    print("- champion_differences.png")
    
    return comparison

def main():
    """Main function"""
    print("Loading all season data...")
    all_data = load_all_seasons()
    
    if all_data is None:
        print("Error: Failed to load data")
        return
    
    print("\nIdentifying champions...")
    all_data = mark_champions(all_data)
    
    print("\nPreparing data for analysis...")
    analysis_df, features = prepare_data(all_data)
    
    # Ensure we have data after preparation
    if len(analysis_df) == 0:
        print("Error: No data available after preparation")
        return
    
    print("\nRunning PCA analysis...")
    pca_df, pca, explained_variance = run_pca(analysis_df, features)
    
    if pca is None:
        print("Error: PCA analysis failed")
        return
    
    print("\nAnalyzing feature importance...")
    loadings_df = analyze_features(pca, features, explained_variance)
    
    print("\nCreating visualizations...")
    create_visualizations(pca_df, loadings_df, explained_variance)
    
    print("\nComparing champions vs regular teams...")
    compare_champions(analysis_df, features)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()