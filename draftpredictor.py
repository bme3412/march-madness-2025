#!/usr/bin/env python3
"""
Improved NCAA Tournament Bracket Predictor and Trainer with Barttorvik Home/Away Data.

Usage:
  To train models: python draftpredictor.py --train
  To predict bracket: python draftpredictor.py [--predictions PATH] [--bracket PATH] [--output PATH]
"""

import os
import glob
import re
from typing import List, Tuple, Dict, Any, Optional, Union
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------------
# PyTorch Model Definitions
# -----------------------------
class BasicChampionPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        """
        A basic two-layer neural network for champion prediction.
        """
        super(BasicChampionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Two classes: not champion (0) and champion (1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class ImprovedPredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        """
        An improved model with batch normalization and additional dropout layers.
        """
        super(ImprovedPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepPredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        """
        A deeper model for handling the additional Barttorvik features.
        """
        super(DeepPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def train_torch_model(model: nn.Module, criterion: Any, optimizer: optim.Optimizer,
                      train_loader: DataLoader, val_loader: DataLoader,
                      num_epochs: int = 100, patience: int = 20) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train a PyTorch model with early stopping based on validation loss.
    """
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    training_losses, validation_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        training_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        validation_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, training_losses, validation_losses


def evaluate_model(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model and return metrics.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].numpy()
        predictions = (probabilities >= threshold).astype(int)
    y_true = y_tensor.numpy()
    prec = precision_score(y_true, predictions) if np.sum(predictions) and np.sum(y_true) else 0
    rec = recall_score(y_true, predictions) if np.sum(predictions) and np.sum(y_true) else 0
    f1 = f1_score(y_true, predictions) if np.sum(predictions) and np.sum(y_true) else 0
    auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else float('nan')
    cm = confusion_matrix(y_true, predictions)
    return {
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'probabilities': probabilities,
        'predictions': predictions
    }

# -----------------------------
# Data Loading and Preprocessing Functions
# -----------------------------

def plot_feature_importance(feature_names: List[str], 
                            importance_values: np.ndarray, 
                            title: str,
                            output_file: str) -> None:
    """
    Plot feature importance values.
    """
    sorted_idx = importance_values.argsort()
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(sorted_idx)), importance_values[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved feature importance plot to {output_file}")
def load_seasons_summary(file_path: str) -> pd.DataFrame:
    """
    Load the seasons summary CSV, extract the season year and champion name, and return a sorted DataFrame.
    """
    df = pd.read_csv(file_path)
    print(f"Columns in seasons summary: {df.columns.tolist()}")
    if 'full_url' in df.columns:
        df['Year'] = df['full_url'].str.extract(r'/men/(\d{4})\.html').astype(int)
    elif 'URL' in df.columns:
        df['Year'] = df['URL'].str.extract(r'/men/(\d{4})\.html').astype(int)
    elif 'Season' in df.columns:
        df['Year'] = df['Season'].apply(lambda x: int(re.findall(r'\d{4}', str(x))[-1]))
    else:
        raise ValueError("Expected column ('full_url', 'URL', or 'Season') not found in the seasons summary file.")
    return df[['Year', 'Champion']].sort_values('Year', ascending=True)


def load_school_stats(data_dir: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load all school stats CSV files from data_dir for years in the given range.
    """
    pattern = os.path.join(data_dir, 'school_stats_*.csv')
    all_files = glob.glob(pattern)
    df_list = []
    for file in sorted(all_files):
        match = re.search(r'school_stats_(\d{4})\.csv', file)
        if match:
            year = int(match.group(1))
            if start_year <= year <= end_year:
                print(f"Loading data for year {year}...")
                df_temp = pd.read_csv(file)
                if 'Season' not in df_temp.columns:
                    df_temp['Season'] = year
                df_temp['Source_File'] = os.path.basename(file)
                df_list.append(df_temp)
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


def load_barttorvik_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Barttorvik Home and Away CSV files.
    
    Returns:
        Tuple containing (home_data, away_data)
    """
    # Load Barttorvik Home data
    home_file = os.path.join(data_dir, 'Barttorvik Home.csv')
    if os.path.exists(home_file):
        home_data = pd.read_csv(home_file)
        print(f"Loaded Barttorvik Home data with {len(home_data)} rows")
    else:
        print(f"Warning: Barttorvik Home file not found at {home_file}")
        home_data = pd.DataFrame()
    
    # Load Barttorvik Away data
    away_file = os.path.join(data_dir, 'Barttorvik Away.csv')
    if os.path.exists(away_file):
        away_data = pd.read_csv(away_file)
        print(f"Loaded Barttorvik Away data with {len(away_data)} rows")
    else:
        print(f"Warning: Barttorvik Away file not found at {away_file}")
        away_data = pd.DataFrame()
    
    return home_data, away_data


def preprocess_barttorvik_data(home_data: pd.DataFrame, away_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and combine Barttorvik home and away data.
    
    Args:
        home_data: DataFrame with Barttorvik home metrics
        away_data: DataFrame with Barttorvik away metrics
        
    Returns:
        Combined DataFrame with processed metrics
    """
    if home_data.empty or away_data.empty:
        print("Warning: Barttorvik data is empty, skipping preprocessing")
        return pd.DataFrame()
    
    # Make copies to avoid modifying originals
    home_df = home_data.copy()
    away_df = away_data.copy()
    
    # Ensure consistent team names
    if 'Team' in home_df.columns:
        home_df['School'] = home_df['Team']
    if 'Team' in away_df.columns:
        away_df['School'] = away_df['Team']
    
    # Ensure year/season column consistency
    if 'Year' in home_df.columns and 'Season' not in home_df.columns:
        home_df['Season'] = home_df['Year']
    if 'Year' in away_df.columns and 'Season' not in away_df.columns:
        away_df['Season'] = away_df['Year']
        
    # Add location identifier
    home_df['Location'] = 'Home'
    away_df['Location'] = 'Away'
    
    # Rename columns to avoid conflicts
    for col in home_df.columns:
        if col not in ['School', 'Team', 'Season', 'Year', 'Location']:
            home_df.rename(columns={col: f'Home_{col}'}, inplace=True)
    
    for col in away_df.columns:
        if col not in ['School', 'Team', 'Season', 'Year', 'Location']:
            away_df.rename(columns={col: f'Away_{col}'}, inplace=True)
    
    # Merge home and away data
    merge_cols = []
    for col in ['School', 'Season']:
        if col in home_df.columns and col in away_df.columns:
            merge_cols.append(col)
    
    if not merge_cols:
        print("Error: No common columns found for merging Barttorvik data")
        return pd.DataFrame()
    
    try:
        barttorvik_combined = pd.merge(home_df, away_df, on=merge_cols, how='outer')
        print(f"Combined Barttorvik data has {len(barttorvik_combined)} rows")
        return barttorvik_combined
    except Exception as e:
        print(f"Error merging Barttorvik data: {e}")
        return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features based on available statistics.
    """
    print("Engineering additional features...")
    if 'Points_For' in df.columns and 'Points_Against' in df.columns:
        df['Point_Differential'] = df['Points_For'] - df['Points_Against']
    if 'Point_Differential' in df.columns and 'Games' in df.columns:
        df['Win_Margin'] = df['Point_Differential'] / df['Games']
    if 'SRS' in df.columns and 'SOS' in df.columns:
        df['SRS_SOS_Product'] = df['SRS'] * df['SOS']
    if 'FG' in df.columns and 'FGA' in df.columns:
        df['FG_Pct'] = df['FG'] / df['FGA'].replace(0, np.nan)
    if 'FT' in df.columns and 'FTA' in df.columns:
        df['FT_Pct'] = df['FT'] / df['FTA'].replace(0, np.nan)
    if '3P' in df.columns and '3PA' in df.columns:
        df['3P_Pct'] = df['3P'] / df['3PA'].replace(0, np.nan)
    if 'FG' in df.columns and 'FGA' in df.columns and '3P' in df.columns:
        df['EFG_Pct'] = (df['FG'] + 0.5 * df['3P']) / df['FGA'].replace(0, np.nan)
    if 'TOV' in df.columns and 'FGA' in df.columns and 'FTA' in df.columns:
        denominator = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
        df['TO_Rate'] = df['TOV'] / denominator.replace(0, np.nan)
    if 'ORB' in df.columns and 'TRB' in df.columns:
        df['ORB_Pct'] = df['ORB'] / df['TRB'].replace(0, np.nan)
    if 'Points_For' in df.columns and 'Games' in df.columns:
        df['Scoring_Efficiency'] = df['Points_For'] / df['Games']
    if 'Points_Against' in df.columns and 'Games' in df.columns:
        df['Defensive_Efficiency'] = df['Points_Against'] / df['Games']
    if 'Points_For' in df.columns and 'Points_Against' in df.columns and 'Games' in df.columns:
        df['Net_Efficiency'] = (df['Points_For'] - df['Points_Against']) / df['Games']
    if '3P' in df.columns and 'Points_For' in df.columns:
        df['Three_Point_Reliance'] = (df['3P'] * 3) / df['Points_For'].replace(0, np.nan)
        
    # Engineer Barttorvik-specific features
    barttorvik_cols = [col for col in df.columns if 'Home_' in col or 'Away_' in col]
    if barttorvik_cols:
        print("Engineering Barttorvik-specific features...")
        
        # Calculate home/away differentials for matching metrics
        home_metrics = [col for col in df.columns if col.startswith('Home_')]
        away_metrics = [col for col in df.columns if col.startswith('Away_')]
        
        for home_col in home_metrics:
            metric_name = home_col.replace('Home_', '')
            away_col = f'Away_{metric_name}'
            
            if away_col in away_metrics:
                diff_col = f'HomeAway_Diff_{metric_name}'
                df[diff_col] = df[home_col] - df[away_col]
                
                ratio_col = f'HomeAway_Ratio_{metric_name}'
                # Avoid division by zero
                df[ratio_col] = df[home_col] / df[away_col].replace(0, np.nan)
                df[ratio_col] = df[ratio_col].replace([np.inf, -np.inf], np.nan)
    
    # Add NCAA Tournament history features
    if 'NCAA_Tournament' in df.columns:
        if df['NCAA_Tournament'].dtype == bool:
            df['NCAA_Tournament'] = df['NCAA_Tournament'].astype(int)
        tournament_history = df[['School', 'Season', 'NCAA_Tournament']].copy()
        tournament_history.sort_values(['School', 'Season'], inplace=True)
        tournament_history['Previous_Year_Tournament'] = tournament_history.groupby('School')['NCAA_Tournament'].shift(1)
        tournament_history['Two_Years_Ago_Tournament'] = tournament_history.groupby('School')['NCAA_Tournament'].shift(2)
        tournament_history['Three_Years_Ago_Tournament'] = tournament_history.groupby('School')['NCAA_Tournament'].shift(3)
        tournament_history.fillna(0, inplace=True)
        tournament_history['Tournament_Streak'] = (
            tournament_history['Previous_Year_Tournament'] +
            tournament_history['Two_Years_Ago_Tournament'] +
            tournament_history['Three_Years_Ago_Tournament']
        )
        tournament_columns = ['School', 'Season', 'Previous_Year_Tournament',
                              'Two_Years_Ago_Tournament', 'Three_Years_Ago_Tournament',
                              'Tournament_Streak']
        df = pd.merge(df, tournament_history[tournament_columns], on=['School', 'Season'], how='left')
        for col in ['Previous_Year_Tournament', 'Two_Years_Ago_Tournament', 'Three_Years_Ago_Tournament', 'Tournament_Streak']:
            df[col] = df[col].fillna(0)
            
    # Fill NaN values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            if 'Pct' in col or 'Ratio' in col:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
                
    return df


def preprocess_data(df: pd.DataFrame, seasons_summary: pd.DataFrame,
                    base_features: List[str],
                    engineered_features: List[str] = [],
                    barttorvik_features: List[str] = []) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Merge season summary with school stats and create the binary target variable.
    Now includes Barttorvik features.
    """
    print("Preprocessing data...")
    df = pd.merge(df, seasons_summary, left_on='Season', right_on='Year', how='left')
    
    def is_champion(row: pd.Series) -> int:
        if pd.isna(row['Champion']):
            return 0
        champ_name = str(row['Champion']).lower()
        school_name = str(row['School']).lower()
        if champ_name == school_name or champ_name in school_name or school_name in champ_name:
            return 1
        abbreviations = {
            'uconn': ['connecticut'],
            'unc': ['north carolina'],
            'ucla': ['california, los angeles'],
            'smu': ['southern methodist'],
            'lsu': ['louisiana state'],
            'vcu': ['virginia commonwealth'],
            'ucf': ['central florida'],
            'st': ['saint'],
            'st.': ['saint']
        }
        for abbr, variations in abbreviations.items():
            if (abbr in champ_name and any(var in school_name for var in variations)) or \
               (abbr in school_name and any(var in champ_name for var in variations)):
                return 1
        return 0

    df['Label'] = df.apply(is_champion, axis=1)
    
    # Collect all features
    all_features = []
    
    # Add base features if they exist in the dataframe
    for feat in base_features:
        if feat in df.columns:
            all_features.append(feat)
    
    # Add engineered features if they exist
    for feat in engineered_features:
        if feat in df.columns:
            all_features.append(feat)
    
    # Add Barttorvik features if they exist
    for feat in barttorvik_features:
        if feat in df.columns:
            all_features.append(feat)
    
    print(f"Total features selected: {len(all_features)}")
    
    # Drop rows with missing feature values
    df_cleaned = df.dropna(subset=all_features)
    if len(df_cleaned) == 0:
        print("WARNING: No data left after dropping rows with missing features!")
        print("Available columns:", df.columns.tolist())
        print("Requested features:", all_features)
        fallback_features = ['Win_Pct', 'SRS', 'SOS']
        all_features = [f for f in fallback_features if f in df.columns]
        df_cleaned = df.dropna(subset=all_features)
        if len(df_cleaned) == 0:
            print("ERROR: Still no data after fallback. Check your data files.")
            return np.array([]).reshape(0, len(all_features)), np.array([]), df, all_features
    
    print(f"After dropping NAs, remaining rows: {len(df_cleaned)}")
    X = df_cleaned[all_features].values.astype(np.float32)
    y = df_cleaned['Label'].values.astype(np.int64)
    
    return X, y, df_cleaned, all_features

# -----------------------------
# Bracket Prediction Functions
# (These remain largely unchanged)
# -----------------------------
def normalize_team_name(name: str) -> str:
    if not name:
        return ""
    replacements = {
        "Saint": "St.",
        "St.": "St",
        "State": "St",
        "College": "",
        "University": "",
        "(": "",
        ")": "",
        "-": " ",
        ".": "",
        "'": "",
        "Mount": "Mt",
        "North Carolina": "UNC",
        "Southern California": "USC",
        "California": "Cal",
        "Connecticut": "UConn",
        "Alabama Birmingham": "UAB",
        "Louisiana State": "LSU",
        "Mississippi": "Ole Miss",
        "Pennsylvania": "Penn",
        "Central Florida": "UCF",
    }
    name = name.strip().lower()
    for old, new in replacements.items():
        name = name.replace(old.lower(), new.lower())
    for word in ["university", "college", "team"]:
        name = name.replace(word, "")
    return name.strip()


def find_team_in_predictions(team_name: str, predictions_df: pd.DataFrame) -> Optional[pd.Series]:
    team_data = predictions_df[predictions_df['School'] == team_name]
    if not team_data.empty:
        return team_data.iloc[0]
    normalized_name = normalize_team_name(team_name)
    for _, row in predictions_df.iterrows():
        if normalize_team_name(row['School']) == normalized_name:
            return row
    for _, row in predictions_df.iterrows():
        row_normalized = normalize_team_name(row['School'])
        if normalized_name in row_normalized or row_normalized in normalized_name:
            return row
    for _, row in predictions_df.iterrows():
        team_first_word = normalized_name.split()[0] if normalized_name.split() else ""
        row_first_word = normalize_team_name(row['School']).split()[0] if normalize_team_name(row['School']).split() else ""
        if team_first_word and row_first_word and team_first_word == row_first_word:
            return row
    return None


def matchup_prediction(team1_data: pd.Series, team2_data: pd.Series, 
                       seed1: int, seed2: int, round_num: int) -> Tuple[str, int, float]:
    # Enhanced key metrics including Barttorvik metrics if available
    key_metrics = {
        'SRS': 0.20,
        'Win_Pct': 0.15,
        'SOS': 0.05,
        'Point_Differential': 0.05,
        'Champion_Probability': 0.35,
    }
    
    # Add Barttorvik metrics if available
    barttorvik_metrics = {
        'HomeAway_Diff_AdjO': 0.05,
        'HomeAway_Diff_AdjD': 0.05,
        'HomeAway_Ratio_WinPct': 0.10,
    }
    
    # Check if Barttorvik metrics are available for both teams
    has_barttorvik = all(
        metric in team1_data and metric in team2_data 
        for metric in barttorvik_metrics.keys()
    )
    
    # If Barttorvik metrics are available, incorporate them
    if has_barttorvik:
        key_metrics.update(barttorvik_metrics)
    
    team1_score = 0
    team2_score = 0
    for metric, weight in key_metrics.items():
        if metric in team1_data and metric in team2_data:
            team1_val = team1_data[metric]
            team2_val = team2_data[metric]
            if pd.isna(team1_val) or pd.isna(team2_val):
                continue
            max_val = max(team1_val, team2_val)
            min_val = min(team1_val, team2_val)
            if max_val == min_val:
                team1_score += 0.5 * weight
                team2_score += 0.5 * weight
            else:
                team1_normalized = (team1_val - min_val) / (max_val - min_val)
                team2_normalized = (team2_val - min_val) / (max_val - min_val)
                team1_score += team1_normalized * weight
                team2_score += team2_normalized * weight
    
    # Adjust for seed differential (less impact in later rounds)
    seed_factor = max(0, 0.2 - (round_num - 1) * 0.03)
    seed_diff = seed2 - seed1
    seed_adjustment = seed_diff * seed_factor
    team1_score += seed_adjustment
    
    # Add randomness factor (decreases in later rounds)
    upset_factor = np.random.normal(0, max(0.05, 0.2 - (round_num * 0.03)))
    team1_score += upset_factor
    
    # Get champion probabilities
    team1_prob = team1_data.get('Champion_Probability', 0)
    team2_prob = team2_data.get('Champion_Probability', 0)
    
    if team1_score > team2_score:
        return team1_data['School'], seed1, team1_prob
    else:
        return team2_data['School'], seed2, team2_prob


def parse_bracket(bracket_file: str) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    regions = {}
    current_region = None
    with open(bracket_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if "REGION" in line:
            current_region = line.split("REGION")[0].strip()
            regions[current_region] = []
        elif line and current_region and "vs" in line:
            parts = line.split("vs")
            team1_parts = parts[0].strip().split(" ", 1)
            team2_parts = parts[1].strip().split(" ", 1)
            seed1 = int(team1_parts[0])
            team1 = team1_parts[1] if len(team1_parts) > 1 else "Unknown"
            seed2 = int(team2_parts[0])
            team2 = team2_parts[1] if len(team2_parts) > 1 else "Unknown"
            regions[current_region].append({
                "team1": team1,
                "seed1": seed1,
                "team2": team2,
                "seed2": seed2
            })
    return regions


def simulate_bracket(regions: Dict[str, List[Dict[str, Union[str, int]]]],
                     predictions_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simulate the entire tournament, round by round, and return the results for each round.
    
    Returns a dictionary like:
    {
      "Round of 64": [...],
      "Round of 32": [...],
      "Sweet 16": [...],
      "Elite 8": [...],
      "Final 4": [...],
      "Championship": [...],
      "Champion": [...]
    }
    """
    results = {
        "Round of 64": [],
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final 4": [],
        "Championship": [],
        "Champion": []
    }
    
    # -----------------------------
    # ROUND OF 64
    # -----------------------------
    round_of_32 = {}
    
    for region_name, matchups in regions.items():
        round_of_32[region_name] = []
        
        # For each pairing in the Round of 64
        for matchup in matchups:
            team1, seed1 = matchup["team1"], matchup["seed1"]
            team2, seed2 = matchup["team2"], matchup["seed2"]
            
            # Lookup or fallback for each team
            team1_data = find_team_in_predictions(team1, predictions_df)
            team2_data = find_team_in_predictions(team2, predictions_df)
            if team1_data is None:
                print(f"Warning: {team1} not found in predictions. Using seed-based fallback.")
                team1_data = pd.Series({
                    'School': team1,
                    'SRS': max(20 - seed1, 0),
                    'Win_Pct': max(1.0 - (seed1 * 0.05), 0.5),
                    'Champion_Probability': max(0.16 - (seed1 * 0.01), 0.001)
                })
            if team2_data is None:
                print(f"Warning: {team2} not found in predictions. Using seed-based fallback.")
                team2_data = pd.Series({
                    'School': team2,
                    'SRS': max(20 - seed2, 0),
                    'Win_Pct': max(1.0 - (seed2 * 0.05), 0.5),
                    'Champion_Probability': max(0.16 - (seed2 * 0.01), 0.001)
                })
            
            # Determine winner
            winner, winner_seed, winner_prob = matchup_prediction(team1_data, team2_data,
                                                                  seed1, seed2, round_num=1)
            
            # Store Round of 64 result
            results["Round of 64"].append({
                "region": region_name,
                "team1": team1, "seed1": seed1,
                "team2": team2, "seed2": seed2,
                "winner": winner,
                "winner_seed": winner_seed,
                "winner_prob": winner_prob
            })
            
            # Winner advances to next round in the same region
            round_of_32[region_name].append({"team": winner, "seed": winner_seed})
    
    # -----------------------------
    # ROUND OF 32
    # -----------------------------
    sweet_16 = {}
    
    for region_name, winners in round_of_32.items():
        sweet_16[region_name] = []
        
        # The winners are stored in a list. We pair them up 2 by 2.
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                team1, seed1 = winners[i]["team"], winners[i]["seed"]
                team2, seed2 = winners[i+1]["team"], winners[i+1]["seed"]
                
                # Lookup or fallback for each team
                team1_data = find_team_in_predictions(team1, predictions_df)
                team2_data = find_team_in_predictions(team2, predictions_df)
                if team1_data is None:
                    team1_data = pd.Series({
                        'School': team1,
                        'SRS': max(20 - seed1, 0),
                        'Win_Pct': max(1.0 - (seed1 * 0.05), 0.5),
                        'Champion_Probability': max(0.16 - (seed1 * 0.01), 0.001)
                    })
                if team2_data is None:
                    team2_data = pd.Series({
                        'School': team2,
                        'SRS': max(20 - seed2, 0),
                        'Win_Pct': max(1.0 - (seed2 * 0.05), 0.5),
                        'Champion_Probability': max(0.16 - (seed2 * 0.01), 0.001)
                    })
                
                # Determine winner
                winner, winner_seed, winner_prob = matchup_prediction(team1_data, team2_data,
                                                                      seed1, seed2, round_num=2)
                
                # Store Round of 32 result
                results["Round of 32"].append({
                    "region": region_name,
                    "team1": team1, "seed1": seed1,
                    "team2": team2, "seed2": seed2,
                    "winner": winner,
                    "winner_seed": winner_seed,
                    "winner_prob": winner_prob
                })
                
                # Winner advances
                sweet_16[region_name].append({"team": winner, "seed": winner_seed})
    
    # -----------------------------
    # SWEET 16
    # -----------------------------
    elite_8 = {}
    
    for region_name, winners in sweet_16.items():
        elite_8[region_name] = []
        
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                team1, seed1 = winners[i]["team"], winners[i]["seed"]
                team2, seed2 = winners[i+1]["team"], winners[i+1]["seed"]
                
                team1_data = find_team_in_predictions(team1, predictions_df)
                team2_data = find_team_in_predictions(team2, predictions_df)
                if team1_data is None:
                    team1_data = pd.Series({
                        'School': team1,
                        'SRS': max(20 - seed1, 0),
                        'Win_Pct': max(1.0 - (seed1 * 0.05), 0.5),
                        'Champion_Probability': max(0.16 - (seed1 * 0.01), 0.001)
                    })
                if team2_data is None:
                    team2_data = pd.Series({
                        'School': team2,
                        'SRS': max(20 - seed2, 0),
                        'Win_Pct': max(1.0 - (seed2 * 0.05), 0.5),
                        'Champion_Probability': max(0.16 - (seed2 * 0.01), 0.001)
                    })
                
                winner, winner_seed, winner_prob = matchup_prediction(team1_data, team2_data,
                                                                      seed1, seed2, round_num=3)
                
                results["Sweet 16"].append({
                    "region": region_name,
                    "team1": team1, "seed1": seed1,
                    "team2": team2, "seed2": seed2,
                    "winner": winner,
                    "winner_seed": winner_seed,
                    "winner_prob": winner_prob
                })
                
                elite_8[region_name].append({"team": winner, "seed": winner_seed})
    
    # -----------------------------
    # ELITE 8
    # -----------------------------
    final_4 = []
    region_order = list(elite_8.keys())
    
    for region_name in region_order:
        winners = elite_8[region_name]
        if len(winners) >= 2:
            team1, seed1 = winners[0]["team"], winners[0]["seed"]
            team2, seed2 = winners[1]["team"], winners[1]["seed"]
            
            team1_data = find_team_in_predictions(team1, predictions_df)
            team2_data = find_team_in_predictions(team2, predictions_df)
            if team1_data is None:
                team1_data = pd.Series({
                    'School': team1,
                    'SRS': max(20 - seed1, 0),
                    'Win_Pct': max(1.0 - (seed1 * 0.05), 0.5),
                    'Champion_Probability': max(0.16 - (seed1 * 0.01), 0.001)
                })
            if team2_data is None:
                team2_data = pd.Series({
                    'School': team2,
                    'SRS': max(20 - seed2, 0),
                    'Win_Pct': max(1.0 - (seed2 * 0.05), 0.5),
                    'Champion_Probability': max(0.16 - (seed2 * 0.01), 0.001)
                })
            
            winner, winner_seed, winner_prob = matchup_prediction(team1_data, team2_data,
                                                                  seed1, seed2, round_num=4)
            
            results["Elite 8"].append({
                "region": region_name,
                "team1": team1, "seed1": seed1,
                "team2": team2, "seed2": seed2,
                "winner": winner,
                "winner_seed": winner_seed,
                "winner_prob": winner_prob
            })
            
            final_4.append({"team": winner, "seed": winner_seed, "region": region_name})
    
    # -----------------------------
    # FINAL FOUR (semifinals)
    # -----------------------------
    championship = []
    
    # Typically, the Final Four bracket pairs the winners of two regions vs. winners of the other two.
    # For simplicity, let's assume final_4[0] vs. final_4[1], final_4[2] vs. final_4[3], etc.
    
    for i in range(0, len(final_4), 2):
        if i + 1 < len(final_4):
            team1, seed1 = final_4[i]["team"], final_4[i]["seed"]
            team2, seed2 = final_4[i+1]["team"], final_4[i+1]["seed"]
            
            team1_data = find_team_in_predictions(team1, predictions_df)
            team2_data = find_team_in_predictions(team2, predictions_df)
            if team1_data is None:
                team1_data = pd.Series({
                    'School': team1,
                    'SRS': max(20 - seed1, 0),
                    'Win_Pct': max(1.0 - (seed1 * 0.05), 0.5),
                    'Champion_Probability': max(0.16 - (seed1 * 0.01), 0.001)
                })
            if team2_data is None:
                team2_data = pd.Series({
                    'School': team2,
                    'SRS': max(20 - seed2, 0),
                    'Win_Pct': max(1.0 - (seed2 * 0.05), 0.5),
                    'Champion_Probability': max(0.16 - (seed2 * 0.01), 0.001)
                })
            
            winner, winner_seed, winner_prob = matchup_prediction(team1_data, team2_data,
                                                                  seed1, seed2, round_num=5)
            
            results["Final 4"].append({
                "matchup": f"{final_4[i]['region']} vs {final_4[i+1]['region']}",
                "team1": team1, "seed1": seed1,
                "team2": team2, "seed2": seed2,
                "winner": winner,
                "winner_seed": winner_seed,
                "winner_prob": winner_prob
            })
            
            championship.append({"team": winner, "seed": winner_seed})
    
    # -----------------------------
    # CHAMPIONSHIP
    # -----------------------------
    if len(championship) >= 2:
        team1, seed1 = championship[0]["team"], championship[0]["seed"]
        team2, seed2 = championship[1]["team"], championship[1]["seed"]
        
        team1_data = find_team_in_predictions(team1, predictions_df)
        team2_data = find_team_in_predictions(team2, predictions_df)
        if team1_data is None:
            team1_data = pd.Series({
                'School': team1,
                'SRS': max(20 - seed1, 0),
                'Win_Pct': max(1.0 - (seed1 * 0.05), 0.5),
                'Champion_Probability': max(0.16 - (seed1 * 0.01), 0.001)
            })
        if team2_data is None:
            team2_data = pd.Series({
                'School': team2,
                'SRS': max(20 - seed2, 0),
                'Win_Pct': max(1.0 - (seed2 * 0.05), 0.5),
                'Champion_Probability': max(0.16 - (seed2 * 0.01), 0.001)
            })
        
        winner, winner_seed, winner_prob = matchup_prediction(team1_data, team2_data,
                                                              seed1, seed2, round_num=6)
        
        results["Championship"].append({
            "team1": team1, "seed1": seed1,
            "team2": team2, "seed2": seed2,
            "winner": winner,
            "winner_seed": winner_seed,
            "winner_prob": winner_prob
        })
        
        results["Champion"].append({
            "team": winner,
            "seed": winner_seed,
            "probability": winner_prob
        })
    
    return results


def print_results(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Print tournament predictions in a clear, easy-to-read format.
    """
    print("\n===== TOURNAMENT PREDICTIONS =====\n")
    
    # Round of 64
    print("==== ROUND OF 64 ====")
    for region in ["SOUTH", "WEST", "EAST", "MIDWEST"]:
        print(f"\n{region} REGION:")
        region_matchups = [m for m in results["Round of 64"] if m["region"] == region]
        
        # Sort by seed to maintain bracket order
        region_matchups.sort(key=lambda x: x["seed1"])
        
        for matchup in region_matchups:
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team1"] else ""
            print(f"({matchup['seed1']}) {matchup['team1']} {winner_marker}")
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team2"] else ""
            print(f"({matchup['seed2']}) {matchup['team2']} {winner_marker}")
            print("---------------------")
    
    # Round of 32
    print("\n==== ROUND OF 32 ====")
    for region in ["SOUTH", "WEST", "EAST", "MIDWEST"]:
        print(f"\n{region} REGION:")
        region_matchups = [m for m in results["Round of 32"] if m["region"] == region]
        
        # Sort to maintain bracket order
        region_matchups.sort(key=lambda x: min(x["seed1"], x["seed2"]))
        
        for matchup in region_matchups:
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team1"] else ""
            print(f"({matchup['seed1']}) {matchup['team1']} {winner_marker}")
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team2"] else ""
            print(f"({matchup['seed2']}) {matchup['team2']} {winner_marker}")
            print("---------------------")
    
    # Sweet 16
    print("\n==== SWEET 16 ====")
    for region in ["SOUTH", "WEST", "EAST", "MIDWEST"]:
        print(f"\n{region} REGION:")
        region_matchups = [m for m in results["Sweet 16"] if m["region"] == region]
        
        for matchup in region_matchups:
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team1"] else ""
            print(f"({matchup['seed1']}) {matchup['team1']} {winner_marker}")
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team2"] else ""
            print(f"({matchup['seed2']}) {matchup['team2']} {winner_marker}")
            print("---------------------")
    
    # Elite 8
    print("\n==== ELITE 8 ====")
    for region in ["SOUTH", "WEST", "EAST", "MIDWEST"]:
        print(f"\n{region} REGION:")
        region_matchups = [m for m in results["Elite 8"] if m["region"] == region]
        
        for matchup in region_matchups:
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team1"] else ""
            print(f"({matchup['seed1']}) {matchup['team1']} {winner_marker}")
            winner_marker = "→ WINNER" if matchup["winner"] == matchup["team2"] else ""
            print(f"({matchup['seed2']}) {matchup['team2']} {winner_marker}")
            print("---------------------")
    
    # Final Four
    print("\n==== FINAL FOUR ====")
    for matchup in results["Final 4"]:
        print(f"MATCHUP: {matchup['matchup']}")
        winner_marker = "→ WINNER" if matchup["winner"] == matchup["team1"] else ""
        print(f"({matchup['seed1']}) {matchup['team1']} {winner_marker}")
        winner_marker = "→ WINNER" if matchup["winner"] == matchup["team2"] else ""
        print(f"({matchup['seed2']}) {matchup['team2']} {winner_marker}")
        print("---------------------")
    
    # Championship
    print("\n==== CHAMPIONSHIP ====")
    if results["Championship"]:
        championship = results["Championship"][0]
        winner_marker = "→ CHAMPION" if championship["winner"] == championship["team1"] else ""
        print(f"({championship['seed1']}) {championship['team1']} {winner_marker}")
        winner_marker = "→ CHAMPION" if championship["winner"] == championship["team2"] else ""
        print(f"({championship['seed2']}) {championship['team2']} {winner_marker}")
        print("---------------------")
    
    # Champion
    if results["Champion"]:
        champion = results["Champion"][0]
        print(f"\n🏆 TOURNAMENT CHAMPION: ({champion['seed']}) {champion['team']}")
        print(f"Championship Probability: {champion['probability']:.4f}")


def export_bracket(results: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
    all_picks = []
    for round_name, matchups in results.items():
        if round_name != "Champion":
            for matchup in matchups:
                round_data = {
                    "Round": round_name,
                    "Winner": matchup.get("winner", ""),
                    "Winner_Seed": matchup.get("winner_seed", ""),
                    "Team1": matchup.get("team1", ""),
                    "Seed1": matchup.get("seed1", ""),
                    "Team2": matchup.get("team2", ""),
                    "Seed2": matchup.get("seed2", ""),
                    "Region": matchup.get("region", matchup.get("matchup", "Championship"))
                }
                all_picks.append(round_data)
    picks_df = pd.DataFrame(all_picks)
    picks_df.to_csv(output_file, index=False)
    print(f"\nExported bracket to {output_file}")

# -----------------------------
# Prediction Function (Module Level)
# -----------------------------
# -----------------------------
# Training Function
# -----------------------------
def train_models() -> None:
    print("=== Training Mode ===")
    output_dir = "output"
    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    data_seasons_dir = "data-seasons"
    seasons_csv_path = os.path.join(data_seasons_dir, "cbb_seasons.csv")
    start_year, end_year = 2000, 2024
    
    seasons_summary = load_seasons_summary(seasons_csv_path)
    print(f"Seasons summary loaded. Total seasons: {len(seasons_summary)}")
    
    df_stats = load_school_stats(data_seasons_dir, start_year, end_year)
    print(f"Loaded school stats data with {len(df_stats)} rows from {start_year} to {end_year}.")
    
    # Load Barttorvik data (new in this version)
    barttorvik_dir = "data-advanced"
    home_data, away_data = load_barttorvik_data(barttorvik_dir)
    barttorvik_df = preprocess_barttorvik_data(home_data, away_data)
    
    # If Barttorvik data is available, merge it with the main dataset
    if not barttorvik_df.empty:
        print("Merging Barttorvik data with main dataset...")
        merge_cols = ['School', 'Season']
        df_stats = pd.merge(df_stats, barttorvik_df, on=merge_cols, how='left')
    
    df_stats = engineer_features(df_stats)
    
    base_features = ['Win_Pct', 'SRS', 'SOS', 'Points_For', 'Points_Against']
    engineered_features = [
        'Point_Differential', 'Win_Margin', 'SRS_SOS_Product', 'FG_Pct', 'FT_Pct', '3P_Pct', 
        'EFG_Pct', 'TO_Rate', 'ORB_Pct', 'Net_Efficiency', 'Scoring_Efficiency', 
        'Defensive_Efficiency', 'Three_Point_Reliance'
    ]
    
    # Include Barttorvik features if available
    barttorvik_features = []
    if not barttorvik_df.empty:
        barttorvik_features = [
            'HomeAway_Diff_AdjO', 'HomeAway_Diff_AdjD', 'HomeAway_Ratio_WinPct',
            'HomeAway_Diff_Tempo', 'HomeAway_Ratio_AdjEM'
        ]
        
        # Filter to only include features that actually exist in the dataframe
        barttorvik_features = [feat for feat in barttorvik_features if feat in df_stats.columns]
        print(f"Including {len(barttorvik_features)} Barttorvik features")
    
    X, y, df_merged, feature_names = preprocess_data(
        df_stats, seasons_summary, base_features, 
        engineered_features=engineered_features,
        barttorvik_features=barttorvik_features
    )
    
    print(f"After preprocessing, feature matrix shape: {X.shape}")
    if X.shape[0] == 0:
        print("Error: No training data available. Check your input files and feature selection.")
        return
    
    train_years = list(range(2000, 2020))
    val_years = list(range(2020, 2023))
    train_mask = df_merged['Season'].isin(train_years)
    val_mask = df_merged['Season'].isin(val_years)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {X_train_resampled.shape[0]} samples")
    
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    
    input_dim = len(feature_names)
    basic_model = BasicChampionPredictor(input_dim, hidden_dim=64)
    improved_model = ImprovedPredictor(input_dim)
    
    # Create DeepPredictor for Barttorvik data (if available)
    if barttorvik_features:
        deep_model = DeepPredictor(input_dim)
    
    champion_count = np.sum(y)
    non_champion_count = len(y) - champion_count
    class_weights = torch.tensor([1.0, champion_count / non_champion_count], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print("Training Basic PyTorch Model...")
    optimizer_basic = optim.Adam(basic_model.parameters(), lr=0.001, weight_decay=1e-5)
    basic_model, basic_train_losses, basic_val_losses = train_torch_model(
        basic_model, criterion, optimizer_basic, train_loader, val_loader, num_epochs=200, patience=30
    )
    
    print("Training Improved PyTorch Model...")
    optimizer_improved = optim.Adam(improved_model.parameters(), lr=0.001, weight_decay=1e-4)
    improved_model, improved_train_losses, improved_val_losses = train_torch_model(
        improved_model, criterion, optimizer_improved, train_loader, val_loader, num_epochs=200, patience=30
    )
    
    # Train DeepPredictor if Barttorvik features are available
    if barttorvik_features:
        print("Training Deep PyTorch Model with Barttorvik Features...")
        optimizer_deep = optim.Adam(deep_model.parameters(), lr=0.0005, weight_decay=1e-4)
        deep_model, deep_train_losses, deep_val_losses = train_torch_model(
            deep_model, criterion, optimizer_deep, train_loader, val_loader, num_epochs=200, patience=30
        )
    
    os.makedirs(models_dir, exist_ok=True)
    torch.save(basic_model.state_dict(), os.path.join(models_dir, "basic_model.pt"))
    torch.save(improved_model.state_dict(), os.path.join(models_dir, "improved_model.pt"))
    if barttorvik_features:
        torch.save(deep_model.state_dict(), os.path.join(models_dir, "deep_model.pt"))
    print("Saved PyTorch models.")
    
    import joblib
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print("Saved scaler and feature names.")
    
    basic_val_metrics = evaluate_model(basic_model, X_val_tensor, y_val_tensor)
    improved_val_metrics = evaluate_model(improved_model, X_val_tensor, y_val_tensor)
    
    print("Validation Metrics:")
    print(f"Basic Model AUC: {basic_val_metrics['auc']:.4f}, F1: {basic_val_metrics['f1_score']:.4f}")
    print(f"Improved Model AUC: {improved_val_metrics['auc']:.4f}, F1: {improved_val_metrics['f1_score']:.4f}")
    
    if barttorvik_features:
        deep_val_metrics = evaluate_model(deep_model, X_val_tensor, y_val_tensor)
        print(f"Deep Model AUC: {deep_val_metrics['auc']:.4f}, F1: {deep_val_metrics['f1_score']:.4f}")
    
    # Save feature importance if using a model that supports it (like XGBoost)
    print("Training XGBoost model for feature importance...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4,
        min_child_weight=2,
        scale_pos_weight=non_champion_count/champion_count,
        objective='binary:logistic',
        random_state=42
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.pkl"))
    
    # Plot feature importance
    plot_feature_importance(
        feature_names, 
        xgb_model.feature_importances_,
        "XGBoost Feature Importance",
        os.path.join(plots_dir, "feature_importance.png")
    )

# -----------------------------
# Prediction Function (Module Level)
# -----------------------------
def predict_bracket(predictions_file: str, bracket_file: str, output_file: str) -> None:
    """
    Run the bracket prediction process using the trained model predictions.
    
    Parameters:
    - predictions_file: Path to the champion probabilities CSV file
    - bracket_file: Path to the bracket structure file
    - output_file: Output file for the filled bracket
    """
    try:
        predictions_df = pd.read_csv(predictions_file)
        print(f"Loaded predictions for {len(predictions_df)} teams")
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        return

    try:
        regions = parse_bracket(bracket_file)
        print(f"Parsed bracket with {len(regions)} regions")
    except Exception as e:
        print(f"Error parsing bracket file: {e}")
        return

    results = simulate_bracket(regions, predictions_df)
    print_results(results)
    export_bracket(results, output_file)

# -----------------------------
# Main function: Train or Predict based on arguments
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='NCAA Tournament Bracket Predictor / Trainer')
    parser.add_argument('--predictions', default='archive/2025_champion_predictions.csv',
                        help='Path to the predictions CSV file (used in prediction mode)')
    parser.add_argument('--bracket', default='bracket_2025.txt',
                        help='Path to the bracket structure file (used in prediction mode)')
    parser.add_argument('--output', default='filled_bracket_2025.csv',
                        help='Output file for the filled bracket (used in prediction mode)')
    parser.add_argument('--train', action='store_true',
                        help='Train models instead of predicting bracket')
    args = parser.parse_args()
    if args.train:
        train_models()
    else:
        predict_bracket(args.predictions, args.bracket, args.output)

if __name__ == "__main__":
    main()