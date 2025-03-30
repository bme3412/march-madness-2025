import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Data Loading Functions
def load_team_data():
    """Load and merge team-level statistics from multiple sources"""
    # Update paths to include the data-2015 directory
    data_dir = "data-2015/"
    
    # Load all relevant data sources
    barttorvik_home = pd.read_csv(data_dir + "Barttorvik Home.csv")
    barttorvik_away = pd.read_csv(data_dir + "Barttorvik Away.csv")
    barttorvik_neutral = pd.read_csv(data_dir + "Barttorvik Neutral.csv")
    team_rankings = pd.read_csv(data_dir + "TeamRankings.csv")
    evanmiya = pd.read_csv(data_dir + "EvanMiya.csv")
    
    # Check if Shooting Splits file exists, if not create an empty dataframe
    try:
        shooting_splits = pd.read_csv(data_dir + "Shooting Splits.csv")
    except FileNotFoundError:
        print("Warning: Shooting Splits.csv not found. Creating empty dataframe.")
        shooting_splits = pd.DataFrame()
    
    # Check if KenPom Barttorvik file exists
    try:
        kenpom_barttorvik = pd.read_csv(data_dir + "KenPom Barttorvik.csv")
    except FileNotFoundError:
        print("Warning: KenPom Barttorvik.csv not found. Creating empty dataframe.")
        kenpom_barttorvik = pd.DataFrame()
    
    # Merge datasets on team number as the common identifier
    # Start with barttorvik as the base
    team_data = barttorvik_home.copy()
    
    # Add suffixes to differentiate between home/away/neutral metrics
    if 'BADJ EM' in barttorvik_away.columns:
        away_cols = {col: f"{col}_AWAY" for col in barttorvik_away.columns 
                    if col not in ['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM', 'SEED', 'ROUND']}
        barttorvik_away = barttorvik_away.rename(columns=away_cols)
        team_data = pd.merge(team_data, barttorvik_away, on=['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM', 'SEED', 'ROUND'], how='left')
    
    if 'BADJ EM' in barttorvik_neutral.columns:
        neutral_cols = {col: f"{col}_NEUTRAL" for col in barttorvik_neutral.columns 
                       if col not in ['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM', 'SEED', 'ROUND']}
        barttorvik_neutral = barttorvik_neutral.rename(columns=neutral_cols)
        team_data = pd.merge(team_data, barttorvik_neutral, on=['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM', 'SEED', 'ROUND'], how='left')
    
    # Add TeamRankings data
    team_data = pd.merge(team_data, team_rankings, on=['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'ROUND'], how='left')
    
    # Add EvanMiya metrics
    evanmiya_cols = ['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'ROUND', 
                     'O RATE', 'D RATE', 'RELATIVE RATING', 'OPPONENT ADJUST', 'PACE ADJUST',
                     'TRUE TEMPO', 'KILL SHOTS PER GAME', 'KILL SHOTS CONCEDED PER GAME']
    evanmiya_subset = evanmiya[evanmiya_cols] if all(col in evanmiya.columns for col in evanmiya_cols) else evanmiya
    team_data = pd.merge(team_data, evanmiya_subset, on=['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'ROUND'], how='left')
    
    # Add Shooting Splits data if available
    if not shooting_splits.empty:
        shooting_cols = ['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM',
                         'DUNKS FG%', 'DUNKS SHARE', 'DUNKS FG%D', 'DUNKS D SHARE',
                         'CLOSE TWOS FG%', 'CLOSE TWOS SHARE', 'CLOSE TWOS FG%D', 'CLOSE TWOS D SHARE',
                         'FARTHER TWOS FG%', 'FARTHER TWOS SHARE', 'FARTHER TWOS FG%D', 'FARTHER TWOS D SHARE',
                         'THREES FG%', 'THREES SHARE', 'THREES FG%D', 'THREES D SHARE']
        shooting_subset = shooting_splits[shooting_cols] if all(col in shooting_splits.columns for col in shooting_cols) else shooting_splits
        team_data = pd.merge(team_data, shooting_subset, on=['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM'], how='left')
    
    # Add KenPom metrics if available
    if not kenpom_barttorvik.empty and 'KADJ EM' not in team_data.columns and 'KADJ EM' in kenpom_barttorvik.columns:
        kenpom_cols = ['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM', 
                        'K TEMPO', 'KADJ T', 'K OFF', 'KADJ O', 'K DEF', 'KADJ D', 'KADJ EM']
        kenpom_subset = kenpom_barttorvik[kenpom_cols] if all(col in kenpom_barttorvik.columns for col in kenpom_cols) else kenpom_barttorvik
        team_data = pd.merge(team_data, kenpom_subset, on=['YEAR', 'TEAM NO', 'TEAM ID', 'TEAM'], how='left')
    
    return team_data

def load_tournament_data():
    """Load historical tournament data"""
    data_dir = "data-2015/"
    
    # Load raw matchups data
    try:
        matchups_raw = pd.read_csv(data_dir + "Tournament Matchups.csv")
        print("Tournament Matchups columns:", matchups_raw.columns.tolist())
        print("Sample rows:")
        print(matchups_raw.head(3))
    except FileNotFoundError:
        print("Warning: Tournament Matchups.csv not found.")
        matchups_raw = pd.DataFrame()
    
    # Load tournament locations data
    try:
        locations = pd.read_csv(data_dir + "Tournament Locations.csv")
    except FileNotFoundError:
        print("Warning: Tournament Locations.csv not found. Creating empty dataframe.")
        locations = pd.DataFrame()
    
    # Process the matchups to create paired data
    matchups = []
    
    if not matchups_raw.empty and 'BY_YEAR_NO' in matchups_raw.columns:
        # Sort by BY_ROUND_NO to ensure consistent ordering
        matchups_raw = matchups_raw.sort_values(['BY_YEAR_NO', 'BY_ROUND_NO'])
        
        # Group by BY_YEAR_NO to get matchup pairs
        for year_no, group in matchups_raw.groupby('BY_YEAR_NO'):
            if len(group) == 2:  # We expect exactly 2 teams per matchup
                team1 = group.iloc[0]
                team2 = group.iloc[1]
                
                # Create a matchup record
                matchup = {
                    'YEAR': team1['YEAR'],
                    'TEAM1_ID': team1['TEAM NO'],
                    'TEAM1_NAME': team1['TEAM'],
                    'TEAM1_SEED': team1['SEED'],
                    'TEAM2_ID': team2['TEAM NO'],
                    'TEAM2_NAME': team2['TEAM'],
                    'TEAM2_SEED': team2['SEED'],
                    'ROUND': team1['ROUND'],
                    'TEAM1_SCORE': team1['SCORE'] if not pd.isna(team1['SCORE']) else None,
                    'TEAM2_SCORE': team2['SCORE'] if not pd.isna(team2['SCORE']) else None,
                }
                
                # Determine winner if scores are available
                if matchup['TEAM1_SCORE'] is not None and matchup['TEAM2_SCORE'] is not None:
                    if matchup['TEAM1_SCORE'] > matchup['TEAM2_SCORE']:
                        matchup['WINNER_ID'] = matchup['TEAM1_ID']
                    else:
                        matchup['WINNER_ID'] = matchup['TEAM2_ID']
                else:
                    # If no scores, we can't determine the winner
                    matchup['WINNER_ID'] = None
                    
                matchups.append(matchup)
    else:
        # Check if columns exist with different naming conventions
        if not matchups_raw.empty:
            print("Using original format for Tournament Matchups")
            if 'BY YEAR NO' in matchups_raw.columns and 'BY ROUND NO' in matchups_raw.columns:
                # Sort by BY ROUND NO to ensure consistent ordering
                matchups_raw = matchups_raw.sort_values(['BY YEAR NO', 'BY ROUND NO'])
                
                # Group by BY YEAR NO to get matchup pairs
                for year_no, group in matchups_raw.groupby('BY YEAR NO'):
                    if len(group) == 2:  # We expect exactly 2 teams per matchup
                        team1 = group.iloc[0]
                        team2 = group.iloc[1]
                        
                        # Create a matchup record
                        matchup = {
                            'YEAR': team1['YEAR'],
                            'TEAM1_ID': team1['TEAM NO'],
                            'TEAM1_NAME': team1['TEAM'],
                            'TEAM1_SEED': team1['SEED'],
                            'TEAM2_ID': team2['TEAM NO'],
                            'TEAM2_NAME': team2['TEAM'],
                            'TEAM2_SEED': team2['SEED'],
                            'ROUND': team1['ROUND'],
                            'TEAM1_SCORE': team1['SCORE'] if not pd.isna(team1['SCORE']) else None,
                            'TEAM2_SCORE': team2['SCORE'] if not pd.isna(team2['SCORE']) else None,
                        }
                        
                        # Determine winner if scores are available
                        if matchup['TEAM1_SCORE'] is not None and matchup['TEAM2_SCORE'] is not None:
                            if matchup['TEAM1_SCORE'] > matchup['TEAM2_SCORE']:
                                matchup['WINNER_ID'] = matchup['TEAM1_ID']
                            else:
                                matchup['WINNER_ID'] = matchup['TEAM2_ID']
                        else:
                            # If no scores, we can't determine the winner
                            matchup['WINNER_ID'] = None
                            
                        matchups.append(matchup)
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups) if matchups else pd.DataFrame(columns=['YEAR', 'TEAM1_ID', 'TEAM2_ID', 'WINNER_ID'])
    return matchups_df, locations

def load_coach_data():
    """Load coach performance data"""
    data_dir = "data-2015/"
    try:
        coaches = pd.read_csv(data_dir + "Coach Results.csv")
    except FileNotFoundError:
        print("Warning: Coach Results.csv not found. Creating empty dataframe.")
        coaches = pd.DataFrame()
    return coaches

def load_conference_data():
    """Load conference data"""
    data_dir = "data-2015/"
    try:
        conf_results = pd.read_csv(data_dir + "Conference Results.csv")
    except FileNotFoundError:
        print("Warning: Conference Results.csv not found. Creating empty dataframe.")
        conf_results = pd.DataFrame()
    
    try:
        conf_stats = pd.read_csv(data_dir + "Conference Stats.csv")
    except FileNotFoundError:
        print("Warning: Conference Stats.csv not found. Creating empty dataframe.")
        conf_stats = pd.DataFrame()
    
    return conf_results, conf_stats

# 2. Feature Engineering
def engineer_matchup_features(team1_data, team2_data, coach_data, conf_data):
    """
    Create features for a matchup between two teams
    
    Parameters:
    - team1_data: DataFrame row for team 1
    - team2_data: DataFrame row for team 2
    - coach_data: Coach performance data
    - conf_data: Conference data
    
    Returns:
    - feature_vector: Array of engineered features for this matchup
    """
    features = []
    
    # Basic efficiency metrics - Barttorvik
    if all(col in team1_data and col in team2_data for col in ["BADJ EM", "BADJ O", "BADJ D", "BARTHAG"]):
        features.append(team1_data["BADJ EM"] - team2_data["BADJ EM"])
        features.append(team1_data["BADJ O"] - team2_data["BADJ O"])
        features.append(team1_data["BADJ D"] - team2_data["BADJ D"])
        features.append(team1_data["BARTHAG"] - team2_data["BARTHAG"])
    else:
        # Add placeholders if metrics not available
        features.extend([0, 0, 0, 0])
    
    # Add KenPom efficiency metrics if available
    if "KADJ EM" in team1_data and "KADJ EM" in team2_data:
        features.append(team1_data["KADJ EM"] - team2_data["KADJ EM"])
        features.append(team1_data["KADJ O"] - team2_data["KADJ O"])
        features.append(team1_data["KADJ D"] - team2_data["KADJ D"])
    else:
        features.extend([0, 0, 0])
    
    # Add EvanMiya ratings if available
    if "RELATIVE RATING" in team1_data and "RELATIVE RATING" in team2_data:
        features.append(team1_data["RELATIVE RATING"] - team2_data["RELATIVE RATING"])
        features.append(team1_data["O RATE"] - team2_data["O RATE"])
        features.append(team1_data["D RATE"] - team2_data["D RATE"])
        features.append(team1_data["OPPONENT ADJUST"] - team2_data["OPPONENT ADJUST"])
        features.append(team1_data["PACE ADJUST"] - team2_data["PACE ADJUST"])
        features.append(team1_data["KILL SHOTS PER GAME"] - team2_data["KILL SHOTS PER GAME"])
        features.append(team1_data["KILL SHOTS CONCEDED PER GAME"] - team2_data["KILL SHOTS CONCEDED PER GAME"])
    else:
        features.extend([0, 0, 0, 0, 0, 0, 0])
    
    # Add TeamRankings data if available
    if "TR RATING" in team1_data and "TR RATING" in team2_data:
        features.append(team1_data["TR RATING"] - team2_data["TR RATING"])
        features.append(team1_data["SOS RATING"] - team2_data["SOS RATING"])
        features.append(team1_data["LUCK RATING"] - team2_data["LUCK RATING"])
        features.append(team1_data["CONSISTENCY TR RATING"] - team2_data["CONSISTENCY TR RATING"])
    else:
        features.extend([0, 0, 0, 0])
    
    # Shooting metrics - Basic
    if all(col in team1_data and col in team2_data for col in ["EFG%", "EFG%D", "3PT%", "3PT%D"]):
        features.append(team1_data["EFG%"] - team2_data["EFG%"])
        features.append(team1_data["EFG%D"] - team2_data["EFG%D"])
        features.append(team1_data["3PT%"] - team2_data["3PT%"])
        features.append(team1_data["3PT%D"] - team2_data["3PT%D"])
    else:
        features.extend([0, 0, 0, 0])
    
    # Detailed shooting splits if available
    if "DUNKS FG%" in team1_data and "DUNKS FG%" in team2_data:
        # Interior scoring
        features.append(team1_data["DUNKS FG%"] - team2_data["DUNKS FG%"])
        features.append(team1_data["DUNKS SHARE"] - team2_data["DUNKS SHARE"])
        features.append(team1_data["CLOSE TWOS FG%"] - team2_data["CLOSE TWOS FG%"])
        features.append(team1_data["CLOSE TWOS SHARE"] - team2_data["CLOSE TWOS SHARE"])
        
        # Defense against inside shots
        features.append(team1_data["DUNKS FG%D"] - team2_data["DUNKS FG%D"])
        features.append(team1_data["DUNKS D SHARE"] - team2_data["DUNKS D SHARE"])
        features.append(team1_data["CLOSE TWOS FG%D"] - team2_data["CLOSE TWOS FG%D"])
        features.append(team1_data["CLOSE TWOS D SHARE"] - team2_data["CLOSE TWOS D SHARE"])
        
        # Mid/long range
        features.append(team1_data["FARTHER TWOS FG%"] - team2_data["FARTHER TWOS FG%"])
        features.append(team1_data["THREES FG%"] - team2_data["THREES FG%"])
        features.append(team1_data["THREES SHARE"] - team2_data["THREES SHARE"])
    else:
        features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Possession metrics
    if all(col in team1_data and col in team2_data for col in ["TOV%", "TOV%D", "OREB%", "DREB%", "FTR", "FTRD"]):
        features.append(team1_data["TOV%"] - team2_data["TOV%"])
        features.append(team1_data["TOV%D"] - team2_data["TOV%D"])
        features.append(team1_data["OREB%"] - team2_data["OREB%"])
        features.append(team1_data["DREB%"] - team2_data["DREB%"])
        features.append(team1_data["FTR"] - team2_data["FTR"])
        features.append(team1_data["FTRD"] - team2_data["FTRD"])
    else:
        features.extend([0, 0, 0, 0, 0, 0])
    
    # Tempo considerations
    if "TRUE TEMPO" in team1_data and "TRUE TEMPO" in team2_data:
        features.append(team1_data["TRUE TEMPO"] - team2_data["TRUE TEMPO"])
    elif "RAW T" in team1_data and "RAW T" in team2_data:
        features.append(team1_data["RAW T"] - team2_data["RAW T"])
    else:
        features.append(0)
    
    # Seeding information
    features.append(team2_data["SEED"] - team1_data["SEED"])  # Higher seed value = weaker team
    
    # Experience & talent
    if "EXP" in team1_data and "EXP" in team2_data:
        features.append(team1_data["EXP"] - team2_data["EXP"])
    else:
        features.append(0)
        
    if "TALENT" in team1_data and "TALENT" in team2_data:
        features.append(team1_data["TALENT"] - team2_data["TALENT"])
    else:
        features.append(0)
    
    # Location-specific performance (neutral court for tournament games)
    if "BADJ EM_NEUTRAL" in team1_data and "BADJ EM_NEUTRAL" in team2_data:
        features.append(team1_data["BADJ EM_NEUTRAL"] - team2_data["BADJ EM_NEUTRAL"])
        features.append(team1_data["BADJ O_NEUTRAL"] - team2_data["BADJ O_NEUTRAL"])
        features.append(team1_data["BADJ D_NEUTRAL"] - team2_data["BADJ D_NEUTRAL"])
    else:
        features.extend([0, 0, 0])
    
    # Coach performance
    if coach_data is not None and not coach_data.empty and "COACH ID" in team1_data and "COACH ID" in team2_data:
        try:
            team1_coach = coach_data[coach_data["COACH ID"] == team1_data["COACH ID"]].iloc[0]
            team2_coach = coach_data[coach_data["COACH ID"] == team2_data["COACH ID"]].iloc[0]
            
            features.append(team1_coach["PASE"] - team2_coach["PASE"])
            features.append(team1_coach["PAKE"] - team2_coach["PAKE"])
            features.append(team1_coach["F4%"] - team2_coach["F4%"])
            features.append(team1_coach["CHAMP%"] - team2_coach["CHAMP%"])
        except (KeyError, IndexError):
            # If coach data isn't available, add zeros as placeholders
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])
    
    # Conference strength
    if conf_data is not None and not conf_data.empty and "CONF ID" in team1_data and "CONF ID" in team2_data:
        try:
            team1_conf = conf_data[conf_data["CONF ID"] == team1_data["CONF ID"]].iloc[0]
            team2_conf = conf_data[conf_data["CONF ID"] == team2_data["CONF ID"]].iloc[0]
            
            features.append(team1_conf["PASE"] - team2_conf["PASE"])
            features.append(team1_conf["PAKE"] - team2_conf["PAKE"])
            features.append(team1_conf["CHAMP%"] - team2_conf["CHAMP%"])
        except (KeyError, IndexError):
            # If conference data isn't available, add zeros as placeholders
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0])
    
    # Convert any NaN values to 0
    features = np.nan_to_num(features)
    
    return np.array(features)

# 3. Dataset Class
class MarchMadnessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Neural Network Models
class MarchMadnessPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.4):
        super(MarchMadnessPredictor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)


class EnsembleMarchMadnessPredictor(nn.Module):
    def __init__(self, input_size, num_models=3, hidden_size=96, dropout_rate=0.35):
        super(EnsembleMarchMadnessPredictor, self).__init__()
        
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ) for _ in range(num_models)
        ])
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# 5. Training Functions
def train_model(train_loader, val_loader, input_size, epochs=100, lr=0.001, model_type="standard", ensemble_size=3):
    """
    Train a neural network model for March Madness prediction
    
    Parameters:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - input_size: Number of input features
    - epochs: Number of training epochs
    - lr: Learning rate
    - model_type: "standard" or "ensemble"
    - ensemble_size: Number of models in ensemble if model_type is "ensemble"
    
    Returns:
    - Trained model
    """
    if model_type == "ensemble":
        model = EnsembleMarchMadnessPredictor(input_size, num_models=ensemble_size)
    else:
        model = MarchMadnessPredictor(input_size)
        
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 20  # For early stopping
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Track predictions and targets for metrics
                predicted = (outputs > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Calculate basic accuracy
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        # For bracket competitions, consider log loss (similar to Kaggle March Madness scoring)
        # But clamp predictions to avoid extreme penalties
        outputs_np = np.clip(np.array(all_preds), 0.05, 0.95)
        targets_np = np.array(all_targets)
        
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, AUC: {auc:.4f}')
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(best_model)
    return model


def train_with_cross_validation(X, y, folds=5, model_type="standard", ensemble_size=3):
    """
    Train with k-fold cross validation to ensure robust performance
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - folds: Number of CV folds
    - model_type: "standard" or "ensemble"
    - ensemble_size: Number of models in ensemble if model_type is "ensemble"
    
    Returns:
    - List of trained models (one per fold)
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    models = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Training fold {fold+1}/{folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = MarchMadnessDataset(X_train, y_train)
        val_dataset = MarchMadnessDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train model
        input_size = X_train.shape[1]
        model = train_model(train_loader, val_loader, input_size, epochs=100, 
                            model_type=model_type, ensemble_size=ensemble_size)
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                outputs = model(inputs).squeeze()
                val_preds.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        val_preds = np.array(val_preds)
        val_pred_classes = (val_preds > 0.5).astype(int)
        accuracy = np.mean(val_pred_classes == y_val)
        
        from sklearn.metrics import roc_auc_score, log_loss
        try:
            auc = roc_auc_score(y_val, val_preds)
            log_loss_val = log_loss(y_val, np.clip(val_preds, 0.05, 0.95))
        except:
            auc = 0.5
            log_loss_val = 1.0
        
        print(f"Fold {fold+1} - Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Log Loss: {log_loss_val:.4f}")
        
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'auc': auc,
            'log_loss': log_loss_val
        })
        
        models.append(model)
    
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_auc = np.mean([m['auc'] for m in fold_metrics])
    avg_log_loss = np.mean([m['log_loss'] for m in fold_metrics])
    
    print(f"\nCross-validation results:")
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"Average Log Loss: {avg_log_loss:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    
    return models

# 6. Simulation Functions
def simulate_tournament(models, teams_data, bracket, coach_data=None, conf_data=None, scaler=None):
    """
    Simulate the entire tournament using the trained model ensemble
    
    Parameters:
    - models: List of trained neural network models
    - teams_data: DataFrame with team data
    - bracket: Initial tournament bracket structure
    - coach_data: Coach performance data
    - conf_data: Conference performance data
    - scaler: Feature scaler used during training
    
    Returns:
    - completed_bracket: Bracket with predicted winners
    - win_probabilities: Dictionary of win probabilities for each matchup
    """
    for model in models:
        model.eval()
    
    completed_bracket = {i: [] for i in range(7)}  # Track winners for each round (0-6)
    completed_bracket[0] = bracket.copy()  # Initialize with first round teams
    win_probabilities = {}  # Track win probabilities for analysis
    
    for round_num in range(6):  # 6 rounds in March Madness
        next_round = []
        
        # Get matchups for this round
        if round_num == 0:
            matchups = [(bracket[i], bracket[i+1]) for i in range(0, len(bracket), 2)]
        else:
            winners = completed_bracket[round_num]
            matchups = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
        
        for team1_id, team2_id in matchups:
            try:
                team1_data = teams_data[teams_data["TEAM ID"] == team1_id].iloc[0]
                team2_data = teams_data[teams_data["TEAM ID"] == team2_id].iloc[0]
                
                features = engineer_matchup_features(team1_data, team2_data, coach_data, conf_data)
                
                if scaler is not None:
                    features = scaler.transform(features.reshape(1, -1)).flatten()
                
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                
                all_probs = []
                for model in models:
                    with torch.no_grad():
                        prob = model(features_tensor).item()
                        all_probs.append(prob)
                
                avg_prob = sum(all_probs) / len(all_probs)
                winner_id = team1_id if avg_prob > 0.5 else team2_id
                
                matchup_key = f"{team1_data['TEAM']} vs {team2_data['TEAM']} (Round {round_num+1})"
                win_probabilities[matchup_key] = {
                    'team1': team1_data['TEAM'],
                    'team2': team2_data['TEAM'],
                    'team1_prob': avg_prob,
                    'team2_prob': 1 - avg_prob,
                    'predicted_winner': team1_data['TEAM'] if avg_prob > 0.5 else team2_data['TEAM'],
                    'round': round_num + 1
                }
            except IndexError:
                print(f"Warning: Data not found for Team ID {team1_id} or {team2_id}, using random winner")
                # If team data is missing, choose a random winner
                winner_id = team1_id if np.random.random() > 0.5 else team2_id
            
            next_round.append(winner_id)
        
        completed_bracket[round_num + 1] = next_round
    
    return completed_bracket, win_probabilities

def generate_multiple_brackets(models, teams_data, initial_bracket, num_brackets=10, coach_data=None, conf_data=None, scaler=None):
    """
    Generate multiple tournament brackets to account for uncertainty
    
    Parameters:
    - models: List of trained neural network models
    - teams_data: DataFrame with team data
    - initial_bracket: Initial tournament bracket structure
    - num_brackets: Number of brackets to generate
    - coach_data: Coach performance data
    - conf_data: Conference performance data
    - scaler: Feature scaler used during training
    
    Returns:
    - brackets: List of simulated brackets
    - team_advancement_probs: Dictionary with probabilities of each team advancing to each round
    """
    brackets = []
    
    team_advancement = {}
    for _, team_row in teams_data.iterrows():
        team_id = team_row["TEAM ID"]
        team_name = team_row["TEAM"]
        team_advancement[team_id] = {
            'team_name': team_name,
            'round_counts': {i: 0 for i in range(7)}  # Rounds 0-6
        }
        team_advancement[team_id]['round_counts'][0] = 1
    
    for i in range(num_brackets):
        completed_bracket = {j: [] for j in range(7)}
        completed_bracket[0] = initial_bracket.copy()
        
        for round_num in range(6):
            next_round = []
            
            if round_num == 0:
                matchups = [(initial_bracket[j], initial_bracket[j+1]) for j in range(0, len(initial_bracket), 2)]
            else:
                winners = completed_bracket[round_num]
                matchups = [(winners[j], winners[j+1]) for j in range(0, len(winners), 2)]
            
            for team1_id, team2_id in matchups:
                try:
                    team1_data = teams_data[teams_data["TEAM ID"] == team1_id].iloc[0]
                    team2_data = teams_data[teams_data["TEAM ID"] == team2_id].iloc[0]
                    
                    features = engineer_matchup_features(team1_data, team2_data, coach_data, conf_data)
                    
                    if scaler is not None:
                        features = scaler.transform(features.reshape(1, -1)).flatten()
                    
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    
                    all_probs = []
                    for model in models:
                        with torch.no_grad():
                            prob = model(features_tensor).item()
                            all_probs.append(prob)
                    
                    avg_prob = sum(all_probs) / len(all_probs)
                    team1_wins = np.random.random() < avg_prob
                    winner_id = team1_id if team1_wins else team2_id
                except IndexError:
                    # If team data is missing, choose a random winner
                    winner_id = team1_id if np.random.random() > 0.5 else team2_id
                
                next_round.append(winner_id)
                team_advancement[winner_id]['round_counts'][round_num + 1] += 1
            
            completed_bracket[round_num + 1] = next_round
        
        brackets.append(completed_bracket)
    
    team_advancement_probs = {}
    for team_id, data in team_advancement.items():
        team_name = data['team_name']
        round_probs = {}
        for round_num in range(7):
            count = data['round_counts'][round_num]
            prob = 1.0 if round_num == 0 else count / num_brackets
            round_probs[round_num] = prob
        team_advancement_probs[team_name] = round_probs
    
    return brackets, team_advancement_probs

# 7. Feature Importance Analysis
def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance using permutation method
    
    Parameters:
    - model: Trained neural network model
    - feature_names: List of feature names
    
    Returns:
    - importance_df: DataFrame with feature importances
    """
    try:
        with torch.no_grad():
            weights = model.model[0].weight.data.cpu().numpy()
        
        abs_weights = np.abs(weights)
        avg_importance = np.mean(abs_weights, axis=0)
        
        # Make sure feature_names is the same length as avg_importance
        if len(feature_names) != len(avg_importance):
            print(f"Warning: Feature names length ({len(feature_names)}) does not match model weights ({len(avg_importance)})")
            # Pad or truncate feature names to match
            if len(feature_names) < len(avg_importance):
                feature_names.extend([f"Feature_{i}" for i in range(len(feature_names), len(avg_importance))])
            else:
                feature_names = feature_names[:len(avg_importance)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_importance
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
        # Return empty dataframe
        return pd.DataFrame(columns=['Feature', 'Importance'])

# 8. Main Pipeline
def main():
    print("Loading data...")
    team_data = load_team_data()
    matchups, locations = load_tournament_data()
    coach_data = load_coach_data()
    conf_results, conf_stats = load_conference_data()
    
    print("Processing historical matchups...")
    X = []  # Features
    y = []  # Outcomes (1 if team1 won, 0 if team2 won)
    feature_names = []  # For feature importance analysis
    
    # Define feature names based on available columns
    if not team_data.empty:
        # Sample team data to determine available columns
        sample_team = team_data.iloc[0]
        
        if 'BADJ EM' in sample_team:
            feature_names.extend(["BADJ EM Diff", "BADJ O Diff", "BADJ D Diff", "BARTHAG Diff"])
        else:
            feature_names.extend(["Placeholder1", "Placeholder2", "Placeholder3", "Placeholder4"])
            
        if 'KADJ EM' in sample_team:
            feature_names.extend(["KADJ EM Diff", "KADJ O Diff", "KADJ D Diff"])
        else:
            feature_names.extend(["Placeholder5", "Placeholder6", "Placeholder7"])
            
        if 'RELATIVE RATING' in sample_team:
            feature_names.extend(["RELATIVE RATING Diff", "O RATE Diff", "D RATE Diff",
                                "OPPONENT ADJUST Diff", "PACE ADJUST Diff", 
                                "KILL SHOTS PER GAME Diff", "KILL SHOTS CONCEDED PER GAME Diff"])
        else:
            feature_names.extend(["Placeholder8", "Placeholder9", "Placeholder10",
                                "Placeholder11", "Placeholder12", "Placeholder13", "Placeholder14"])
            
        if 'TR RATING' in sample_team:
            feature_names.extend(["TR RATING Diff", "SOS RATING Diff", "LUCK RATING Diff", "CONSISTENCY TR RATING Diff"])
        else:
            feature_names.extend(["Placeholder15", "Placeholder16", "Placeholder17", "Placeholder18"])
        
        feature_names.extend(["EFG% Diff", "EFG%D Diff", "3PT% Diff", "3PT%D Diff"])
        
        if 'DUNKS FG%' in sample_team:
            feature_names.extend([
                "DUNKS FG% Diff", "DUNKS SHARE Diff", 
                "CLOSE TWOS FG% Diff", "CLOSE TWOS SHARE Diff",
                "DUNKS FG%D Diff", "DUNKS D SHARE Diff",
                "CLOSE TWOS FG%D Diff", "CLOSE TWOS D SHARE Diff",
                "FARTHER TWOS FG% Diff", "THREES FG% Diff", "THREES SHARE Diff"
            ])
        else:
            feature_names.extend(["Placeholder19", "Placeholder20", "Placeholder21", 
                                "Placeholder22", "Placeholder23", "Placeholder24",
                                "Placeholder25", "Placeholder26", "Placeholder27", 
                                "Placeholder28", "Placeholder29"])
        
        feature_names.extend([
            "TOV% Diff", "TOV%D Diff", "OREB% Diff", "DREB% Diff", 
            "FTR Diff", "FTRD Diff"
        ])
        
        if "TRUE TEMPO" in sample_team:
            feature_names.append("TRUE TEMPO Diff")
        else:
            feature_names.append("RAW T Diff")
        
        feature_names.extend(["SEED Diff", "EXP Diff", "TALENT Diff"])
        
        if "BADJ EM_NEUTRAL" in sample_team:
            feature_names.extend(["BADJ EM_NEUTRAL Diff", "BADJ O_NEUTRAL Diff", "BADJ D_NEUTRAL Diff"])
        else:
            feature_names.extend(["Placeholder30", "Placeholder31", "Placeholder32"])
        
        feature_names.extend(["Coach PASE Diff", "Coach PAKE Diff", "Coach F4% Diff", "Coach CHAMP% Diff"])
        feature_names.extend(["Conf PASE Diff", "Conf PAKE Diff", "Conf CHAMP% Diff"])
    
    # Process historical matchups if available
    if not matchups.empty and 'WINNER_ID' in matchups.columns:
        for _, matchup in matchups.iterrows():
            if pd.isna(matchup["WINNER_ID"]):
                continue  # Skip matchups without winners
                
            team1_id = matchup["TEAM1_ID"]
            team2_id = matchup["TEAM2_ID"]
            winner_id = matchup["WINNER_ID"]
            
            # Get team data rows
            try:
                team1_data = team_data[team_data["TEAM ID"] == team1_id].iloc[0]
                team2_data = team_data[team_data["TEAM ID"] == team2_id].iloc[0]
                
                # Engineer features
                features = engineer_matchup_features(team1_data, team2_data, coach_data, conf_results)
                
                # Add to dataset
                X.append(features)
                y.append(1 if winner_id == team1_id else 0)
            except (IndexError, KeyError) as e:
                # Skip if team data not found
                print(f"Warning: Data not found for Team {team1_id} or Team {team2_id}: {e}")
                continue
    
    # Check if we have enough data to train
    if len(X) > 0 and len(y) > 0:
        X = np.array(X)
        y = np.array(y)
        
        print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Training models with cross-validation...")
        models = train_with_cross_validation(X_scaled, y, folds=5, model_type="ensemble", ensemble_size=3)
        
        for i, model in enumerate(models):
            torch.save(model.state_dict(), f"march_madness_predictor_fold_{i+1}.pth")
        
        print("Analyzing feature importance...")
        importance_df = analyze_feature_importance(models[0], feature_names)
        print("Top 10 most important features:")
        print(importance_df.head(10))
    else:
        print("Not enough historical data for training. Creating a simple model based on seed differentials.")
        # Create a simple model without training
        input_size = len(feature_names)  # Use the same input size
        models = [MarchMadnessPredictor(input_size)]
    
    print("\nPreparing 2025 tournament simulation...")
    current_teams = team_data[team_data["YEAR"] == 2025]
    
    # Create the initial bracket from tournament matchups
    initial_bracket = []
    
    # Extract teams from matchups or use team IDs directly
    if not matchups.empty and 'ROUND' in matchups.columns:
        # Filter first round matchups
        first_round = matchups[matchups['ROUND'] == 1]
        for _, row in first_round.iterrows():
            initial_bracket.append(row['TEAM1_ID'])
            initial_bracket.append(row['TEAM2_ID'])
    
    # If we couldn't extract from matchups, use team data
    if not initial_bracket:
        team_ids = current_teams["TEAM ID"].tolist()
        if len(team_ids) >= 64:
            initial_bracket = team_ids[:64]
        else:
            # If we have fewer than 64 teams, repeat to fill bracket
            initial_bracket = []
            while len(initial_bracket) < 64:
                initial_bracket.extend(team_ids[:min(len(team_ids), 64-len(initial_bracket))])
    
    # Ensure bracket has correct number of teams
    if len(initial_bracket) > 64:
        initial_bracket = initial_bracket[:64]
    
    print(f"Initial bracket created with {len(initial_bracket)} teams")
    
    print("Simulating 2025 tournament...")
    completed_bracket, win_probabilities = simulate_tournament(
        models, current_teams, initial_bracket, coach_data, conf_results, scaler if 'scaler' in locals() else None
    )
    
    print("Generating multiple bracket scenarios...")
    brackets, team_advancement_probs = generate_multiple_brackets(
        models, current_teams, initial_bracket, num_brackets=100, 
        coach_data=coach_data, conf_data=conf_results, scaler=scaler if 'scaler' in locals() else None
    )
    
    print("\nChampionship Probabilities:")
    champion_round = 6  # Championship round
    champion_probs = {team: probs[champion_round] for team, probs in team_advancement_probs.items() 
                     if probs[champion_round] > 0}
    
    for team, prob in sorted(champion_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{team}: {prob:.1%}")
    
    print("\nFinal Four Probabilities:")
    final_four_round = 5  # Final Four round
    ff_probs = {team: probs[final_four_round] for team, probs in team_advancement_probs.items() 
               if probs[final_four_round] > 0}
    
    for team, prob in sorted(ff_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{team}: {prob:.1%}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Team': list(champion_probs.keys()),
        'Championship_Probability': list(champion_probs.values()),
        'Final_Four_Probability': [ff_probs.get(team, 0) for team in champion_probs.keys()]
    })
    results_df.sort_values('Championship_Probability', ascending=False, inplace=True)
    results_df.to_csv("2025_tournament_predictions.csv", index=False)
    print("\nResults saved to 2025_tournament_predictions.csv")
    
    print("\nTraining and simulation complete!")
    if len(X) > 0 and len(y) > 0:
        print("Models saved as march_madness_predictor_fold_*.pth")

if __name__ == "__main__":
    main()