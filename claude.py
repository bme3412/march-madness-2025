import os
import glob
import re
from typing import List, Tuple, Dict, Any

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
        A deeper neural network model with LeakyReLU activations.
        """
        super(DeepPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -----------------------------
# Data Loading and Preprocessing Functions
# -----------------------------
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features based on available statistics.
    """
    print("Engineering additional features...")

    # Basic features
    if 'Points_For' in df.columns and 'Points_Against' in df.columns:
        df['Point_Differential'] = df['Points_For'] - df['Points_Against']
    if 'Point_Differential' in df.columns and 'Games' in df.columns:
        df['Win_Margin'] = df['Point_Differential'] / df['Games']
    if 'SRS' in df.columns and 'SOS' in df.columns:
        df['SRS_SOS_Product'] = df['SRS'] * df['SOS']
    
    # Advanced features
    if 'FG' in df.columns and 'FGA' in df.columns:
        df['FG_Pct'] = df['FG'] / df['FGA']
    if 'FT' in df.columns and 'FTA' in df.columns:
        df['FT_Pct'] = df['FT'] / df['FTA']
    if '3P' in df.columns and '3PA' in df.columns:
        df['3P_Pct'] = df['3P'] / df['3PA']
    if 'FG' in df.columns and 'FGA' in df.columns and '3P' in df.columns:
        df['EFG_Pct'] = (df['FG'] + 0.5 * df['3P']) / df['FGA']
    if 'TOV' in df.columns and 'FGA' in df.columns and 'FTA' in df.columns:
        df['TO_Rate'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    if 'ORB' in df.columns and 'TRB' in df.columns:
        df['ORB_Pct'] = df['ORB'] / df['TRB']
    
    # Tournament experience features
    if 'NCAA_Tournament' in df.columns:
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
    
    return df


def preprocess_data(df: pd.DataFrame, seasons_summary: pd.DataFrame,
                    base_features: List[str],
                    engineered_features: List[str] = []) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Merge season summary with school stats and create the binary target variable.
    """
    print("Preprocessing data...")
    df = pd.merge(df, seasons_summary, left_on='Season', right_on='Year', how='left')
    
    def is_champion(row: pd.Series) -> int:
        if pd.isna(row['Champion']):
            return 0
        champ_name = row['Champion'].lower()
        school_name = row['School'].lower()
        if champ_name == school_name or champ_name in school_name or school_name in champ_name:
            return 1
        
        abbreviations = {
            'uconn': ['connecticut'],
            'unc': ['north carolina'],
            'ucla': ['california, los angeles'],
            'smu': ['southern methodist'],
            'lsu': ['louisiana state'],
            'vcu': ['virginia commonwealth'],
            'ucf': ['central florida']
        }
        for abbr, variations in abbreviations.items():
            if (abbr in champ_name and any(var in school_name for var in variations)) or \
               (abbr in school_name and any(var in champ_name for var in variations)):
                return 1
        return 0

    df['Label'] = df.apply(is_champion, axis=1)
    all_features = base_features.copy()
    for feat in engineered_features:
        if feat in df.columns:
            all_features.append(feat)
    
    df = df.dropna(subset=all_features)
    X = df[all_features].values.astype(np.float32)
    y = df['Label'].values.astype(np.int64)
    return X, y, df, all_features


# -----------------------------
# Model Training and Evaluation Functions
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
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        training_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                loss = criterion(model(batch_X), batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        validation_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_losses, validation_losses


def evaluate_model(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model using various metrics.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].numpy()
        predictions = (probabilities >= threshold).astype(int)
    y_true = y_tensor.numpy()

    precision = precision_score(y_true, predictions) if np.sum(predictions) and np.sum(y_true) else 0
    recall = recall_score(y_true, predictions) if np.sum(predictions) and np.sum(y_true) else 0
    f1 = f1_score(y_true, predictions) if np.sum(predictions) and np.sum(y_true) else 0
    auc = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else float('nan')
    cm = confusion_matrix(y_true, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'probabilities': probabilities,
        'predictions': predictions
    }


def train_sklearn_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train and return a scikit-learn model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_sklearn_model(model: Any, X: np.ndarray, y: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate a scikit-learn model using various metrics.
    """
    try:
        probabilities = model.predict_proba(X)[:, 1]
    except Exception:
        probabilities = model.predict(X)
    predictions = (probabilities >= threshold).astype(int)
    precision = precision_score(y, predictions) if np.sum(predictions) and np.sum(y) else 0
    recall = recall_score(y, predictions) if np.sum(predictions) and np.sum(y) else 0
    f1 = f1_score(y, predictions) if np.sum(predictions) and np.sum(y) else 0
    auc = roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else float('nan')
    cm = confusion_matrix(y, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'probabilities': probabilities,
        'predictions': predictions
    }


def create_ensemble_prediction(models: List[Tuple[Any, str]], scaler: StandardScaler,
                               X_raw: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Create ensemble predictions by averaging or weighting individual model outputs.
    """
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    all_probs = []

    for model, model_type in models:
        if model_type == 'torch':
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
        else:  # sklearn
            try:
                probs = model.predict_proba(X_scaled)[:, 1]
            except Exception:
                probs = model.predict(X_scaled)
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    if weights is not None:
        weights = np.array(weights).reshape(-1, 1)
        weighted_probs = all_probs * weights
        ensemble_probs = weighted_probs.sum(axis=0) / weights.sum()
    else:
        ensemble_probs = all_probs.mean(axis=0)
    
    return ensemble_probs


def plot_feature_importance(model: Any, feature_names: List[str], title: str, output_path: str) -> None:
    """
    Plot and save feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def plot_training_history(training_losses: List[float], validation_losses: List[float],
                          title: str, output_path: str) -> None:
    """
    Plot and save training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: str) -> None:
    """
    Plot and save a confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -----------------------------
# Main function
# -----------------------------
def main() -> None:
    # Define directories
    output_dir = 'output'
    plots_dir = os.path.join(output_dir, 'plots')
    models_dir = os.path.join(output_dir, 'models')
    archive_dir = 'archive'
    for directory in [output_dir, plots_dir, models_dir, archive_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Set file paths and year ranges
    data_seasons_dir = os.path.join('data-seasons')
    seasons_csv_path = os.path.join(data_seasons_dir, 'cbb_seasons.csv')
    start_year, end_year = 2000, 2024

    # Load and preview season summary
    seasons_summary = load_seasons_summary(seasons_csv_path)
    print(f"Seasons summary loaded. Total seasons: {len(seasons_summary)}")
    print(seasons_summary.head())
    
    # Load and process school stats
    df_stats = load_school_stats(data_seasons_dir, start_year, end_year)
    print(f"Loaded school stats data with {len(df_stats)} rows from {start_year} to {end_year}.")
    df_stats = engineer_features(df_stats)
    
    # Define features to be used
    base_features = ['Win_Pct', 'SRS', 'SOS', 'Points_For', 'Points_Against']
    engineered_features = [
        'Point_Differential', 'Win_Margin', 'SRS_SOS_Product',
        'FG_Pct', 'FT_Pct', '3P_Pct', 'EFG_Pct', 'TO_Rate', 'ORB_Pct',
        'Previous_Year_Tournament', 'Tournament_Streak'
    ]
    
    X, y, df_merged, feature_names = preprocess_data(df_stats, seasons_summary, base_features, engineered_features)
    print(f"After preprocessing, feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names}")
    
    # Print class balance
    champion_count = np.sum(y)
    non_champion_count = len(y) - champion_count
    print(f"Class balance: Champions: {champion_count}, Non-champions: {non_champion_count}")
    
    # Split data by season
    train_years = list(range(2000, 2020))
    val_years = list(range(2020, 2023))
    test_years = list(range(2023, 2025))
    train_mask = df_merged['Season'].isin(train_years)
    val_mask = df_merged['Season'].isin(val_years)
    test_mask = df_merged['Season'].isin(test_years)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Training set: {X_train.shape[0]} samples from years {train_years}")
    print(f"Validation set: {X_val.shape[0]} samples from years {val_years}")
    print(f"Test set: {X_test.shape[0]} samples from years {test_years}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle imbalance with SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE - Training samples: {X_train_resampled.shape[0]}, Champions: {np.sum(y_train_resampled)}")
    
    # Convert to PyTorch tensors and create DataLoaders
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)
    
    # Initialize models
    input_dim = len(feature_names)
    basic_model = BasicChampionPredictor(input_dim, hidden_dim=64)
    improved_model = ImprovedPredictor(input_dim)
    deep_model = DeepPredictor(input_dim)
    
    # Define loss function with class weighting for PyTorch models
    class_weights = torch.tensor([1.0, champion_count / non_champion_count], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train PyTorch models
    print("\nTraining Basic PyTorch Model...")
    optimizer_basic = optim.Adam(basic_model.parameters(), lr=0.001, weight_decay=1e-5)
    basic_model, basic_train_losses, basic_val_losses = train_torch_model(
        basic_model, criterion, optimizer_basic, train_loader, val_loader, num_epochs=200, patience=30
    )
    
    print("\nTraining Improved PyTorch Model...")
    optimizer_improved = optim.Adam(improved_model.parameters(), lr=0.001, weight_decay=1e-4)
    improved_model, improved_train_losses, improved_val_losses = train_torch_model(
        improved_model, criterion, optimizer_improved, train_loader, val_loader, num_epochs=200, patience=30
    )
    
    print("\nTraining Deep PyTorch Model...")
    optimizer_deep = optim.Adam(deep_model.parameters(), lr=0.0005, weight_decay=1e-4)
    deep_model, deep_train_losses, deep_val_losses = train_torch_model(
        deep_model, criterion, optimizer_deep, train_loader, val_loader, num_epochs=300, patience=40
    )
    
    # Train scikit-learn models on original (non-resampled) data
    print("\nTraining Random Forest Model...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42
    )
    rf_model = train_sklearn_model(rf_model, X_train_scaled, y_train)
    
    print("\nTraining Gradient Boosting Model...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, min_samples_split=5,
        min_samples_leaf=2, subsample=0.8, random_state=42
    )
    gb_model = train_sklearn_model(gb_model, X_train_scaled, y_train)
    
    print("\nTraining XGBoost Model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, min_child_weight=2,
        gamma=0.1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=non_champion_count / champion_count, random_state=42
    )
    xgb_model = train_sklearn_model(xgb_model, X_train_scaled, y_train)
    
    # Evaluate all models on validation data
    print("\nEvaluating models on validation data...")
    basic_val_metrics = evaluate_model(basic_model, X_val_tensor, y_val_tensor)
    improved_val_metrics = evaluate_model(improved_model, X_val_tensor, y_val_tensor)
    deep_val_metrics = evaluate_model(deep_model, X_val_tensor, y_val_tensor)
    rf_val_metrics = evaluate_sklearn_model(rf_model, X_val_scaled, y_val)
    gb_val_metrics = evaluate_sklearn_model(gb_model, X_val_scaled, y_val)
    xgb_val_metrics = evaluate_sklearn_model(xgb_model, X_val_scaled, y_val)
    
    print("\nValidation Metrics:")
    print(f"Basic PyTorch Model - AUC: {basic_val_metrics['auc']:.4f}, F1: {basic_val_metrics['f1_score']:.4f}")
    print(f"Improved PyTorch Model - AUC: {improved_val_metrics['auc']:.4f}, F1: {improved_val_metrics['f1_score']:.4f}")
    print(f"Deep PyTorch Model - AUC: {deep_val_metrics['auc']:.4f}, F1: {deep_val_metrics['f1_score']:.4f}")
    print(f"Random Forest - AUC: {rf_val_metrics['auc']:.4f}, F1: {rf_val_metrics['f1_score']:.4f}")
    print(f"Gradient Boosting - AUC: {gb_val_metrics['auc']:.4f}, F1: {gb_val_metrics['f1_score']:.4f}")
    print(f"XGBoost - AUC: {xgb_val_metrics['auc']:.4f}, F1: {xgb_val_metrics['f1_score']:.4f}")
    
    # Plot training history and feature importance
    plot_training_history(basic_train_losses, basic_val_losses, "Basic Model Training History",
                            os.path.join(plots_dir, "basic_model_history.png"))
    plot_training_history(improved_train_losses, improved_val_losses, "Improved Model Training History",
                            os.path.join(plots_dir, "improved_model_history.png"))
    plot_training_history(deep_train_losses, deep_val_losses, "Deep Model Training History",
                            os.path.join(plots_dir, "deep_model_history.png"))
    plot_feature_importance(rf_model, feature_names, "Random Forest Feature Importance",
                            os.path.join(plots_dir, "rf_feature_importance.png"))
    plot_feature_importance(gb_model, feature_names, "Gradient Boosting Feature Importance",
                            os.path.join(plots_dir, "gb_feature_importance.png"))
    plot_feature_importance(xgb_model, feature_names, "XGBoost Feature Importance",
                            os.path.join(plots_dir, "xgb_feature_importance.png"))
    plot_confusion_matrix(basic_val_metrics['confusion_matrix'], "Basic Model Confusion Matrix",
                            os.path.join(plots_dir, "basic_confusion_matrix.png"))
    plot_confusion_matrix(improved_val_metrics['confusion_matrix'], "Improved Model Confusion Matrix",
                            os.path.join(plots_dir, "improved_confusion_matrix.png"))
    plot_confusion_matrix(rf_val_metrics['confusion_matrix'], "Random Forest Confusion Matrix",
                            os.path.join(plots_dir, "rf_confusion_matrix.png"))
    
    # Evaluate on test set
    print("\nEvaluating best models on test data (2023-2024)...")
    val_metrics = [
        (basic_val_metrics['auc'], 'Basic PyTorch'),
        (improved_val_metrics['auc'], 'Improved PyTorch'),
        (deep_val_metrics['auc'], 'Deep PyTorch'),
        (rf_val_metrics['auc'], 'Random Forest'),
        (gb_val_metrics['auc'], 'Gradient Boosting'),
        (xgb_val_metrics['auc'], 'XGBoost')
    ]
    best_auc, best_model_name = max(val_metrics, key=lambda x: x[0])
    print(f"Best model by validation AUC: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Create ensemble predictions
    ensemble_models = [
        (basic_model, 'torch'),
        (improved_model, 'torch'),
        (deep_model, 'torch'),
        (rf_model, 'sklearn'),
        (gb_model, 'sklearn'),
        (xgb_model, 'sklearn')
    ]
    ensemble_weights = np.array([
        basic_val_metrics['auc'], improved_val_metrics['auc'], deep_val_metrics['auc'],
        rf_val_metrics['auc'], gb_val_metrics['auc'], xgb_val_metrics['auc']
    ])
    ensemble_weights = ensemble_weights / ensemble_weights.sum() if ensemble_weights.sum() > 0 else np.ones_like(ensemble_weights) / len(ensemble_weights)
    
    print("\nEnsemble model weights:")
    for name, weight in zip(['Basic', 'Improved', 'Deep', 'RF', 'GB', 'XGB'], ensemble_weights):
        print(f"{name}: {weight:.4f}")
    
    test_ensemble_probs = create_ensemble_prediction(ensemble_models, scaler, X_test, ensemble_weights)
    test_ensemble_preds = (test_ensemble_probs >= 0.5).astype(int)
    test_precision = precision_score(y_test, test_ensemble_preds) if np.sum(test_ensemble_preds) and np.sum(y_test) else 0
    test_recall = recall_score(y_test, test_ensemble_preds) if np.sum(test_ensemble_preds) and np.sum(y_test) else 0
    test_f1 = f1_score(y_test, test_ensemble_preds) if np.sum(test_ensemble_preds) and np.sum(y_test) else 0
    test_auc = roc_auc_score(y_test, test_ensemble_probs) if len(np.unique(y_test)) > 1 else float('nan')
    test_cm = confusion_matrix(y_test, test_ensemble_preds)
    
    print("\nEnsemble Test Set Results:")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    plot_confusion_matrix(test_cm, "Ensemble Model Test Confusion Matrix",
                          os.path.join(plots_dir, "ensemble_test_confusion_matrix.png"))
    
    # Save test predictions
    test_predictions_df = df_merged[test_mask].copy()
    test_predictions_df['Predicted_Probability'] = test_ensemble_probs
    test_predictions_df['Predicted_Champion'] = test_ensemble_preds
    test_predictions_sorted = pd.concat([
        test_predictions_df[test_predictions_df['Season'] == season].sort_values('Predicted_Probability', ascending=False)
        for season in test_predictions_df['Season'].unique()
    ])
    
    print("\nTop 5 predicted teams for each test season:")
    for season in test_predictions_df['Season'].unique():
        season_df = test_predictions_df[test_predictions_df['Season'] == season].sort_values('Predicted_Probability', ascending=False).head(5)
        print(f"\nSeason {season}:")
        for i, (_, row) in enumerate(season_df.iterrows()):
            mark = "✓" if row['Label'] == 1 else " "
            print(f"{i+1}. {row['School']} - {row['Predicted_Probability']:.4f} {mark}")
    
    test_predictions_sorted.to_csv(
        os.path.join(output_dir, 'test_predictions.csv'),
        columns=['Season', 'School', 'Label', 'Predicted_Probability', 'Predicted_Champion'],
        index=False
    )
    
    # -----------------------------
    # 2025 Predictions
    # -----------------------------
    print("\n" + "=" * 50)
    print("Making Predictions for 2025")
    print("=" * 50)
    
    stats_2025_path = os.path.join(data_seasons_dir, 'school_stats_2025.csv')
    if not os.path.exists(stats_2025_path):
        print("2025 school stats file not found.")
        return
    
    df_2025 = pd.read_csv(stats_2025_path)
    if 'Season' not in df_2025.columns:
        df_2025['Season'] = 2025
    df_2025 = engineer_features(df_2025)
    
    all_features_2025 = base_features.copy()
    for feat in engineered_features:
        if feat in df_2025.columns:
            all_features_2025.append(feat)
    
    missing_feats = [f for f in all_features_2025 if f not in df_2025.columns]
    if missing_feats:
        print(f"Warning: Missing features in 2025 data: {missing_feats}")
        all_features_2025 = [f for f in all_features_2025 if f in df_2025.columns]
    
    print(f"Using {len(all_features_2025)} features for 2025 prediction: {all_features_2025}")
    df_2025 = df_2025.dropna(subset=all_features_2025)
    X_2025 = df_2025[all_features_2025].values.astype(np.float32)
    
    ensemble_probs_2025 = create_ensemble_prediction(ensemble_models, scaler, X_2025, ensemble_weights)
    df_2025['Champion_Probability'] = ensemble_probs_2025
    df_2025_sorted = df_2025.sort_values('Champion_Probability', ascending=False)
    
    print("\nTop 10 teams most likely to be 2025 NCAA Champions:")
    top_teams = df_2025_sorted.head(10)
    for i, (_, row) in enumerate(top_teams.iterrows()):
        print(f"{i+1}. {row['School']} - {row['Champion_Probability']:.4f}")
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_teams['School'][::-1], top_teams['Champion_Probability'][::-1])
    plt.xlabel('Probability of Winning Championship')
    plt.title('Top 10 Teams Most Likely to Win 2025 NCAA Championship')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '2025_championship_predictions.png'))
    plt.close()
    
    output_csv = os.path.join(archive_dir, '2025_champion_predictions.csv')
    df_2025_sorted[['School', 'Champion_Probability'] + all_features_2025].to_csv(output_csv, index=False)
    print(f"\nSaved 2025 champion predictions to {output_csv}")
    
    detailed_csv = os.path.join(archive_dir, '2025_detailed_predictions.csv')
    df_2025_sorted.to_csv(detailed_csv, index=False)
    print(f"Saved detailed 2025 predictions to {detailed_csv}")
    
    # Save PyTorch models
    torch.save(basic_model.state_dict(), os.path.join(models_dir, 'basic_model.pt'))
    torch.save(improved_model.state_dict(), os.path.join(models_dir, 'improved_model.pt'))
    torch.save(deep_model.state_dict(), os.path.join(models_dir, 'deep_model.pt'))
    
    best_team = df_2025_sorted.iloc[0]['School']
    best_prob = df_2025_sorted.iloc[0]['Champion_Probability']
    print(f"\n🏆 Final Prediction for 2025 NCAA Champion: {best_team} ({best_prob:.2%})")
    print("\nKey Statistics for predicted champion:")
    for feature in all_features_2025:
        print(f"{feature}: {df_2025_sorted.iloc[0][feature]}")

    
if __name__ == "__main__":
    main()
