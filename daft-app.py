import os
import glob
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# PyTorch Model Definition
# -----------------------------
class ChampionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ChampionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # Output 2 classes: 0 = not champion, 1 = champion
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Data Loading and Preprocessing Functions
# -----------------------------
def load_seasons_summary(file_path):
    """
    Load the seasons summary CSV and extract the season year and champion name.
    Checks for one of the expected columns ('full_url', 'URL', or 'Season') to extract the year.
    """
    df = pd.read_csv(file_path)
    # Debug: print out columns
    print("Columns in seasons summary:", df.columns.tolist())
    
    if 'full_url' in df.columns:
        df['Year'] = df['full_url'].str.extract(r'/men/(\d{4})\.html').astype(int)
    elif 'URL' in df.columns:
        df['Year'] = df['URL'].str.extract(r'/men/(\d{4})\.html').astype(int)
    elif 'Season' in df.columns:
        df['Year'] = df['Season'].apply(lambda x: int(re.findall(r'\d{4}', str(x))[-1]))
    else:
        raise ValueError("None of the expected columns ('full_url', 'URL', 'Season') were found in the seasons summary file.")
    return df[['Year', 'Champion']]

def load_school_stats(data_dir, start_year, end_year):
    """
    Load all school_stats CSV files from data_dir between start_year and end_year.
    Assumes filenames are of the form "school_stats_YYYY.csv".
    """
    pattern = os.path.join(data_dir, 'school_stats_*.csv')
    all_files = glob.glob(pattern)
    df_list = []
    for file in all_files:
        m = re.search(r'school_stats_(\d{4})\.csv', file)
        if m:
            year = int(m.group(1))
            if start_year <= year <= end_year:
                df_temp = pd.read_csv(file)
                # Add a 'Season' column if not present (using the year from the filename)
                if 'Season' not in df_temp.columns:
                    df_temp['Season'] = year
                df_list.append(df_temp)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def preprocess_data(df, seasons_summary, features):
    """
    Merge the season summary into the school stats DataFrame and create a binary target.
    The target is 1 if the team is the champion for that season, else 0.
    Uses a simple case-insensitive substring check between the 'School' column and the 'Champion' name.
    """
    # Merge on season/year; using df['Season'] and seasons_summary['Year']
    df = pd.merge(df, seasons_summary, left_on='Season', right_on='Year', how='left')
    
    def is_champion(row):
        if pd.isna(row['Champion']):
            return 0
        # Simple check: if the champion name appears in the School name (ignoring case)
        return 1 if row['Champion'].lower() in row['School'].lower() else 0
    
    df['Label'] = df.apply(is_champion, axis=1)
    # Drop rows with missing features
    df = df.dropna(subset=features)
    X = df[features].values.astype(np.float32)
    y = df['Label'].values.astype(np.int64)
    return X, y, df

def train_model(model, criterion, optimizer, train_loader, num_epochs=100):
    """
    Standard PyTorch training loop.
    """
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
        if (epoch+1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# -----------------------------
# Main function
# -----------------------------
def main():
    # Set paths relative to project root
    data_seasons_dir = os.path.join('data-seasons')
    seasons_csv_path = os.path.join(data_seasons_dir, 'cbb_seasons.csv')
    
    # Use seasons from 2000 to 2024 for training
    start_year = 2000
    end_year = 2024
    
    # Load season summary data
    seasons_summary = load_seasons_summary(seasons_csv_path)
    print("Seasons summary (first few rows):")
    print(seasons_summary.head())
    
    # Load school stats data for training
    df_stats = load_school_stats(data_seasons_dir, start_year, end_year)
    print(f"Loaded school stats data with {len(df_stats)} rows from {start_year} to {end_year}.")
    
    # Choose feature columns (adjust based on your CSV's available columns)
    features = ['Win_Pct', 'SRS', 'SOS', 'Points_For', 'Points_Against']
    
    # Preprocess the data: merge champion info and create target labels
    X, y, df_merged = preprocess_data(df_stats, seasons_summary, features)
    print("After preprocessing, feature matrix shape:", X.shape)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Create a DataLoader for training
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define the model, loss function, and optimizer
    input_dim = len(features)
    hidden_dim = 64
    model = ChampionPredictor(input_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 100
    print("Training model...")
    train_model(model, criterion, optimizer, train_loader, num_epochs)
    
    # Evaluate on the validation set
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_val_tensor).float().mean().item()
    print(f'Validation Accuracy: {accuracy*100:.2f}%')
    
    # -----------------------------
    # Prediction for 2025
    # -----------------------------
    # Load 2025 school stats data
    stats_2025_path = os.path.join(data_seasons_dir, 'school_stats_2025.csv')
    if not os.path.exists(stats_2025_path):
        print("2025 school stats file not found.")
        return
    
    df_2025 = pd.read_csv(stats_2025_path)
    # Ensure a Season column exists (if not, add it)
    if 'Season' not in df_2025.columns:
        df_2025['Season'] = 2025
    # Drop rows missing any of the selected features
    df_2025 = df_2025.dropna(subset=features)
    X_2025 = df_2025[features].values.astype(np.float32)
    # Scale using the previously fitted scaler
    X_2025 = scaler.transform(X_2025)
    X_2025_tensor = torch.tensor(X_2025, dtype=torch.float32)
    
    # Get predicted probabilities for being champion
    model.eval()
    with torch.no_grad():
        outputs_2025 = model(X_2025_tensor)
        # Apply softmax and extract probability for class 1 (champion)
        probabilities = torch.softmax(outputs_2025, dim=1)[:, 1].numpy()
    
    # Add the probability to the DataFrame
    df_2025['Champion_Prob'] = probabilities
    # Identify the team with the highest predicted champion probability
    best_idx = df_2025['Champion_Prob'].idxmax()
    best_team = df_2025.loc[best_idx, 'School']
    print(f'\nPredicted 2025 Champion: {best_team}')
    
    # Optionally, save the prediction results to CSV in the archive folder
    archive_dir = 'archive'
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    output_csv = os.path.join(archive_dir, '2025_champion_predictions.csv')
    df_2025[['School', 'Champion_Prob']].to_csv(output_csv, index=False)
    print(f'Saved 2025 champion predictions to {output_csv}')

if __name__ == "__main__":
    main()
