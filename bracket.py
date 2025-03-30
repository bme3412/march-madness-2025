import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
import argparse

# Function to load the saved prediction data or model outputs
def load_predictions(prediction_file):
    """Load the champion probability predictions from the CSV file"""
    try:
        df = pd.read_csv(prediction_file)
        print(f"Loaded predictions for {len(df)} teams")
        return df
    except FileNotFoundError:
        print(f"Error: Prediction file '{prediction_file}' not found")
        return None

# Function to extract teams and seeds from the bracket
def parse_bracket(bracket_file):
    """Parse the bracket structure from a text file"""
    regions = {}
    current_region = None
    
    with open(bracket_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.endswith("REGION"):
            current_region = line
            regions[current_region] = []
        elif line and current_region and "vs" in line:
            # Format: Seed Team vs Seed Team
            parts = line.split("vs")
            team1_parts = parts[0].strip().split(" ", 1)
            team2_parts = parts[1].strip().split(" ", 1)
            
            seed1 = int(team1_parts[0])
            team1 = team1_parts[1]
            seed2 = int(team2_parts[0])
            team2 = team2_parts[1]
            
            regions[current_region].append({
                "team1": team1,
                "seed1": seed1,
                "team2": team2,
                "seed2": seed2
            })
    
    return regions

# Function to normalize team names to match between bracket and predictions
def normalize_team_name(name):
    """Standardize team names to handle variations"""
    # Common replacements
    replacements = {
        "Saint": "St.",
        "State": "St",
        "College": "",
        "University": "",
        "(": "",
        ")": ""
    }
    
    # Remove common words and standardize
    name = name.strip()
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name.strip()

# Function to get team probability from predictions
def get_team_probability(team_name, predictions_df):
    """Get the champion probability for a team, with fuzzy matching if needed"""
    # First try exact match
    match = predictions_df[predictions_df['School'] == team_name]
    
    if not match.empty:
        return match.iloc[0]['Champion_Probability']
    
    # Try normalized name
    normalized_name = normalize_team_name(team_name)
    for _, row in predictions_df.iterrows():
        if normalize_team_name(row['School']) == normalized_name:
            return row['Champion_Probability']
    
    # Try substring match (in case of abbreviations or partial names)
    for _, row in predictions_df.iterrows():
        if normalized_name in normalize_team_name(row['School']) or normalize_team_name(row['School']) in normalized_name:
            return row['Champion_Probability']
    
    # If still not found, return a very low probability
    print(f"Warning: Could not find probability for team: {team_name}")
    return 0.001  # Very low default probability

# Function to predict winner of a matchup
def predict_matchup(team1, team2, seed1, seed2, predictions_df, round_number=1):
    """Predict the winner of a matchup based on team probabilities and seeds"""
    prob1 = get_team_probability(team1, predictions_df)
    prob2 = get_team_probability(team2, predictions_df)
    
    # Apply seed-based adjustment for early rounds (upsets are less likely in earlier rounds)
    seed_factor = max(0, 0.15 - (round_number - 1) * 0.03)  # Decreases with each round
    
    # Calculate seed difference and apply adjustment
    seed_diff = seed2 - seed1  # Positive if team1 is higher seeded (lower number)
    adjustment = seed_diff * seed_factor
    
    # Adjusted probabilities
    adjusted_prob1 = prob1 + adjustment
    adjusted_prob2 = prob2 - adjustment
    
    if adjusted_prob1 > adjusted_prob2:
        return team1, seed1, prob1
    else:
        return team2, seed2, prob2

# Function to simulate the tournament
def simulate_tournament(regions, predictions_df):
    """Simulate the entire tournament and return the results for each round"""
    results = {
        "Round of 64": [],
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final 4": [],
        "Championship": [],
        "Champion": []
    }
    
    # First round - Round of 64
    round_of_32 = {}
    for region_name, matchups in regions.items():
        round_of_32[region_name] = []
        for matchup in matchups:
            winner, seed, prob = predict_matchup(
                matchup["team1"], matchup["team2"], 
                matchup["seed1"], matchup["seed2"], 
                predictions_df, 1
            )
            
            results["Round of 64"].append({
                "region": region_name,
                "team1": matchup["team1"],
                "seed1": matchup["seed1"],
                "team2": matchup["team2"],
                "seed2": matchup["seed2"],
                "winner": winner,
                "winner_seed": seed,
                "winner_prob": prob
            })
            
            round_of_32[region_name].append({"team": winner, "seed": seed})
    
    # Second round - Round of 32
    sweet_16 = {}
    for region_name, winners in round_of_32.items():
        sweet_16[region_name] = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                team1, seed1 = winners[i]["team"], winners[i]["seed"]
                team2, seed2 = winners[i+1]["team"], winners[i+1]["seed"]
                
                winner, seed, prob = predict_matchup(team1, team2, seed1, seed2, predictions_df, 2)
                
                results["Round of 32"].append({
                    "region": region_name,
                    "team1": team1,
                    "seed1": seed1,
                    "team2": team2,
                    "seed2": seed2,
                    "winner": winner,
                    "winner_seed": seed,
                    "winner_prob": prob
                })
                
                sweet_16[region_name].append({"team": winner, "seed": seed})
    
    # Third round - Sweet 16
    elite_8 = {}
    for region_name, winners in sweet_16.items():
        elite_8[region_name] = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                team1, seed1 = winners[i]["team"], winners[i]["seed"]
                team2, seed2 = winners[i+1]["team"], winners[i+1]["seed"]
                
                winner, seed, prob = predict_matchup(team1, team2, seed1, seed2, predictions_df, 3)
                
                results["Sweet 16"].append({
                    "region": region_name,
                    "team1": team1,
                    "seed1": seed1,
                    "team2": team2,
                    "seed2": seed2,
                    "winner": winner,
                    "winner_seed": seed,
                    "winner_prob": prob
                })
                
                elite_8[region_name].append({"team": winner, "seed": seed})
    
    # Fourth round - Elite 8
    final_4 = []
    for region_name, winners in elite_8.items():
        if len(winners) >= 2:
            team1, seed1 = winners[0]["team"], winners[0]["seed"]
            team2, seed2 = winners[1]["team"], winners[1]["seed"]
            
            winner, seed, prob = predict_matchup(team1, team2, seed1, seed2, predictions_df, 4)
            
            results["Elite 8"].append({
                "region": region_name,
                "team1": team1,
                "seed1": seed1,
                "team2": team2,
                "seed2": seed2,
                "winner": winner,
                "winner_seed": seed,
                "winner_prob": prob
            })
            
            final_4.append({"team": winner, "seed": seed, "region": region_name})
    
    # Final Four
    championship = []
    # Match first two regions against each other
    if len(final_4) >= 2:
        team1, seed1 = final_4[0]["team"], final_4[0]["seed"]
        team2, seed2 = final_4[1]["team"], final_4[1]["seed"]
        
        winner, seed, prob = predict_matchup(team1, team2, seed1, seed2, predictions_df, 5)
        
        results["Final 4"].append({
            "matchup": f"{final_4[0]['region']} vs {final_4[1]['region']}",
            "team1": team1,
            "seed1": seed1,
            "team2": team2,
            "seed2": seed2,
            "winner": winner,
            "winner_seed": seed,
            "winner_prob": prob
        })
        
        championship.append({"team": winner, "seed": seed})
    
    # Match other two regions against each other
    if len(final_4) >= 4:
        team1, seed1 = final_4[2]["team"], final_4[2]["seed"]
        team2, seed2 = final_4[3]["team"], final_4[3]["seed"]
        
        winner, seed, prob = predict_matchup(team1, team2, seed1, seed2, predictions_df, 5)
        
        results["Final 4"].append({
            "matchup": f"{final_4[2]['region']} vs {final_4[3]['region']}",
            "team1": team1,
            "seed1": seed1,
            "team2": team2,
            "seed2": seed2,
            "winner": winner,
            "winner_seed": seed,
            "winner_prob": prob
        })
        
        championship.append({"team": winner, "seed": seed})
    
    # Championship game
    if len(championship) >= 2:
        team1, seed1 = championship[0]["team"], championship[0]["seed"]
        team2, seed2 = championship[1]["team"], championship[1]["seed"]
        
        winner, seed, prob = predict_matchup(team1, team2, seed1, seed2, predictions_df, 6)
        
        results["Championship"].append({
            "team1": team1,
            "seed1": seed1,
            "team2": team2,
            "seed2": seed2,
            "winner": winner,
            "winner_seed": seed,
            "winner_prob": prob
        })
        
        results["Champion"].append({
            "team": winner,
            "seed": seed,
            "probability": prob
        })
    
    return results

# Format results for printing
def print_results(results):
    """Print the tournament results in a readable format"""
    print("\n===== TOURNAMENT PREDICTIONS =====\n")
    
    # Round of 64
    print("ROUND OF 64:")
    for matchup in results["Round of 64"]:
        print(f"{matchup['region']}: ({matchup['seed1']}) {matchup['team1']} vs ({matchup['seed2']}) {matchup['team2']} → ({matchup['winner_seed']}) {matchup['winner']}")
    
    print("\nROUND OF 32:")
    for matchup in results["Round of 32"]:
        print(f"{matchup['region']}: ({matchup['seed1']}) {matchup['team1']} vs ({matchup['seed2']}) {matchup['team2']} → ({matchup['winner_seed']}) {matchup['winner']}")
    
    print("\nSWEET 16:")
    for matchup in results["Sweet 16"]:
        print(f"{matchup['region']}: ({matchup['seed1']}) {matchup['team1']} vs ({matchup['seed2']}) {matchup['team2']} → ({matchup['winner_seed']}) {matchup['winner']}")
    
    print("\nELITE 8:")
    for matchup in results["Elite 8"]:
        print(f"{matchup['region']}: ({matchup['seed1']}) {matchup['team1']} vs ({matchup['seed2']}) {matchup['team2']} → ({matchup['winner_seed']}) {matchup['winner']}")
    
    print("\nFINAL FOUR:")
    for matchup in results["Final 4"]:
        print(f"{matchup['matchup']}: ({matchup['seed1']}) {matchup['team1']} vs ({matchup['seed2']}) {matchup['team2']} → ({matchup['winner_seed']}) {matchup['winner']}")
    
    print("\nCHAMPIONSHIP:")
    if results["Championship"]:
        championship = results["Championship"][0]
        print(f"({championship['seed1']}) {championship['team1']} vs ({championship['seed2']}) {championship['team2']} → ({championship['winner_seed']}) {championship['winner']}")
    
    print("\nCHAMPION:")
    if results["Champion"]:
        champion = results["Champion"][0]
        print(f"({champion['seed']}) {champion['team']} - Championship Probability: {champion['probability']:.4f}")

def export_bracket(results, output_file):
    """Export the filled bracket to a CSV file"""
    # Collect all picks from all rounds
    all_picks = []
    
    for round_name, matchups in results.items():
        if round_name != "Champion":  # Skip the champion summary
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
    
    # Create DataFrame and export to CSV
    picks_df = pd.DataFrame(all_picks)
    picks_df.to_csv(output_file, index=False)
    print(f"\nExported bracket to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='NCAA Tournament Bracket Predictor')
    parser.add_argument('--predictions', default='archive/2025_champion_predictions.csv',
                        help='Path to the predictions CSV file')
    parser.add_argument('--bracket', default='bracket_2025.txt',
                        help='Path to the bracket structure file')
    parser.add_argument('--output', default='filled_bracket_2025.csv',
                        help='Output file for the filled bracket')
    
    args = parser.parse_args()
    
    # Load predictions
    predictions_df = load_predictions(args.predictions)
    if predictions_df is None:
        return
    
    # Parse bracket structure (if it doesn't exist, use hardcoded bracket)
    if os.path.exists(args.bracket):
        regions = parse_bracket(args.bracket)
    else:
        print(f"Bracket file {args.bracket} not found. Using hardcoded bracket structure.")
        # Hardcoded bracket from the image
        regions = {
            "SOUTH (ATLANTA)": [
                {"team1": "Auburn", "seed1": 2, "team2": "Alabama St/Saint Francis", "seed2": 15},
                {"team1": "Louisville", "seed1": 7, "team2": "Creighton", "seed2": 10},
                {"team1": "Michigan", "seed1": 3, "team2": "UC San Diego", "seed2": 14},
                {"team1": "Texas A&M", "seed1": 6, "team2": "Yale", "seed2": 11},
                {"team1": "Ole Miss", "seed1": 8, "team2": "San Diego St/North Carolina", "seed2": 9},
                {"team1": "Iowa State", "seed1": 5, "team2": "Lipscomb", "seed2": 12},
                {"team1": "Marquette", "seed1": 4, "team2": "New Mexico", "seed2": 13},
                {"team1": "Michigan State", "seed1": 1, "team2": "Bryant", "seed2": 16}
            ],
            "WEST (SAN FRANCISCO)": [
                {"team1": "Florida", "seed1": 1, "team2": "Norfolk State", "seed2": 16},
                {"team1": "UConn", "seed1": 8, "team2": "Oklahoma", "seed2": 9},
                {"team1": "Memphis", "seed1": 5, "team2": "Colorado State", "seed2": 12},
                {"team1": "Maryland", "seed1": 4, "team2": "Grand Canyon", "seed2": 13},
                {"team1": "Missouri", "seed1": 6, "team2": "Drake", "seed2": 11},
                {"team1": "Texas Tech", "seed1": 3, "team2": "UNCW", "seed2": 14},
                {"team1": "Kansas", "seed1": 7, "team2": "Arkansas", "seed2": 10},
                {"team1": "St. John's", "seed1": 2, "team2": "Omaha", "seed2": 15}
            ],
            "EAST (NEWARK)": [
                {"team1": "Duke", "seed1": 1, "team2": "American/Mount St Mary's", "seed2": 16},
                {"team1": "Mississippi St", "seed1": 8, "team2": "Baylor", "seed2": 9},
                {"team1": "Oregon", "seed1": 5, "team2": "Liberty", "seed2": 12},
                {"team1": "Arizona", "seed1": 4, "team2": "Akron", "seed2": 13},
                {"team1": "BYU", "seed1": 6, "team2": "VCU", "seed2": 11},
                {"team1": "Wisconsin", "seed1": 3, "team2": "Montana", "seed2": 14},
                {"team1": "Saint Mary's", "seed1": 7, "team2": "Vanderbilt", "seed2": 10},
                {"team1": "Alabama", "seed1": 2, "team2": "Robert Morris", "seed2": 15}
            ],
            "MIDWEST (INDIANAPOLIS)": [
                {"team1": "Houston", "seed1": 1, "team2": "SIU Edwardsville", "seed2": 16},
                {"team1": "Gonzaga", "seed1": 8, "team2": "Georgia", "seed2": 9},
                {"team1": "Clemson", "seed1": 5, "team2": "McNeese", "seed2": 12},
                {"team1": "Purdue", "seed1": 4, "team2": "High Point", "seed2": 13},
                {"team1": "Illinois", "seed1": 6, "team2": "Texas/Xavier", "seed2": 11},
                {"team1": "Kentucky", "seed1": 3, "team2": "Troy", "seed2": 14},
                {"team1": "UCLA", "seed1": 7, "team2": "Utah State", "seed2": 10},
                {"team1": "Tennessee", "seed1": 2, "team2": "Wofford", "seed2": 15}
            ]
        }
    
    # Simulate the tournament
    results = simulate_tournament(regions, predictions_df)
    
    # Print the results
    print_results(results)
    
    # Export the filled bracket
    export_bracket(results, args.output)

if __name__ == "__main__":
    main()