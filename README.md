# March Madness Bracket Predictor

A sophisticated NCAA tournament bracket prediction system that uses machine learning and statistical analysis to forecast tournament outcomes with confidence levels.

## 🏀 Overview

This project provides an advanced bracket prediction system for the NCAA Men's Basketball Tournament (March Madness). It combines historical data, team statistics, and machine learning models to generate predictions for each tournament matchup, including confidence levels for each prediction.

## 🤖 Machine Learning Implementation

### Model Architecture

The prediction system employs a multi-layered approach combining several ML techniques:

1. **Ensemble Learning**
   - Random Forest for base predictions
   - Gradient Boosting for probability estimation
   - Neural Network for pattern recognition in historical data

2. **Feature Engineering**
   - Advanced statistical metrics (SRS, SOS, Win_Pct)
   - Historical tournament performance data
   - Team-specific trend analysis
   - Conference strength indicators
   - Recent form metrics

3. **Confidence Scoring Algorithm**
   ```python
   confidence = base_confidence * seed_factor * round_factor * upset_penalty
   ```
   Where:
   - Base confidence derived from probability ratios
   - Seed factor accounts for seed differentials
   - Round factor adjusts for tournament progression
   - Upset penalty considers historical upset patterns

### Training Process

1. **Data Collection**
   - Historical tournament results (1985-present)
   - Team statistics and metrics
   - Conference performance data
   - Head-to-head matchup history

2. **Model Training**
   - Cross-validation across multiple tournament years
   - Hyperparameter optimization using GridSearchCV
   - Ensemble model weight optimization
   - Regular retraining with new data

3. **Validation Metrics**
   - Prediction accuracy by round
   - Upset prediction success rate
   - Championship prediction accuracy
   - Confidence score calibration

### AI Components

1. **Probability Estimation**
   - Bayesian inference for matchup probabilities
   - Monte Carlo simulations for tournament paths
   - Dynamic probability adjustment based on round progression

2. **Pattern Recognition**
   - Neural network analysis of historical trends
   - Time series analysis of team performance
   - Conference strength impact analysis

3. **Adaptive Learning**
   - Real-time probability adjustments
   - Historical performance weighting
   - Recent form consideration

## ✨ Features

- **Comprehensive Prediction Model**: Uses multiple factors including:
  - Team statistics (SRS, Win Percentage, SOS)
  - Historical tournament performance
  - Seed-based analysis
  - Point differentials
  - Champion probabilities

- **Confidence Scoring**: Each prediction includes a confidence level (0-99%) based on:
  - Probability ratios between teams
  - Seed differentials
  - Tournament round progression
  - Historical upset patterns

- **Detailed Output**: Generates a complete tournament bracket with:
  - Round-by-round predictions
  - Team seeds and matchups
  - Regional breakdowns
  - Championship probabilities

## 🎯 Example Predictions

Based on the 2025 tournament predictions, here are some notable outcomes:

### Final Four
- South vs West: Florida (1) vs Auburn (1)
- East vs Midwest: Duke (1) vs Houston (1)

### Championship
- Winner: Duke (1) over Florida (1)

## 🛠️ Usage

1. Generate predictions:
```bash
python run-bracket-predictor.py --predictions predictions.csv --bracket bracket.txt --output filled_bracket.csv
```

2. View championship probabilities:
```bash
python run-bracket-predictor.py --print-probs
```

3. Generate predictions with confidence levels:
```bash
python run-bracket-predictor.py --confidence
```

## 📊 Output Format

The system generates a CSV file containing:
- Round information
- Team matchups
- Seeds
- Regional assignments
- Confidence levels
- Winner predictions

## 🔧 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- Additional dependencies listed in requirements.txt

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📈 Future Improvements

- Integration with real-time tournament data
- Enhanced confidence scoring algorithms
- Web interface for bracket visualization
- Historical performance analysis
- Team-specific trend analysis
- Advanced deep learning models
- Real-time model updates during tournament
- Enhanced feature engineering pipeline

## 🙏 Acknowledgments

- NCAA for tournament data and structure
- Contributors and maintainers
- Basketball analytics community
- Open-source ML libraries and frameworks

---

Made with ❤️ for March Madness fans everywhere
