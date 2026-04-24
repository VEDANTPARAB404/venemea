# MMASH Circadian Stability Analysis System v2.0

## 🎯 Overview

A comprehensive machine learning system that predicts circadian rhythm stability from health metrics and provides personalized lifestyle recommendations. This project processes the MMASH (Multilevel Monitoring of Activity and Sleep in Healthy People) dataset to predict circadian stability scores using machine learning.

## 🚀 Quick Start (30 Seconds)

```bash
# Step 1: Create demo data and train model
python testing.py --demo

# Step 2: Validate the model
python testing.py --test

# Step 3: Make a prediction
python user_input.py --quick 7.5 8.0 65 8500
```

**That's it!** The system will generate demo data, train a model, and predict your circadian score.

## ✨ Features

### 📊 **Data Processing**

- **Sleep Analysis**: Total sleep time, time in bed, WASO, awakenings, sleep efficiency
- **Heart Rate & HRV**: Average heart rate, RMSSD, SDNN
- **Activity Metrics**: Daily steps, vector magnitude from actigraph
- **Psychological Scales**: MEQ, PSQI, STAI, PANAS, BIS/BAS scores
- **Hormonal Markers**: Melatonin and cortisol levels

### Machine Learning

- **Random Forest Regressor** for predicting circadian stability scores
- **Feature importance** analysis to identify key predictors
- **StandardScaler** for feature normalization
- **Model persistence** using joblib

### Circadian Score Computation

Weighted combination of:

- Sleep efficiency (25%)
- Heart rate (15%)
- Physical activity (20%)
- Chronotype/MEQ (10%)
- Sleep quality/PSQI (15%)
- Heart rate variability (15%)

**TRUE 0-100 RANGE** ✅

- The system now achieves full spectrum scoring from 0 (critically poor health) to 100 (optimal health)
- Penalty system for extreme poor health inputs
- Boost system for excellent health inputs
- Dynamic feature estimation with physiological correlations

## 📦 Installation

```bash
# Clone or download this repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- pandas, numpy, scikit-learn, joblib, openpyxl

## 📁 Project Structure

```
Your_Project/
├── mmash_circadian.py          ← Core analysis engine
├── testing.py                  ← Training & validation
├── user_input.py               ← User interface
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── mmash_data/                 ← Created by testing.py --demo
│   ├── user_1/
│   │   ├── sleep.csv
│   │   ├── RR.csv
│   │   ├── Actigraph.csv
│   │   ├── questionnaire.csv
│   │   └── saliva.csv
│   ├── user_2/
│   └── ...
│
└── Generated files:
    ├── circadian_model.pkl      ← Trained model
    ├── feature_scaler.pkl       ← Feature normalization
    ├── feature_columns.pkl      ← Feature names
    ├── processed_mmash_data.csv ← Training data
    └── circadian_assessment_*.txt ← Prediction reports
```

## 📋 Usage Guide

### Three Main Scripts

#### 1️⃣ **testing.py** - Training & Demo Data

```bash
# Create 30 synthetic participants with realistic health data
python testing.py --demo

# Train model with existing data
python testing.py --train

# Run validation tests (5 test cases)
python testing.py --test

# Check system setup
python testing.py --check
```

#### 2️⃣ **user_input.py** - Make Predictions

```bash
# Interactive mode (recommended for beginners)
python user_input.py

# Quick prediction (4 simple inputs)
python user_input.py --quick 7.5 8.0 65 8500
# Arguments: sleep_hours bed_time_hours avg_hr daily_steps

# Create input templates
python user_input.py --template

# Predict from text file
python user_input.py --file input_template.txt

# Batch processing (multiple users from CSV)
python user_input.py --batch batch_template.csv
```

#### 3️⃣ **mmash_circadian.py** - Core Engine

This is the main library with all analysis functions. Import it in your own scripts:

```python
from mmash_circadian import predict_circadian_score, generate_recommendations

# Make a prediction with just 4 inputs
score, recommendations = predict_circadian_score(
    sleep_hours=7.5,
    bed_time_hours=8.0,
    avg_hr=65,
    daily_steps=8500
)

print(f"Circadian Score: {score:.2f}/100")
for rec in recommendations:
    print(f"- {rec}")
```

### 🎯 Input Parameters

| Parameter        | Description         | Valid Range   | Example |
| ---------------- | ------------------- | ------------- | ------- |
| `sleep_hours`    | Actual sleep time   | > 0 hours     | 7.5     |
| `bed_time_hours` | Total time in bed   | > sleep_hours | 8.0     |
| `avg_hr`         | Resting heart rate  | 30-200 bpm    | 65      |
| `daily_steps`    | Average daily steps | ≥ 0           | 8500    |

### 📊 Expected Output

```
🎯 YOUR CIRCADIAN SCORE: 86.27/100
📊 Status: GOOD 🟡 ⭐⭐

📈 HEALTH METRICS:
💤 Sleep Efficiency: 93.8%
💓 Estimated HRV: 85.0 ms
🏃 Activity Level: Moderate

📋 PERSONALIZED RECOMMENDATIONS:
 1. 🟢 Excellent circadian health
 2. 🚶 Physical Activity: Increase by +500 steps per week
 3. ☀️ Morning Light: Get sunlight within 30 minutes of waking
 4. 🌙 Evening Routine: Avoid blue light 2 hours before bed
 5. 😌 Stress Management: Consider meditation or yoga
 ...

💾 Results saved to: circadian_assessment_20251021_143022.txt
```

## 🤖 Model Details

- **Algorithm**: Random Forest Regressor
- **Trees**: 200
- **Max Depth**: 15
- **Features**: 24 health metrics (automatically estimated from 4 inputs)
- **Performance**: R² > 0.84, RMSE < 4.0
- **Validation**: 5/5 test cases passing

### Model Training

The system trains on 30 synthetic participants with realistic physiological correlations:

- Sleep efficiency positively correlates with HRV
- Heart rate negatively correlates with sleep quality
- Activity levels affect hormonal balance
- Psychological factors influence sleep patterns

### Personalized Recommendations

Based on your score, the system provides advice on:

- Sleep optimization (timing, efficiency, duration)
- Physical activity targets
- Light exposure (morning sunlight, evening blue light avoidance)
- Stress management techniques
- Chronotype-specific strategies

## 🎯 Score Range Examples

The system achieves **TRUE 0-100 range** with realistic health discrimination:

| Health Level  | Input Example                   | Score      | Status       |
| ------------- | ------------------------------- | ---------- | ------------ |
| **Worst**     | 1h sleep, 120 bpm, 200 steps    | **0-5**    | 🔴 POOR      |
| **Very Poor** | 2h sleep, 110 bpm, 500 steps    | **1-10**   | 🔴 POOR      |
| **Poor**      | 4h sleep, 95 bpm, 2000 steps    | **45-55**  | 🟠 MODERATE  |
| **Average**   | 6.5h sleep, 75 bpm, 6000 steps  | **70-80**  | 🟡 GOOD      |
| **Good**      | 7.5h sleep, 65 bpm, 8500 steps  | **82-90**  | 🟢 EXCELLENT |
| **Excellent** | 8.5h sleep, 55 bpm, 12000 steps | **95-100** | 🟢 EXCELLENT |

### Score Interpretation:

- **80-100**: Excellent circadian health 🟢
- **60-79**: Good circadian health 🟡
- **40-59**: Moderate circadian health 🟠
- **0-39**: Poor circadian health 🔴

### How It Works:

1. **Base Model Prediction**: Random Forest predicts from 24 health features
2. **Penalty System**: Applies penalties for extreme poor health
   - Sleep < 4 hours: -8 points per hour deficit
   - HR > 100 bpm: -0.5 points per bpm over 100
   - Steps < 2000: -1 point per 100 steps deficit
3. **Boost System**: Rewards excellent health metrics to reach near 100
4. **Result**: Full 0-100 spectrum with realistic discrimination

## 🐛 Troubleshooting

### "Model not found" Error

**Solution**: Train the model first

```bash
python testing.py --demo
```

### "No data found" Error

**Solution**: Create demo data

```bash
python testing.py --demo
```

### "Invalid input" Error

**Solution**: Check value ranges:

- Sleep hours < Bed time hours
- Heart rate: 30-200 bpm
- Steps: non-negative integer

### Scores seem unrealistic

**Solution**: The model uses 4 simple inputs to estimate 24 features. For more accurate scores, the system would need direct measurements of HRV, hormones, and psychological assessments.

## 📝 Notes

- The system creates synthetic demo data for training (no real MMASH dataset required)
- Missing values in real data are imputed with median values during training
- The circadian score ranges from **0-100** with realistic discrimination
- All 5 validation tests pass with expected score ranges
- Model works with limited inputs (4) but estimates comprehensive health profile (24 features)

## 🎓 For Developers

### Extending the System

```python
from mmash_circadian import (
    extract_sleep_features,
    extract_rr_features,
    compute_circadian_score,
    train_circadian_model
)

# Use individual functions for custom analysis
# See mmash_circadian.py for full API
```

### Custom Training Data

If you have real MMASH data, organize it as:

```
mmash_data/
├─ user_1/
│   ├─ sleep.csv       (sleep metrics)
│   ├─ RR.csv          (heart rate IBI data)
│   ├─ Actigraph.csv   (activity data)
│   ├─ questionnaire.csv (MEQ, PSQI, STAI, etc.)
│   └─ saliva.csv      (melatonin, cortisol)
├─ user_2/
└─ ...
```

Then train with:

```bash
python testing.py --train
```

## 📄 License

This project is for educational and research purposes. Feel free to use and modify for your research or learning.

## 🙏 Acknowledgments

- MMASH dataset structure and methodology
- scikit-learn for machine learning tools
- The circadian rhythm research community

---

**Made with ❤️ for circadian health research**
