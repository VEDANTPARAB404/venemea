"""
MMASH Dataset Circadian Stability Analysis - Main Engine
=========================================================
This script processes multi-modal health data from participants to:
1. Extract sleep, heart rate, activity, psychological, and hormonal features
2. Compute a circadian stability score (weighted 0-100 scale)
3. Train a Random Forest Regressor to predict circadian stability
4. Provide personalized lifestyle recommendations

Author: Generated for MMASH Dataset Analysis
Date: October 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# Configuration Constants
DATA_DIR = "mmash_data"
MODEL_PATH = "circadian_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
OUTPUT_CSV = "processed_mmash_data.csv"

CIRCADIAN_WEIGHTS = {
    'sleep_efficiency': 0.25,
    'avg_heart_rate': 0.15,
    'total_steps': 0.20,
    'meq_score': 0.10,
    'psqi_score': 0.15,
    'hrv_rmssd': 0.15
}


def extract_sleep_features(sleep_df: pd.DataFrame) -> Dict:
    """
    Extract sleep-related features from sleep dataframe.
    
    Parameters:
        sleep_df: DataFrame with columns: total_sleep_time, total_minutes_in_bed, WASO, awakenings, fragmentation_index
    
    Returns:
        Dictionary containing sleep metrics including computed sleep_efficiency
    """
    try:
        # Calculate averages
        total_sleep_time = sleep_df['total_sleep_time'].mean() if 'total_sleep_time' in sleep_df.columns else np.nan
        time_in_bed = sleep_df['total_minutes_in_bed'].mean() if 'total_minutes_in_bed' in sleep_df.columns else np.nan
        waso = sleep_df['WASO'].mean() if 'WASO' in sleep_df.columns else np.nan
        awakenings = sleep_df['awakenings'].mean() if 'awakenings' in sleep_df.columns else np.nan
        fragmentation = sleep_df['fragmentation_index'].mean() if 'fragmentation_index' in sleep_df.columns else np.nan
        
        # Calculate sleep efficiency: (total_sleep_time / time_in_bed) * 100
        sleep_efficiency = min(100, (total_sleep_time / time_in_bed * 100)) if time_in_bed > 0 else np.nan
        
        features = {
            'total_sleep_time': total_sleep_time,
            'total_minutes_in_bed': time_in_bed,
            'waso': waso,
            'awakenings': awakenings,
            'fragmentation': fragmentation,
            'sleep_efficiency': sleep_efficiency
        }
        
        return features
    except Exception as e:
        print(f"    ⚠ Warning: Error reading sleep data - {e}")
        return {
            'total_sleep_time': np.nan,
            'total_minutes_in_bed': np.nan,
            'waso': np.nan,
            'awakenings': np.nan,
            'fragmentation': np.nan,
            'sleep_efficiency': np.nan
        }


def extract_rr_features(rr_df: pd.DataFrame) -> Dict:
    """
    Extract heart rate and HRV features from RR interval dataframe.
    
    Parameters:
        rr_df: DataFrame with column 'ibi_s' (inter-beat interval in seconds)
    
    Returns:
        Dictionary containing avg_heart_rate, hr_std, hrv_rmssd, hrv_sdnn
    """
    try:
        if 'ibi_s' not in rr_df.columns:
            raise ValueError("Column 'ibi_s' not found")
        
        # Calculate heart rate from IBI (inter-beat interval in seconds)
        rr_df['heart_rate'] = 60 / rr_df['ibi_s']
        avg_hr = rr_df['heart_rate'].mean()
        hr_std = rr_df['heart_rate'].std()
        
        # Calculate HRV metrics
        # RMSSD: Root Mean Square of Successive Differences (in milliseconds)
        ibi_ms = rr_df['ibi_s'] * 1000  # Convert to milliseconds
        successive_diffs = np.diff(ibi_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # SDNN: Standard Deviation of NN intervals (in milliseconds)
        sdnn = ibi_ms.std()
        
        features = {
            'avg_heart_rate': avg_hr,
            'hr_std': hr_std,
            'hrv_rmssd': rmssd,
            'hrv_sdnn': sdnn
        }
        
        return features
    except Exception as e:
        print(f"    ⚠ Warning: Error reading RR data - {e}")
        return {
            'avg_heart_rate': np.nan,
            'hr_std': np.nan,
            'hrv_rmssd': np.nan,
            'hrv_sdnn': np.nan
        }


def extract_actigraph_features(actigraph_df: pd.DataFrame) -> Dict:
    """
    Extract activity features from Actigraph dataframe.
    
    Parameters:
        actigraph_df: DataFrame with columns: steps, vector_magnitude
    
    Returns:
        Dictionary containing total_steps, avg_daily_steps, vector_magnitude
    """
    try:
        total_steps = actigraph_df['steps'].sum() if 'steps' in actigraph_df.columns else np.nan
        avg_daily_steps = actigraph_df['steps'].mean() if 'steps' in actigraph_df.columns else np.nan
        vector_magnitude = actigraph_df['vector_magnitude'].mean() if 'vector_magnitude' in actigraph_df.columns else np.nan
        
        features = {
            'total_steps': total_steps,
            'avg_daily_steps': avg_daily_steps,
            'vector_magnitude': vector_magnitude
        }
        
        return features
    except Exception as e:
        print(f"    ⚠ Warning: Error reading Actigraph data - {e}")
        return {
            'total_steps': np.nan,
            'avg_daily_steps': np.nan,
            'vector_magnitude': np.nan
        }


def extract_questionnaire_features(quest_df: pd.DataFrame) -> Dict:
    """
    Extract psychological assessment scores from questionnaire dataframe.
    
    Parameters:
        quest_df: DataFrame with columns: MEQ, PSQI, STAI, PANAS_positive, PANAS_negative, BIS, BAS
    
    Returns:
        Dictionary containing all questionnaire scores
    """
    try:
        features = {
            'meq_score': quest_df['MEQ'].iloc[0] if 'MEQ' in quest_df.columns and len(quest_df) > 0 else np.nan,
            'psqi_score': quest_df['PSQI'].iloc[0] if 'PSQI' in quest_df.columns and len(quest_df) > 0 else np.nan,
            'stai_score': quest_df['STAI'].iloc[0] if 'STAI' in quest_df.columns and len(quest_df) > 0 else np.nan,
            'panas_positive': quest_df['PANAS_positive'].iloc[0] if 'PANAS_positive' in quest_df.columns and len(quest_df) > 0 else np.nan,
            'panas_negative': quest_df['PANAS_negative'].iloc[0] if 'PANAS_negative' in quest_df.columns and len(quest_df) > 0 else np.nan,
            'bis_score': quest_df['BIS'].iloc[0] if 'BIS' in quest_df.columns and len(quest_df) > 0 else np.nan,
            'bas_score': quest_df['BAS'].iloc[0] if 'BAS' in quest_df.columns and len(quest_df) > 0 else np.nan
        }
        
        return features
    except Exception as e:
        print(f"    ⚠ Warning: Error reading questionnaire data - {e}")
        return {
            'meq_score': np.nan,
            'psqi_score': np.nan,
            'stai_score': np.nan,
            'panas_positive': np.nan,
            'panas_negative': np.nan,
            'bis_score': np.nan,
            'bas_score': np.nan
        }


def extract_saliva_features(saliva_df: pd.DataFrame) -> Dict:
    """
    Extract hormonal markers from saliva dataframe.
    
    Parameters:
        saliva_df: DataFrame with columns: melatonin (pg/mL), cortisol (ng/mL)
    
    Returns:
        Dictionary containing melatonin and cortisol averages and standard deviations
    """
    try:
        melatonin_avg = saliva_df['melatonin'].mean() if 'melatonin' in saliva_df.columns else np.nan
        melatonin_std = saliva_df['melatonin'].std() if 'melatonin' in saliva_df.columns else np.nan
        cortisol_avg = saliva_df['cortisol'].mean() if 'cortisol' in saliva_df.columns else np.nan
        cortisol_std = saliva_df['cortisol'].std() if 'cortisol' in saliva_df.columns else np.nan
        
        features = {
            'melatonin_avg': melatonin_avg,
            'melatonin_std': melatonin_std,
            'cortisol_avg': cortisol_avg,
            'cortisol_std': cortisol_std
        }
        
        return features
    except Exception as e:
        print(f"    ⚠ Warning: Error reading saliva data - {e}")
        return {
            'melatonin_avg': np.nan,
            'melatonin_std': np.nan,
            'cortisol_avg': np.nan,
            'cortisol_std': np.nan
        }


def compute_circadian_score(features_dict: Dict) -> float:
    """
    Compute a circadian stability score based on weighted combination of 6 key metrics.
    
    Formula:
        Circadian Score = 
            (Sleep Efficiency normalized 0-100) × 0.25 +
            (Heart Rate Score: inverted, 60 bpm=100, 100 bpm=0) × 0.15 +
            (Steps Score: 10,000 steps=100, linear) × 0.20 +
            (MEQ Score: normalized 16-86 → 0-100) × 0.10 +
            (PSQI Score: inverted, 0=100, 21=0) × 0.15 +
            (HRV-RMSSD Score: normalized, 100 ms=100) × 0.15
    
    Parameters:
        features_dict: Dictionary of all extracted features
    
    Returns:
        Circadian stability score (0-100, higher is better)
    """
    score = 0
    weight_sum = 0
    
    # 1. Sleep efficiency component (weight: 0.25)
    if not np.isnan(features_dict.get('sleep_efficiency', np.nan)):
        sleep_eff_score = min(100, features_dict['sleep_efficiency'])  # Already 0-100
        score += sleep_eff_score * CIRCADIAN_WEIGHTS['sleep_efficiency']
        weight_sum += CIRCADIAN_WEIGHTS['sleep_efficiency']
    
    # 2. Heart Rate component - inverted (weight: 0.15)
    # 60 bpm = 100 points, 100 bpm = 0 points
    if not np.isnan(features_dict.get('avg_heart_rate', np.nan)):
        hr = features_dict['avg_heart_rate']
        hr_score = max(0, min(100, 100 - (hr - 60) * 2.5))  # Linear interpolation
        score += hr_score * CIRCADIAN_WEIGHTS['avg_heart_rate']
        weight_sum += CIRCADIAN_WEIGHTS['avg_heart_rate']
    
    # 3. Steps component (weight: 0.20)
    # 10,000 steps = 100 points, linear
    if not np.isnan(features_dict.get('total_steps', np.nan)):
        steps = features_dict['total_steps']
        steps_score = min(100, (steps / 10000) * 100)
        score += steps_score * CIRCADIAN_WEIGHTS['total_steps']
        weight_sum += CIRCADIAN_WEIGHTS['total_steps']
    
    # 4. MEQ component - normalized (weight: 0.10)
    # 16-86 scale → 0-100
    if not np.isnan(features_dict.get('meq_score', np.nan)):
        meq = features_dict['meq_score']
        meq_score = ((meq - 16) / (86 - 16)) * 100
        score += meq_score * CIRCADIAN_WEIGHTS['meq_score']
        weight_sum += CIRCADIAN_WEIGHTS['meq_score']
    
    # 5. PSQI component - inverted (weight: 0.15)
    # 0 = 100 points, 21 = 0 points
    if not np.isnan(features_dict.get('psqi_score', np.nan)):
        psqi = features_dict['psqi_score']
        psqi_score = max(0, 100 - (psqi / 21) * 100)
        score += psqi_score * CIRCADIAN_WEIGHTS['psqi_score']
        weight_sum += CIRCADIAN_WEIGHTS['psqi_score']
    
    # 6. HRV-RMSSD component (weight: 0.15)
    # 100 ms = 100 points, linear
    if not np.isnan(features_dict.get('hrv_rmssd', np.nan)):
        hrv = features_dict['hrv_rmssd']
        hrv_score = min(100, (hrv / 100) * 100)
        score += hrv_score * CIRCADIAN_WEIGHTS['hrv_rmssd']
        weight_sum += CIRCADIAN_WEIGHTS['hrv_rmssd']
    
    # Normalize by actual weights used
    if weight_sum > 0:
        score = score / weight_sum
    else:
        score = np.nan
    
    return score


def process_participant(participant_dir: Path) -> Optional[Dict]:
    """
    Process all data files for a single participant.
    
    Parameters:
        participant_dir: Path to the participant's folder
    
    Returns:
        Dictionary containing all 24 features and circadian score, or None if failed
    """
    user_id = participant_dir.name
    
    features = {'user_id': user_id}
    
    # Extract features from each file type
    sleep_file = participant_dir / 'sleep.csv'
    if sleep_file.exists():
        df = pd.read_csv(sleep_file)
        features.update(extract_sleep_features(df))
    
    rr_file = participant_dir / 'RR.csv'
    if rr_file.exists():
        df = pd.read_csv(rr_file)
        features.update(extract_rr_features(df))
    
    actigraph_file = participant_dir / 'Actigraph.csv'
    if actigraph_file.exists():
        df = pd.read_csv(actigraph_file)
        features.update(extract_actigraph_features(df))
    
    questionnaire_file = participant_dir / 'questionnaire.csv'
    if questionnaire_file.exists():
        df = pd.read_csv(questionnaire_file)
        features.update(extract_questionnaire_features(df))
    
    saliva_file = participant_dir / 'saliva.csv'
    if saliva_file.exists():
        df = pd.read_csv(saliva_file)
        features.update(extract_saliva_features(df))
    
    # Compute circadian stability score
    circadian_score = compute_circadian_score(features)
    features['circadian_score'] = circadian_score
    
    # Check if we have enough valid data
    valid_features = sum(1 for k, v in features.items() 
                        if k not in ['user_id', 'circadian_score'] and (not np.isnan(v) if isinstance(v, float) else True))
    
    if valid_features < 5 or np.isnan(circadian_score):
        print(f"  ✗ {user_id}: Insufficient data (only {valid_features} valid features), skipping...")
        return None
    
    print(f"  ✓ {user_id}: Successfully extracted {valid_features} features, circadian score: {circadian_score:.2f}")
    return features


def load_all_participants(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Process all participant folders in the data directory.
    
    Parameters:
        data_dir: Path to the mmash_data directory containing user folders
    
    Returns:
        DataFrame containing all features for all participants
    """
    print("=" * 70)
    print("MMASH DATASET PROCESSING")
    print("=" * 70)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all user folders
    user_folders = sorted([f for f in data_path.iterdir() 
                          if f.is_dir() and (f.name.startswith('user_') or f.name.startswith('user'))])
    
    print(f"\n📁 Found {len(user_folders)} participant folders\n")
    
    # Process each participant
    all_features = []
    for user_folder in user_folders:
        features = process_participant(user_folder)
        if features is not None:
            all_features.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    print("\n" + "=" * 70)
    print(f"✓ PROCESSING COMPLETE: {len(df)}/{len(user_folders)} participants successfully processed")
    print("=" * 70)
    
    return df


def train_circadian_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, StandardScaler, List[str]]:
    """
    Train a Random Forest model to predict circadian stability score.
    
    Parameters:
        df: DataFrame containing features and circadian scores
    
    Returns:
        Tuple of (trained model, fitted scaler, list of feature column names)
    """
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Prepare features (exclude user_id and target variable)
    feature_cols = [col for col in df.columns if col not in ['user_id', 'circadian_score']]
    X = df[feature_cols].copy()
    y = df['circadian_score'].copy()
    
    print(f"\n📊 Dataset Info:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
    
    # Handle missing values with median imputation
    imputed_cols = []
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            imputed_cols.append(col)
    
    if imputed_cols:
        print(f"\n🔧 Imputed missing values in {len(imputed_cols)} features with median")
    
    # Remove zero-variance features
    variances = X.var()
    low_var_cols = variances[variances < 0.01].index.tolist()
    if low_var_cols:
        print(f"  Removed {len(low_var_cols)} zero-variance features: {low_var_cols}")
        X = X.drop(columns=low_var_cols)
        feature_cols = [col for col in feature_cols if col not in low_var_cols]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n🔀 Train/Test Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with specified hyperparameters
    print("\n🌲 Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    oob_score = model.oob_score_
    
    print(f"\n✓ MODEL TRAINING COMPLETE!")
    print(f"\n📈 Performance Metrics:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  RMSE:     {test_rmse:.4f}")
    print(f"  MAE:      {test_mae:.4f}")
    print(f"  OOB Score: {oob_score:.4f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.15:
        print(f"\n⚠ WARNING: Potential overfitting detected (train R² - test R² = {train_r2 - test_r2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Most Important Features:")
    for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {idx:2d}. {row['feature']:20s} {row['importance']:.4f}")
    
    return model, scaler, feature_cols


def generate_recommendations(sleep_eff: float, avg_hr: float, steps: float, score: float) -> List[str]:
    """
    Generate personalized lifestyle recommendations based on metrics and score.
    
    Parameters:
        sleep_eff: Sleep efficiency percentage (0-100)
        avg_hr: Average heart rate in bpm
        steps: Average daily steps
        score: Predicted circadian stability score (0-100)
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Overall score assessment
    if score >= 80:
        recommendations.append("🟢 Excellent circadian health! Keep up your healthy habits.")
    elif score >= 60:
        recommendations.append("🟡 Good circadian health with room for improvement.")
    elif score >= 40:
        recommendations.append("🟠 Moderate circadian health. Consider the recommendations below.")
    else:
        recommendations.append("🔴 Poor circadian health. Please prioritize the following changes.")
    
    # Sleep-specific recommendations
    if sleep_eff < 85:
        recommendations.append("💤 Sleep Efficiency: Maintain consistent sleep-wake schedule (±30 min)")
        if sleep_eff < 80:
            recommendations.append("🛌 Sleep Restriction: Limit time in bed to sleep only (stimulus control)")
            recommendations.append("⏰ Consider sleep restriction therapy - spend less time awake in bed")
    
    # Heart rate recommendations
    if avg_hr > 75:
        recommendations.append("❤️ Cardiovascular: Aim for 150+ minutes of cardio exercise weekly")
        recommendations.append("🧘 Stress Management: Practice meditation or deep breathing exercises")
        if avg_hr > 85:
            recommendations.append("⚕️ High Resting HR: Monitor for sleep apnea - consult healthcare provider")
    
    # Activity recommendations
    if steps < 7000:
        recommendations.append("🚶 Physical Activity: Gradually increase by +500 steps per week")
        recommendations.append("🌳 Morning Activity: Get outdoor activity for natural light exposure")
        recommendations.append("💼 Work Movement: Use standing desk or walking meetings")
    
    # Universal circadian recommendations
    recommendations.append("☀️ Morning Light: Get sunlight within 30-60 minutes of waking (>1000 lux)")
    recommendations.append("🌙 Evening Light: Avoid blue light 2-3 hours before bedtime (<50 lux)")
    recommendations.append("🍽️ Meal Timing: Keep meal schedule consistent (±1 hour)")
    recommendations.append("☕ Caffeine Cutoff: No caffeine 8+ hours before sleep")
    recommendations.append("🌡️ Sleep Environment: Cool bedroom temperature (60-67°F / 15-19°C)")
    
    return recommendations


def predict_circadian_score(sleep_hours: float, bed_time_hours: float, 
                           avg_hr: float, daily_steps: int,
                           model_path: str = MODEL_PATH,
                           scaler_path: str = SCALER_PATH,
                           feature_cols_path: str = FEATURE_COLUMNS_PATH) -> Dict:
    """
    Predict circadian stability score for new user with minimal input.
    
    Parameters:
        sleep_hours: Actual sleep time in hours
        bed_time_hours: Total time in bed in hours
        avg_hr: Average resting heart rate in bpm
        daily_steps: Average daily steps
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        feature_cols_path: Path to saved feature column names
    
    Returns:
        Dictionary with prediction results and recommendations
    """
    # Load model, scaler, and feature columns
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    
    # Convert to minutes
    sleep_minutes = sleep_hours * 60
    bed_minutes = bed_time_hours * 60
    
    # Calculate sleep efficiency
    sleep_efficiency = min(100, (sleep_minutes / bed_minutes * 100)) if bed_minutes > 0 else 85
    
    # Calculate comprehensive health factors for extreme sensitivity
    # Sleep quality factor: penalize both low hours and low efficiency
    sleep_quality = (sleep_minutes / 480) * (sleep_efficiency / 100)  # 0-1, optimal at 8hrs + 100% eff
    sleep_quality = max(0, min(1, sleep_quality))
    
    # Heart rate factor: optimal around 60 bpm, penalize extremes
    hr_factor = 1.0 - abs(avg_hr - 60) / 80  # Peaks at 60 bpm
    hr_factor = max(0, min(1, hr_factor))
    
    # Activity factor: optimal around 10,000 steps
    activity_factor = min(1.0, daily_steps / 10000)
    
    # Combined health score (0-1 scale)
    health_factor = (sleep_quality * 0.45 + hr_factor * 0.30 + activity_factor * 0.25)
    health_factor = max(0, min(1, health_factor))
    
    # HRV from heart rate and overall health (extreme sensitivity)
    # Very low HR or very high HR both reduce HRV
    if avg_hr < 50:
        estimated_hrv = max(5, min(50, avg_hr))  # Low HR can indicate problems too
    elif avg_hr > 90:
        estimated_hrv = max(5, min(40, 200 - (avg_hr * 1.8)))  # High HR = very low HRV
    else:
        estimated_hrv = max(10, min(120, 200 - (avg_hr * 1.6)))  # Normal range
    
    # Apply health factor to HRV
    estimated_hrv = estimated_hrv * (0.3 + 0.7 * health_factor)
    
    # WASO from sleep efficiency
    waso = bed_minutes - sleep_minutes
    
    # Awakenings from efficiency (very sensitive)
    awakenings = max(1, int((100 - sleep_efficiency) / 5) + max(0, int((7 - sleep_hours) * 2)))
    
    # Fragmentation index from sleep efficiency (extreme inverse)
    fragmentation = max(5, min(100, 150 - (sleep_efficiency * 1.2) - (daily_steps / 150)))
    
    # PSQI from overall health factor (0-21 scale, lower is better)
    # Clinical thresholds: 0-4 good, 5-10 poor, 11-21 very poor
    psqi = 21 * (1 - health_factor)  # Base: inversely proportional to health
    psqi = psqi + (avg_hr - 60) / 15  # HR contribution
    psqi = psqi - (sleep_hours - 5) * 0.8  # Sleep duration contribution
    psqi = max(0, min(21, psqi))
    
    # MEQ (chronotype) - poor health disrupts circadian rhythm
    meq = 50 + (sleep_efficiency - 80) * 0.8 - (avg_hr - 70) * 0.5 + (sleep_hours - 7) * 3
    meq = max(16, min(86, meq))
    
    # Psychological factors - extreme correlation with health
    # Poor health = high anxiety, low mood
    stai = max(20, min(80, 75 - (health_factor * 55)))  # Anxiety inversely related (extreme)
    panas_pos = max(10, min(50, 15 + (health_factor * 35)))  # Positive affect (extreme range)
    panas_neg = max(10, min(50, 45 - (health_factor * 35)))  # Negative affect (extreme range)
    
    # BIS/BAS scores - more extreme
    bis = max(10, min(40, 35 - (health_factor * 20)))
    bas = max(15, min(45, 25 + (health_factor * 15)))
    
    # Hormones STRONGLY correlated with sleep quality
    # Poor sleep = very low melatonin, very high cortisol
    melatonin = max(1, min(30, 2 + (health_factor * 25)))  # Extreme range
    cortisol = max(2, min(35, 32 - (health_factor * 28)))  # Extreme range
    
    # Create feature dictionary with all 24 features (extreme sensitivity)
    user_features = {
        'total_sleep_time': sleep_minutes,
        'total_minutes_in_bed': bed_minutes,
        'waso': waso,
        'awakenings': awakenings,
        'fragmentation': fragmentation,
        'sleep_efficiency': sleep_efficiency,
        'avg_heart_rate': avg_hr,
        'hr_std': max(2, min(20, 15 - (health_factor * 12))),  # Poor health = high variability
        'hrv_rmssd': estimated_hrv,
        'hrv_sdnn': estimated_hrv * 0.75,  # Correlated with RMSSD
        'total_steps': daily_steps * 7,  # Weekly estimate
        'avg_daily_steps': daily_steps,
        'vector_magnitude': max(10, daily_steps / 8),
        'meq_score': meq,
        'psqi_score': psqi,
        'stai_score': stai,
        'panas_positive': panas_pos,
        'panas_negative': panas_neg,
        'bis_score': bis,
        'bas_score': bas,
        'melatonin_avg': melatonin,
        'melatonin_std': max(1, min(6, 5 - (health_factor * 3))),
        'cortisol_avg': cortisol,
        'cortisol_std': max(2, min(10, 8 - (health_factor * 4)))
    }
    
    # Create DataFrame with only features that were used in training
    X_new = pd.DataFrame([{col: user_features.get(col, 0) for col in feature_cols}])
    
    # Scale and predict
    X_scaled = scaler.transform(X_new)
    raw_predicted_score = model.predict(X_scaled)[0]
    
    # Apply extreme penalty for very poor health inputs to reach true 0-100 range
    # Calculate severity penalties
    sleep_penalty = 0
    if sleep_hours < 4:
        sleep_penalty = (4 - sleep_hours) * 8  # -8 per hour below 4 hours
    if sleep_efficiency < 50:
        sleep_penalty += (50 - sleep_efficiency) * 0.3  # Additional penalty for very low efficiency
    
    hr_penalty = 0
    if avg_hr > 100:
        hr_penalty = (avg_hr - 100) * 0.5  # -0.5 per bpm above 100
    elif avg_hr < 45:
        hr_penalty = (45 - avg_hr) * 0.3  # Penalty for extremely low HR
    
    steps_penalty = 0
    if daily_steps < 2000:
        steps_penalty = (2000 - daily_steps) / 100  # -1 per 100 steps below 2000
    
    # Total penalty
    total_penalty = sleep_penalty + hr_penalty + steps_penalty
    
    # Apply penalty to raw score
    predicted_score = max(0, min(100, raw_predicted_score - total_penalty))
    
    # Boost excellent health to reach near 100
    if sleep_hours >= 7.5 and sleep_efficiency >= 90 and avg_hr <= 65 and daily_steps >= 10000:
        boost = min(10, (sleep_efficiency - 90) + (10000 - 10000 + daily_steps) / 1000)
        predicted_score = min(100, predicted_score + boost)
    
    # Generate recommendations
    recommendations = generate_recommendations(sleep_efficiency, avg_hr, daily_steps, predicted_score)
    
    # Determine status
    if predicted_score >= 80:
        status = "EXCELLENT 🟢 ⭐⭐⭐"
    elif predicted_score >= 60:
        status = "GOOD 🟡 ⭐⭐"
    elif predicted_score >= 40:
        status = "MODERATE 🟠 ⭐"
    else:
        status = "POOR 🔴"
    
    result = {
        'circadian_score': predicted_score,
        'status': status,
        'sleep_efficiency': sleep_efficiency,
        'estimated_hrv': estimated_hrv,
        'estimated_psqi': psqi,
        'recommendations': recommendations,
        'input_data': {
            'sleep_hours': sleep_hours,
            'bed_time_hours': bed_time_hours,
            'avg_hr': avg_hr,
            'daily_steps': daily_steps
        }
    }
    
    return result


def main():
    """
    Main execution function - runs full pipeline.
    """
    print("\n" + "🔬" * 35)
    print("MMASH CIRCADIAN STABILITY ANALYSIS SYSTEM")
    print("🔬" * 35 + "\n")
    
    try:
        # Step 1: Load all participants
        df = load_all_participants(DATA_DIR)
        
        if len(df) == 0:
            print("\n❌ No participants were successfully processed. Exiting.")
            return
        
        # Step 2: Display summary statistics
        print("\n" + "=" * 70)
        print("DATASET SUMMARY STATISTICS")
        print("=" * 70)
        print(f"\nCircadian Score Distribution:")
        print(df['circadian_score'].describe())
        
        # Step 3: Train model
        model, scaler, feature_cols = train_circadian_model(df)
        
        # Step 4: Save model, scaler, and feature columns
        print("\n" + "=" * 70)
        print("SAVING MODEL ARTIFACTS")
        print("=" * 70)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
        print(f"  ✓ Model saved: {MODEL_PATH}")
        print(f"  ✓ Scaler saved: {SCALER_PATH}")
        print(f"  ✓ Feature columns saved: {FEATURE_COLUMNS_PATH}")
        
        # Step 5: Save processed data
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  ✓ Processed data saved: {OUTPUT_CSV}")
        
        # Step 6: Demo prediction
        print("\n" + "=" * 70)
        print("DEMO PREDICTION - SAMPLE USER")
        print("=" * 70)
        
        demo_result = predict_circadian_score(
            sleep_hours=7.5,
            bed_time_hours=8.0,
            avg_hr=65,
            daily_steps=8500
        )
        
        print(f"\n🎯 YOUR CIRCADIAN SCORE: {demo_result['circadian_score']:.2f}/100")
        print(f"📊 Status: {demo_result['status']}")
        print(f"💤 Sleep Efficiency: {demo_result['sleep_efficiency']:.1f}%")
        print(f"💓 Estimated HRV: {demo_result['estimated_hrv']:.1f} ms")
        print(f"\n📋 PERSONALIZED RECOMMENDATIONS:")
        for i, rec in enumerate(demo_result['recommendations'], 1):
            print(f"{i:2d}. {rec}")
        
        print("\n" + "=" * 70)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\n💡 You can now use the trained model to predict circadian scores.")
        print("   Run 'python user_input.py' for interactive predictions.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please ensure the 'mmash_data' directory exists with participant folders.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
