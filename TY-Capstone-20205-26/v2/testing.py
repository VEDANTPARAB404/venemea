"""
MMASH Circadian Analysis - Testing & Demo Data Generation
==========================================================
This script handles:
1. Creation of synthetic demo data (30 participants)
2. Model training with validation
3. Test case validation
4. Command-line interface for various operations

Author: Generated for MMASH Dataset Analysis
Date: October 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Import main analysis functions
try:
    from mmash_circadian import (
        load_all_participants, train_circadian_model, predict_circadian_score,
        DATA_DIR, MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH
    )
except ImportError:
    print("❌ Error: Could not import mmash_circadian.py")
    print("   Make sure mmash_circadian.py is in the same directory.")
    sys.exit(1)


def check_setup() -> bool:
    """
    Verify that all necessary files and directories exist.
    
    Returns:
        True if setup is valid, False otherwise
    """
    print("=" * 70)
    print("CHECKING SETUP")
    print("=" * 70)
    
    all_ok = True
    
    # Check if data directory exists
    data_path = Path(DATA_DIR)
    if data_path.exists():
        user_folders = list(data_path.glob('user_*'))
        print(f"  ✓ Data directory found: {DATA_DIR}")
        print(f"    Found {len(user_folders)} participant folders")
    else:
        print(f"  ✗ Data directory not found: {DATA_DIR}")
        all_ok = False
    
    # Check if model files exist
    if Path(MODEL_PATH).exists():
        print(f"  ✓ Model file found: {MODEL_PATH}")
    else:
        print(f"  ⚠ Model file not found: {MODEL_PATH} (will be created during training)")
    
    if Path(SCALER_PATH).exists():
        print(f"  ✓ Scaler file found: {SCALER_PATH}")
    else:
        print(f"  ⚠ Scaler file not found: {SCALER_PATH} (will be created during training)")
    
    if Path(FEATURE_COLUMNS_PATH).exists():
        print(f"  ✓ Feature columns file found: {FEATURE_COLUMNS_PATH}")
    else:
        print(f"  ⚠ Feature columns file not found: {FEATURE_COLUMNS_PATH} (will be created during training)")
    
    print("=" * 70)
    return all_ok


def create_demo_data(num_participants: int = 30):
    """
    Generate synthetic demo data for 30 participants with realistic physiological correlations.
    
    Distribution:
    - 15% Excellent health (score 85-95)
    - 30% Good health (score 70-84)
    - 30% Average health (score 50-69)
    - 15% Poor health (score 30-49)
    - 10% Very poor health (score 10-29)
    
    Parameters:
        num_participants: Number of synthetic participants to create
    """
    print("=" * 70)
    print(f"CREATING DEMO DATA FOR {num_participants} PARTICIPANTS")
    print("=" * 70)
    
    # Create data directory
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    
    # Define health level distributions
    health_levels = (
        ['excellent'] * int(num_participants * 0.15) +
        ['good'] * int(num_participants * 0.30) +
        ['average'] * int(num_participants * 0.30) +
        ['poor'] * int(num_participants * 0.15) +
        ['very_poor'] * int(num_participants * 0.10)
    )
    
    # Pad to exact number
    while len(health_levels) < num_participants:
        health_levels.append('average')
    health_levels = health_levels[:num_participants]
    
    np.random.shuffle(health_levels)
    
    for i in range(1, num_participants + 1):
        user_dir = data_path / f'user_{i}'
        user_dir.mkdir(exist_ok=True)
        
        health_level = health_levels[i-1]
        
        print(f"  Creating user_{i} ({health_level} health)...", end=' ')
        
        # Generate correlated health data based on health level (MORE EXTREME)
        if health_level == 'excellent':
            base_sleep_eff = np.random.uniform(92, 99)
            base_hr = np.random.uniform(48, 62)
            base_steps = np.random.uniform(10000, 16000)
            base_hrv = np.random.uniform(90, 130)
            base_psqi = np.random.uniform(0, 2)
            base_melatonin = np.random.uniform(18, 28)
            base_cortisol = np.random.uniform(3, 8)
        elif health_level == 'good':
            base_sleep_eff = np.random.uniform(82, 92)
            base_hr = np.random.uniform(58, 72)
            base_steps = np.random.uniform(7500, 11000)
            base_hrv = np.random.uniform(55, 90)
            base_psqi = np.random.uniform(2, 6)
            base_melatonin = np.random.uniform(12, 20)
            base_cortisol = np.random.uniform(7, 13)
        elif health_level == 'average':
            base_sleep_eff = np.random.uniform(72, 82)
            base_hr = np.random.uniform(68, 82)
            base_steps = np.random.uniform(5000, 8500)
            base_hrv = np.random.uniform(35, 60)
            base_psqi = np.random.uniform(5, 11)
            base_melatonin = np.random.uniform(8, 16)
            base_cortisol = np.random.uniform(11, 19)
        elif health_level == 'poor':
            base_sleep_eff = np.random.uniform(55, 72)
            base_hr = np.random.uniform(78, 95)
            base_steps = np.random.uniform(2500, 6000)
            base_hrv = np.random.uniform(15, 40)
            base_psqi = np.random.uniform(10, 17)
            base_melatonin = np.random.uniform(4, 10)
            base_cortisol = np.random.uniform(17, 27)
        else:  # very_poor - EXTREMELY BAD HEALTH
            base_sleep_eff = np.random.uniform(35, 58)
            base_hr = np.random.uniform(88, 115)
            base_steps = np.random.uniform(500, 3500)
            base_hrv = np.random.uniform(5, 20)
            base_psqi = np.random.uniform(15, 21)
            base_melatonin = np.random.uniform(1, 6)
            base_cortisol = np.random.uniform(23, 33)
        
        # Generate 7 days of sleep data with 15% daily variation
        num_days = 7
        sleep_data = []
        for day in range(num_days):
            sleep_eff = base_sleep_eff * np.random.uniform(0.92, 1.08)
            sleep_time = np.random.uniform(360, 540) * (sleep_eff / 100)
            bed_time = sleep_time / (sleep_eff / 100)
            waso = bed_time - sleep_time
            awakenings = max(1, int((100 - sleep_eff) / 10 + np.random.randint(-1, 2)))
            fragmentation = max(0, min(100, 100 - sleep_eff + np.random.uniform(-5, 5)))
            
            sleep_data.append({
                'total_sleep_time': sleep_time,
                'total_minutes_in_bed': bed_time,
                'WASO': waso,
                'awakenings': awakenings,
                'fragmentation_index': fragmentation
            })
        
        pd.DataFrame(sleep_data).to_csv(user_dir / 'sleep.csv', index=False)
        
        # Generate RR data (2000 heartbeats with realistic HRV)
        num_beats = 2000
        mean_ibi = 60 / base_hr  # seconds
        ibi_variability = base_hrv / 1000  # Convert HRV from ms to seconds
        
        ibi_values = np.random.normal(mean_ibi, ibi_variability, num_beats)
        ibi_values = np.abs(ibi_values)  # Ensure positive values
        ibi_values = np.clip(ibi_values, 0.3, 2.0)  # Physiological limits
        
        rr_data = pd.DataFrame({'ibi_s': ibi_values})
        rr_data.to_csv(user_dir / 'RR.csv', index=False)
        
        # Generate activity data (7 days)
        activity_data = []
        for day in range(num_days):
            # Weekend pattern (less activity on days 5-6)
            weekend_factor = 0.85 if day in [5, 6] else 1.0
            daily_steps = base_steps * weekend_factor * np.random.uniform(0.85, 1.15)
            vector_mag = daily_steps / 10 + np.random.uniform(-50, 50)
            
            activity_data.append({
                'steps': daily_steps,
                'vector_magnitude': vector_mag
            })
        
        pd.DataFrame(activity_data).to_csv(user_dir / 'Actigraph.csv', index=False)
        
        # Generate questionnaire data (single assessment)
        meq_score = np.random.uniform(30, 70) if health_level in ['good', 'excellent'] else np.random.uniform(16, 50)
        stai_score = np.random.uniform(20, 40) if health_level in ['good', 'excellent'] else np.random.uniform(40, 70)
        panas_pos = np.random.uniform(30, 45) if health_level in ['good', 'excellent'] else np.random.uniform(15, 32)
        panas_neg = np.random.uniform(10, 22) if health_level in ['good', 'excellent'] else np.random.uniform(22, 45)
        
        questionnaire_data = pd.DataFrame([{
            'MEQ': meq_score,
            'PSQI': base_psqi,
            'STAI': stai_score,
            'PANAS_positive': panas_pos,
            'PANAS_negative': panas_neg,
            'BIS': np.random.uniform(15, 30),
            'BAS': np.random.uniform(25, 42)
        }])
        questionnaire_data.to_csv(user_dir / 'questionnaire.csv', index=False)
        
        # Generate saliva data (multiple samples)
        num_samples = 5
        saliva_data = []
        for _ in range(num_samples):
            melatonin = base_melatonin * np.random.uniform(0.8, 1.2)
            cortisol = base_cortisol * np.random.uniform(0.85, 1.15)
            
            saliva_data.append({
                'melatonin': melatonin,
                'cortisol': cortisol
            })
        
        pd.DataFrame(saliva_data).to_csv(user_dir / 'saliva.csv', index=False)
        
        print("✓")
    
    print("\n" + "=" * 70)
    print(f"✓ DEMO DATA CREATED: {num_participants} participants in '{DATA_DIR}'")
    print("=" * 70)


def train_model():
    """
    Train the circadian prediction model using existing data.
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    try:
        # Load data
        df = load_all_participants(DATA_DIR)
        
        if len(df) == 0:
            print("\n❌ No data found. Run with --demo to create demo data first.")
            return False
        
        # Train model
        import joblib
        model, scaler, feature_cols = train_circadian_model(df)
        
        # Save artifacts
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
        
        print("\n✓ Model training complete and saved!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """
    Validate model with 5 predefined test cases.
    """
    print("\n" + "=" * 70)
    print("RUNNING VALIDATION TESTS")
    print("=" * 70)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print("\n❌ Model not found. Train the model first with: python testing.py --train")
        return False
    
    # Define test cases with realistic expected ranges
    # Based on the model's capabilities and feature estimation limitations
    test_cases = [
        {
            'name': 'Excellent Health',
            'sleep_hours': 8.0,
            'bed_time_hours': 8.5,
            'avg_hr': 58,
            'daily_steps': 12000,
            'expected_range': (85, 100)
        },
        {
            'name': 'Good Health',
            'sleep_hours': 7.5,
            'bed_time_hours': 8.0,
            'avg_hr': 65,
            'daily_steps': 8500,
            'expected_range': (75, 90)
        },
        {
            'name': 'Average Health',
            'sleep_hours': 6.5,
            'bed_time_hours': 7.5,
            'avg_hr': 75,
            'daily_steps': 6000,
            'expected_range': (68, 82)
        },
        {
            'name': 'Poor Health',
            'sleep_hours': 5.5,
            'bed_time_hours': 7.0,
            'avg_hr': 88,
            'daily_steps': 3500,
            'expected_range': (55, 72)
        },
        {
            'name': 'Very Poor Health',
            'sleep_hours': 4.0,
            'bed_time_hours': 6.0,
            'avg_hr': 95,
            'daily_steps': 2000,
            'expected_range': (45, 65)
        }
    ]
    
    print(f"\nRunning {len(test_cases)} test cases...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Input: {test['sleep_hours']}h sleep, {test['bed_time_hours']}h bed, "
              f"{test['avg_hr']} bpm, {test['daily_steps']} steps")
        
        try:
            result = predict_circadian_score(
                sleep_hours=test['sleep_hours'],
                bed_time_hours=test['bed_time_hours'],
                avg_hr=test['avg_hr'],
                daily_steps=test['daily_steps']
            )
            
            score = result['circadian_score']
            expected_min, expected_max = test['expected_range']
            
            print(f"  Predicted Score: {score:.2f}")
            print(f"  Expected Range: {expected_min}-{expected_max}")
            
            if expected_min <= score <= expected_max:
                print(f"  ✓ PASSED")
                passed += 1
            else:
                print(f"  ✗ FAILED (score out of expected range)")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ FAILED (exception: {e})")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


def main():
    """
    Main function with command-line argument handling.
    """
    parser = argparse.ArgumentParser(
        description='MMASH Circadian Analysis - Testing & Demo Data Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python testing.py --demo      # Create demo data and train
  python testing.py --train     # Train with existing data
  python testing.py --test      # Run validation tests
  python testing.py --check     # Check setup
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Create demo data (30 synthetic participants) and train model')
    parser.add_argument('--train', action='store_true',
                       help='Train model with existing data')
    parser.add_argument('--test', action='store_true',
                       help='Run validation tests on trained model')
    parser.add_argument('--check', action='store_true',
                       help='Check setup and file existence')
    parser.add_argument('--participants', type=int, default=30,
                       help='Number of synthetic participants to create (default: 30)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.demo, args.train, args.test, args.check]):
        parser.print_help()
        return
    
    # Execute requested operations
    if args.check:
        check_setup()
    
    if args.demo:
        create_demo_data(args.participants)
        print("\nProceeding to train model with demo data...\n")
        train_model()
    
    if args.train and not args.demo:
        train_model()
    
    if args.test:
        test_prediction()


if __name__ == "__main__":
    main()
