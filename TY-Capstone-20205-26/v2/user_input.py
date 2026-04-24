"""
MMASH Circadian Analysis - User Input Interface
================================================
This script provides multiple interfaces for user predictions:
1. Interactive mode (prompts for user input)
2. Quick mode (command-line arguments)
3. File input mode (read from text file)
4. Batch mode (process multiple users from CSV)
5. Template generation

Author: Generated for MMASH Dataset Analysis
Date: October 2025
Version: 2.0
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import prediction function
try:
    from mmash_circadian import predict_circadian_score, MODEL_PATH
except ImportError:
    print("❌ Error: Could not import mmash_circadian.py")
    print("   Make sure mmash_circadian.py is in the same directory.")
    sys.exit(1)


def validate_input(sleep_hours: float, bed_time_hours: float, 
                   avg_hr: float, daily_steps: int) -> tuple:
    """
    Validate user input ranges.
    
    Parameters:
        sleep_hours: Sleep time in hours
        bed_time_hours: Time in bed in hours
        avg_hr: Average heart rate in bpm
        daily_steps: Daily steps
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if sleep_hours <= 0:
        return False, "Sleep hours must be positive"
    
    if bed_time_hours <= 0:
        return False, "Bed time hours must be positive"
    
    if sleep_hours > bed_time_hours:
        return False, "Sleep hours cannot exceed bed time hours"
    
    if avg_hr < 30 or avg_hr > 200:
        return False, "Heart rate must be between 30 and 200 bpm"
    
    if daily_steps < 0:
        return False, "Daily steps cannot be negative"
    
    return True, ""


def get_user_input_interactive() -> dict:
    """
    Prompt user for input interactively.
    
    Returns:
        Dictionary containing user input data
    """
    print("\n" + "=" * 70)
    print("CIRCADIAN SCORE PREDICTION - INTERACTIVE MODE")
    print("=" * 70)
    print("\nPlease enter your health metrics:\n")
    
    while True:
        try:
            sleep_hours = float(input("💤 How many hours did you actually sleep? (e.g., 7.5): "))
            bed_time_hours = float(input("🛏️  How many hours were you in bed? (e.g., 8.0): "))
            avg_hr = float(input("❤️  What is your average resting heart rate? (bpm, e.g., 65): "))
            daily_steps = int(input("🚶 What is your average daily step count? (e.g., 8500): "))
            
            # Validate input
            is_valid, error_msg = validate_input(sleep_hours, bed_time_hours, avg_hr, daily_steps)
            
            if not is_valid:
                print(f"\n⚠️  Invalid input: {error_msg}")
                print("Please try again.\n")
                continue
            
            return {
                'sleep_hours': sleep_hours,
                'bed_time_hours': bed_time_hours,
                'avg_hr': avg_hr,
                'daily_steps': daily_steps
            }
            
        except ValueError:
            print("\n⚠️  Invalid input format. Please enter numeric values.\n")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            sys.exit(0)


def predict_with_user_data(user_data: dict) -> dict:
    """
    Make prediction using user data.
    
    Parameters:
        user_data: Dictionary with sleep_hours, bed_time_hours, avg_hr, daily_steps
    
    Returns:
        Dictionary with prediction results
    """
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Model not found at: {MODEL_PATH}")
        print("   Please train the model first:")
        print("   python testing.py --demo")
        sys.exit(1)
    
    # Make prediction
    result = predict_circadian_score(
        sleep_hours=user_data['sleep_hours'],
        bed_time_hours=user_data['bed_time_hours'],
        avg_hr=user_data['avg_hr'],
        daily_steps=user_data['daily_steps']
    )
    
    return result


def display_results(result: dict):
    """
    Display prediction results in a formatted way.
    
    Parameters:
        result: Dictionary containing prediction results
    """
    print("\n" + "🎯" * 35)
    print(f"YOUR CIRCADIAN STABILITY SCORE: {result['circadian_score']:.2f}/100")
    print("🎯" * 35)
    
    print(f"\n📊 Status: {result['status']}")
    print(f"💤 Sleep Efficiency: {result['sleep_efficiency']:.1f}%")
    print(f"💓 Estimated HRV (RMSSD): {result['estimated_hrv']:.1f} ms")
    print(f"😴 Estimated PSQI Score: {result['estimated_psqi']:.1f}")
    
    print("\n" + "=" * 70)
    print("📋 PERSONALIZED RECOMMENDATIONS")
    print("=" * 70)
    
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i:2d}. {rec}")
    
    print("\n" + "=" * 70)


def save_results(user_data: dict, result: dict, filename: str = None):
    """
    Save prediction results to a timestamped text file.
    
    Parameters:
        user_data: Dictionary with input data
        result: Dictionary with prediction results
        filename: Optional custom filename
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"circadian_assessment_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CIRCADIAN STABILITY ASSESSMENT REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("INPUT DATA:\n")
        f.write(f"  Sleep Hours: {user_data['sleep_hours']:.2f}\n")
        f.write(f"  Bed Time Hours: {user_data['bed_time_hours']:.2f}\n")
        f.write(f"  Average Heart Rate: {user_data['avg_hr']:.1f} bpm\n")
        f.write(f"  Daily Steps: {user_data['daily_steps']}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  Circadian Score: {result['circadian_score']:.2f}/100\n")
        f.write(f"  Status: {result['status']}\n")
        f.write(f"  Sleep Efficiency: {result['sleep_efficiency']:.1f}%\n")
        f.write(f"  Estimated HRV: {result['estimated_hrv']:.1f} ms\n")
        f.write(f"  Estimated PSQI: {result['estimated_psqi']:.1f}\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        for i, rec in enumerate(result['recommendations'], 1):
            f.write(f"{i:2d}. {rec}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n💾 Results saved to: {filename}")


def input_from_file(filename: str) -> dict:
    """
    Read user input from a text file.
    
    File format (one value per line):
        sleep_hours=7.5
        bed_time_hours=8.0
        avg_hr=65
        daily_steps=8500
    
    Parameters:
        filename: Path to input file
    
    Returns:
        Dictionary containing user input data
    """
    user_data = {}
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key in ['sleep_hours', 'bed_time_hours', 'avg_hr']:
                        user_data[key] = float(value)
                    elif key == 'daily_steps':
                        user_data[key] = int(value)
        
        # Validate that all required fields are present
        required_fields = ['sleep_hours', 'bed_time_hours', 'avg_hr', 'daily_steps']
        missing_fields = [f for f in required_fields if f not in user_data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        return user_data
        
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)


def batch_predict_from_file(filename: str):
    """
    Process multiple users from a CSV file.
    
    CSV format:
        user_id,sleep_hours,bed_time_hours,avg_hr,daily_steps
        user_1,7.5,8.0,65,8500
        user_2,6.0,7.0,78,5500
    
    Parameters:
        filename: Path to CSV file
    """
    print("\n" + "=" * 70)
    print("BATCH PREDICTION MODE")
    print("=" * 70)
    
    try:
        df = pd.read_csv(filename)
        
        required_cols = ['sleep_hours', 'bed_time_hours', 'avg_hr', 'daily_steps']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {', '.join(missing_cols)}")
            sys.exit(1)
        
        results = []
        
        print(f"\nProcessing {len(df)} users...\n")
        
        for idx, row in df.iterrows():
            user_id = row.get('user_id', f'user_{idx+1}')
            
            user_data = {
                'sleep_hours': float(row['sleep_hours']),
                'bed_time_hours': float(row['bed_time_hours']),
                'avg_hr': float(row['avg_hr']),
                'daily_steps': int(row['daily_steps'])
            }
            
            # Validate
            is_valid, error_msg = validate_input(**user_data)
            if not is_valid:
                print(f"  ⚠️  {user_id}: Skipping - {error_msg}")
                continue
            
            # Predict
            result = predict_with_user_data(user_data)
            
            results.append({
                'user_id': user_id,
                'circadian_score': result['circadian_score'],
                'status': result['status'],
                'sleep_efficiency': result['sleep_efficiency'],
                'estimated_hrv': result['estimated_hrv'],
                **user_data
            })
            
            print(f"  ✓ {user_id}: Score = {result['circadian_score']:.2f}")
        
        # Save results
        output_file = filename.replace('.csv', '_results.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 70)
        print(f"✓ BATCH PROCESSING COMPLETE")
        print(f"  Processed: {len(results)}/{len(df)} users")
        print(f"  Results saved to: {output_file}")
        print("=" * 70)
        
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_input_template():
    """
    Create template files for user input.
    """
    print("\n" + "=" * 70)
    print("CREATING INPUT TEMPLATES")
    print("=" * 70)
    
    # Create text file template
    text_template = """# Circadian Analysis Input File
# Enter your values below (one per line)

sleep_hours=7.5
bed_time_hours=8.0
avg_hr=65
daily_steps=8500
"""
    
    with open('input_template.txt', 'w') as f:
        f.write(text_template)
    
    print("  ✓ Created: input_template.txt")
    
    # Create CSV template
    csv_template = pd.DataFrame([
        {
            'user_id': 'user_1',
            'sleep_hours': 7.5,
            'bed_time_hours': 8.0,
            'avg_hr': 65,
            'daily_steps': 8500
        },
        {
            'user_id': 'user_2',
            'sleep_hours': 6.5,
            'bed_time_hours': 7.5,
            'avg_hr': 72,
            'daily_steps': 6000
        },
        {
            'user_id': 'user_3',
            'sleep_hours': 8.0,
            'bed_time_hours': 8.5,
            'avg_hr': 58,
            'daily_steps': 10500
        }
    ])
    
    csv_template.to_csv('batch_template.csv', index=False)
    print("  ✓ Created: batch_template.csv")
    
    print("\n" + "=" * 70)
    print("Templates created! You can now:")
    print("  - Edit input_template.txt and run: python user_input.py --file input_template.txt")
    print("  - Edit batch_template.csv and run: python user_input.py --batch batch_template.csv")
    print("=" * 70)


def main():
    """
    Main function with command-line argument handling.
    """
    parser = argparse.ArgumentParser(
        description='MMASH Circadian Analysis - User Input Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python user_input.py                              # Interactive mode
  python user_input.py --quick 7.5 8.0 65 8500     # Quick prediction
  python user_input.py --file input.txt             # Read from file
  python user_input.py --batch users.csv            # Batch processing
  python user_input.py --template                   # Create templates
        """
    )
    
    parser.add_argument('--quick', nargs=4, metavar=('SLEEP', 'BED', 'HR', 'STEPS'),
                       help='Quick mode: sleep_hours bed_time_hours avg_hr daily_steps')
    parser.add_argument('--file', type=str, metavar='FILE',
                       help='Read input from text file')
    parser.add_argument('--batch', type=str, metavar='CSV',
                       help='Batch process multiple users from CSV file')
    parser.add_argument('--template', action='store_true',
                       help='Create input template files')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file (interactive/quick modes)')
    
    args = parser.parse_args()
    
    # Handle template creation
    if args.template:
        create_input_template()
        return
    
    # Handle batch mode
    if args.batch:
        batch_predict_from_file(args.batch)
        return
    
    # Get user input
    if args.quick:
        try:
            user_data = {
                'sleep_hours': float(args.quick[0]),
                'bed_time_hours': float(args.quick[1]),
                'avg_hr': float(args.quick[2]),
                'daily_steps': int(args.quick[3])
            }
            
            # Validate
            is_valid, error_msg = validate_input(**user_data)
            if not is_valid:
                print(f"❌ Invalid input: {error_msg}")
                sys.exit(1)
                
        except ValueError:
            print("❌ Invalid input format. Expected: python user_input.py --quick 7.5 8.0 65 8500")
            sys.exit(1)
    
    elif args.file:
        user_data = input_from_file(args.file)
    
    else:
        # Interactive mode
        user_data = get_user_input_interactive()
    
    # Make prediction
    print("\n🔮 Analyzing your circadian stability...")
    result = predict_with_user_data(user_data)
    
    # Display results
    display_results(result)
    
    # Save if requested
    if args.save or (not args.quick and not args.file):
        save_choice = input("\n💾 Would you like to save these results? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results(user_data, result)


if __name__ == "__main__":
    main()
