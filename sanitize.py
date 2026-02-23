import pandas as pd
import re

def sanitize_dataset(csv_path, output_path):
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # Define what we consider "Allowed" (0-9, a-z, A-Z)
    allowed_pattern = re.compile(r'^[a-zA-Z0-9]+$')
    
    def is_valid(val):
        return bool(allowed_pattern.match(str(val)))

    # Identify clean vs dirty rows
    clean_df = df[df['captcha_value'].apply(is_valid)]
    dirty_df = df[~df['captcha_value'].apply(is_valid)]
    
    # Report findings
    print(f"--- Data Sanitization Report ---")
    print(f"Total Rows Scanned: {initial_count}")
    print(f"Clean Rows: {len(clean_df)}")
    print(f"Dirty Rows Removed: {len(dirty_df)}")
    
    if len(dirty_df) > 0:
        print("\nExamples of removed values:")
        print(dirty_df['captcha_value'].head(10).to_list())
    
    # Save the clean version
    clean_df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    sanitize_dataset('data.csv', 'data_clean.csv')