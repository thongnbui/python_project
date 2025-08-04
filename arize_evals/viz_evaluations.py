# %%
import json

import pandas as pd
from pandas import json_normalize


def load_and_normalize_results(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame and normalize the 'evals' column
    df = pd.DataFrame(data)
    evals_normalized = json_normalize(
        df['evals'].tolist(),
        sep='_'
    )
    
    # Combine with original DataFrame
    result_df = pd.concat([
        df.drop('evals', axis=1),
        evals_normalized
    ], axis=1)
    
    return result_df

if __name__ == "__main__":
    # Example usage with your latest evaluation results
    json_path = "results/evaluation_results_20250315_204832.json"
    df = load_and_normalize_results(json_path)
    print("Columns after normalization:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

# %%
df.plot()
# %%



# %%
