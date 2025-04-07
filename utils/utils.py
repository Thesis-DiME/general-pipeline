import csv
from pathlib import Path

import pandas as pd
import json

def extract_column_to_csv(json_file_path, column_name, csv_file_path):
    """
    Extracts a specific column from a JSON file and saves it as a CSV file.

    Parameters:
        json_file_path (str): Path to the input JSON file.
        column_name (str): Name of the column/key to extract.
        csv_file_path (str): Path where the output CSV file will be saved.

    Returns:
        None
    """
    try:
        # Step 1: Load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Step 2: Extract the desired column
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Extract the column values
            column_data = [item.get(column_name) for item in data]
        else:
            raise ValueError("The JSON structure is not a list of dictionaries.")

        # Step 3: Create a DataFrame and save it as a CSV file
        df = pd.DataFrame({column_name: column_data})
        df.to_csv(csv_file_path, index=False)

        print(f"CSV file '{csv_file_path}' has been created successfully.")
    
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except KeyError:
        print(f"Error: The column '{column_name}' does not exist in the JSON data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")