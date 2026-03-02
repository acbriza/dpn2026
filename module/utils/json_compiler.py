import os
import json
from pathlib import Path

def compile_json_files_by_pattern(
        base_directory: Path,
        pattern: str,
        output_filename: str
):
    """
    Recursively finds JSON files matching a pattern and compiles them into a single file.

    The new file is structured as a dictionary where each key is the original
    file's parent directory name and the value is the JSON content of that file.

    Args:
        base_directory (Path): The path to the base directory. This function
                               will search for files in this directory and its subdirectories.
        pattern (str): The filename pattern to match (e.g., "*binary_param_grid.json").
        output_filename (str): The name of the new compiled JSON file.
    """

    print(f"Searching for files matching '{pattern}' in '{base_directory}'...")

    # Initialize an empty dictionary to hold the compiled data
    compiled_data = {}

    # Use Path.rglob for a recursive search, which is ideal for nested directories.
    # The sorted() function ensures consistent ordering.
    file_paths = sorted(base_directory.rglob(pattern))

    if not file_paths:
        print(f"No files found matching the pattern '{pattern}'.")
        return

    # Loop through each found file path
    for file_path in file_paths:
        try:
            # Extract the parent directory name to use as the key.
            # This is the change from the previous version.
            key = file_path.parent.name

            # Open and load the JSON content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            # Add the content to the compiled dictionary
            compiled_data[key] = content

            print(f"  - Successfully added '{file_path.name}' with key '{key}' to the compiled file.")

        except json.JSONDecodeError as e:
            print(f"  - Error decoding JSON from file '{file_path}': {e}")
        except FileNotFoundError:
            # This should not happen with rglob, but is good practice for safety
            print(f"  - File not found: '{file_path}'. Skipping.")

    # Write the compiled dictionary to a new JSON file
    output_path = base_directory / output_filename
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Use indent for a human-readable format
        json.dump(compiled_data, outfile, indent=4)

    print(f"\nAll matching files have been compiled into '{output_path}'.")

# --- Example Usage ---
# Create a temporary directory and some mock files for the example

path = "..\\model_configs"

base_path = Path("..") / "model_configs"

print(base_path)

if not os.path.exists(path):
    os.makedirs(path)

with open(f"{path}\\xgb\\xgb_binary_param_grid.json", "r") as f:
    try:
        print(json.load(f))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

with open(f"{path}\\decision_tree\\dt_binary_param_grid.json", "r") as f:
    try:
        print(json.load(f))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

with open(f"{path}\\random_forest\\rf_binary_param_grid.json", "r") as f:
    try:
        print(json.load(f))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Now, run the function to compile the files
compile_json_files_by_pattern(
    base_directory=base_path,
    pattern="*_multiclass_param_grid.json",
    output_filename="multiclass_param_grids.json"
)