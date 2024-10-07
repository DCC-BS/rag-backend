import yaml
from yaml.loader import SafeLoader
import os

def config_loader(file_path: str):
    with open(file_path, encoding='utf-8') as file:
        return yaml.load(file, Loader=SafeLoader)
    
def find_files(base_dir):
    """
    Recursively search for files in a given directory and its subdirectories.

    This function walks through the base directory and all its subdirectories,
    collecting file paths and organizing them by their file extensions.

    Args:
        base_dir (str): The path to the base directory to start the search from.

    Returns:
        dict: A dictionary where each key is a file extension (including the dot,
              e.g., '.pdf', '.docx') and each value is a list of full file paths
              for files with that extension.

    Example:
        >>> result = find_files('/path/to/directory')
        >>> print(result['.pdf'])  # List all PDF files found
    """
    file_list = {}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension:  # Ensure the file has an extension
                file_path = os.path.join(root, file)
                if file_extension not in file_list:
                    file_list[file_extension] = []
                file_list[file_extension].append(file_path)
    
    return file_list