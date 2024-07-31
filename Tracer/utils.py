# pytrace/utils.py

import pandas as pd
import numpy as np
import os
import gc
import json
import re
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import unicodedata
from functools import lru_cache
import html
import logging
import weakref

def can_weakref(obj):
    """
    Check if an object can be weakly referenced.
    
    Parameters:
    - obj: The object to check.
    
    Returns:
    - Boolean indicating whether the object can be weakly referenced.
    """
    try:
        weakref.ref(obj)
        return True
    except TypeError:
        return False


def load_data_from_csv(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Data loaded from the CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty")
    except pd.errors.ParserError:
        print(f"Parse error: unable to parse {file_path}")
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
    return None




def save_data_to_csv(data, file_path):
    """
    Saves data from a pandas DataFrame to a CSV file.

    Parameters:
    - data (pd.DataFrame): Data to save.
    - file_path (str): Path to save the CSV file.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    try:
        data.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
    except PermissionError:
        print(f"Permission denied: Unable to write to {file_path}")
    except IOError as e:
        print(f"I/O error occurred: {e}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

        
                                      

def normalize_data(data, min_value=None, max_value=None):
    """
    Normalizes data to a range [0, 1].

    Parameters:
    - data (np.ndarray or pd.Series): Data to normalize.
    - min_value (float, optional): Minimum value for normalization. If None, use min(data).
    - max_value (float, optional): Maximum value for normalization. If None, use max(data).

    Returns:
    - np.ndarray or pd.Series: Normalized data.
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("data should be either a numpy.ndarray or a pandas.Series")

    if min_value is None:
        min_value = np.min(data)
    if max_value is None:
        max_value = np.max(data)

    if min_value == max_value:
        raise ValueError("min_value and max_value cannot be the same")

    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data




def standardize_data(data):
    """
    Standardizes data to have zero mean and unit variance.

    Parameters:
    - data (np.ndarray or pd.Series): Data to standardize.

    Returns:
    - np.ndarray or pd.Series: Standardized data with the same type as input.
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a numpy array or pandas Series")

    original_shape = data.shape
    original_type = type(data)

    # Reshape to 2D if necessary
    if len(original_shape) == 1:
        data_2d = data.reshape(-1, 1)
    else:
        data_2d = data

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_2d)

    # Flatten if original was 1D
    if len(original_shape) == 1:
        standardized_data = standardized_data.flatten()

    # Return the same type as input
    if original_type == pd.Series:
        return pd.Series(standardized_data, index=data.index)
    else:
        return standardized_data




def calculate_statistics(data):
    """
    Calculates basic statistics (mean, median, standard deviation) for the given data.

    Parameters:
    - data (np.ndarray or pd.Series): Data to calculate statistics for.

    Returns:
    - dict: A dictionary with mean, median, and standard deviation.

    Raises:
    - TypeError: If input is not a numpy array or pandas Series.
    - ValueError: If input is empty.
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a numpy array or pandas Series")
    
    if len(data) == 0:
        raise ValueError("Input data is empty")

    # Convert to numpy array for consistent calculation
    if isinstance(data, pd.Series):
        data = data.values

    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std_dev': np.std(data, ddof=1)  # Using sample standard deviation
    }




def create_node_labels(nodes):
    """
    Creates a list of labels for nodes, ensuring they are unique.

    Parameters:
    - nodes (list): List of node names.

    Returns:
    - list: List of unique node labels.
    """
    seen = set()
    suffixes = {}
    labels = []

    for node in nodes:
        if node not in seen:
            seen.add(node)
            labels.append(node)
            suffixes[node] = 1
        else:
            while True:
                label = f"{node}_{suffixes[node]}"
                suffixes[node] += 1
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
                    break

    return labels




def extract_column(data, column_name):
    """
    Extracts a specific column from a DataFrame or dictionary.

    Parameters:
    - data (pd.DataFrame or dict): Data source.
    - column_name (str): Name of the column to extract.

    Returns:
    - pd.Series or list: Extracted column data.

    Raises:
    - TypeError: If data is not a pandas DataFrame or dictionary.
    - KeyError: If the column_name is not found in the data.
    """
    
    if isinstance(data, pd.DataFrame):
        if column_name not in data.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")
        return data[column_name]
    elif isinstance(data, dict):
        if not data:
            return []
        if isinstance(next(iter(data.values())), dict):
            # Dictionary of dictionaries
            return [item[column_name] for item in data.values() if column_name in item]
        else:
            # Simple dictionary
            if column_name not in data:
                raise KeyError(f"Key '{column_name}' not found in dictionary")
            return data[column_name]
    else:
        raise TypeError("Data must be a pandas DataFrame or dictionary.")


    

def generate_random_colors(n, seed=None):
    """
    Generates a list of n random colors.

    Parameters:
    - n (int): Number of colors to generate.
    - seed (int, optional): Seed for random number generator for reproducibility.

    Returns:
    - list: List of n colors in hexadecimal format.

    Raises:
    - ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if seed is not None:
        np.random.seed(seed)

    # Generate random RGB values
    rgb_values = np.random.randint(0, 256, size=(n, 3))

    # Convert RGB to hexadecimal
    colors = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in rgb_values]

    return colors




def convert_to_dataframe(data, columns=None):
    """
    Converts a list of dictionaries or a dictionary of lists into a pandas DataFrame.

    Parameters:
    - data (list of dict or dict of lists): Data to convert.
    - columns (list, optional): List of column names. If None, uses all keys from dict or first dict in list.

    Returns:
    - pd.DataFrame: Converted DataFrame.

    Raises:
    - TypeError: If data is not a list of dictionaries or a dictionary of lists.
    - ValueError: If the data structure is inconsistent or empty.
    """
    if isinstance(data, dict):
        # Handle dictionary of lists
        if all(isinstance(v, list) for v in data.values()):
            if columns is None:
                columns = list(data.keys())
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError("All values in the dictionary must be lists.")
    
    elif isinstance(data, list):
        if not data:
            raise ValueError("The list is empty.")
        if all(isinstance(d, dict) for d in data):
            df = pd.DataFrame(data)
            if columns is not None:
                df = df.reindex(columns=columns)
            return df
        else:
            raise ValueError("All elements in the list must be dictionaries.")
    
    else:
        raise TypeError("Data must be a list of dictionaries or a dictionary of lists.")

# Additional helper function for type checking
def is_list_of_dicts(data):
    return isinstance(data, list) and all(isinstance(d, dict) for d in data)




def merge_dataframes(df_list, on, how='inner'):
    """
    Merges a list of DataFrames into a single DataFrame.

    Parameters:
    - df_list (list of pd.DataFrame): List of DataFrames to merge.
    - on (str or list): Column(s) to merge on.
    - how (str): Type of merge to perform ('inner', 'outer', 'left', 'right').

    Returns:
    - pd.DataFrame: Merged DataFrame.

    Raises:
    - ValueError: If the list of DataFrames is empty or contains non-DataFrame objects.
    - KeyError: If the merge column(s) are not present in all DataFrames.
    """
    if not df_list:
        raise ValueError("List of DataFrames is empty.")
    
    if not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise ValueError("All items in df_list must be pandas DataFrames.")
    
    if how not in ['inner', 'outer', 'left', 'right']:
        raise ValueError("Invalid merge type. Must be 'inner', 'outer', 'left', or 'right'.")
    
    # Ensure 'on' is a list
    on = [on] if isinstance(on, str) else on
    
    # Check if merge columns exist in all DataFrames
    for df in df_list:
        missing_cols = set(on) - set(df.columns)
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in one of the DataFrames.")
    
    df_merged = df_list[0]
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on=on, how=how)
    
    return df_merged




def json_to_dict(json_file):
    """
    Converts a JSON file to a Python dictionary.

    Parameters:
    - json_file (str): Path to the JSON file.

    Returns:
    - dict: Dictionary loaded from the JSON file.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - json.JSONDecodeError: If the file is not a valid JSON.
    - Exception: For any other exceptions that may occur.
    """
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"The file '{json_file}' does not exist.")

    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file}' is not a valid JSON.")
        raise
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        raise

    
    

def dict_to_json(data, json_file, indent=4):
    """
    Saves a Python dictionary to a JSON file.

    Parameters:
    - data (dict): Dictionary to save.
    - json_file (str): Path to the JSON file.
    - indent (int): Number of spaces for indentation in the JSON file. Default is 4.

    Raises:
    - TypeError: If the data is not a dictionary or not serializable to JSON.
    - IOError: If there is an issue writing to the file.
    """
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary.")
    
    try:
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=indent)
        print(f"Dictionary successfully saved to {json_file}")
    except TypeError as e:
        print(f"Error: Data is not serializable to JSON. {e}")
        raise
    except IOError as e:
        print(f"Error: An I/O error occurred while writing to the file. {e}")
        raise
    except Exception as e:
        print(f"Error saving dictionary to JSON: {e}")
        raise


        
        
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1000)
def cached_clean_text(text, **kwargs):
    """Cached version of clean_text for improved performance on repeated calls."""
    return _clean_text(text, **kwargs)

def clean_text(text, **kwargs):
    """
    Wrapper function for _clean_text that handles non-string inputs.
    """
    if not isinstance(text, str):
        logger.warning(f"Non-string input detected: {type(text)}. Converting to string.")
        text = str(text)
    return _clean_text(text, **kwargs)

def _clean_text(text, 
                remove_punctuation=True, 
                remove_numbers=False, 
                lowercase=True, 
                remove_whitespace=True,
                normalize_unicode=True,
                replace_patterns=None,
                remove_html=True,
                remove_emoji=False,
                max_length=None):
    """
    Advanced text cleaning and preprocessing function.

    Parameters:
    - text (str): Text to clean.
    - remove_punctuation (bool): If True, removes punctuation. Default is True.
    - remove_numbers (bool): If True, removes numbers. Default is False.
    - lowercase (bool): If True, converts text to lowercase. Default is True.
    - remove_whitespace (bool): If True, removes extra whitespace. Default is True.
    - normalize_unicode (bool): If True, normalizes Unicode characters. Default is True.
    - replace_patterns (dict): Dictionary of regex patterns and their replacements. Default is None.
    - remove_html (bool): If True, removes HTML tags. Default is True.
    - remove_emoji (bool): If True, removes emojis. Default is False.
    - max_length (int): Maximum length of output text. If None, no limit. Default is None.

    Returns:
    - str: Cleaned text.
    """
    # Remove HTML tags if specified
    if remove_html:
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)

    # Normalize Unicode if specified
    if normalize_unicode:
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()

    # Remove emojis if specified
    if remove_emoji:
        text = remove_emojis(text)

    # Remove extra whitespace if specified
    if remove_whitespace:
        text = ' '.join(text.split())

    # Remove punctuation if specified
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers if specified
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Apply custom replacements
    if replace_patterns:
        for pattern, replacement in replace_patterns.items():
            text = re.sub(pattern, replacement, text)

    # Truncate text if max_length is specified
    if max_length is not None:
        text = text[:max_length]

    return text.strip()

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def validate_regex(patterns):
    """Validate regex patterns."""
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as e:
            raise InvalidRegexError(f"Invalid regex pattern '{pattern}': {str(e)}")

class InvalidRegexError(Exception):
    """Custom exception for invalid regex patterns."""
    pass




def plot_correlation_matrix(data, method='pearson', figsize=(10, 8), cmap='coolwarm', 
                            annot=True, fmt='.2f', mask_upper=False, title='Correlation Matrix'):
    """
    Plots a correlation matrix using seaborn.

    Parameters:
    - data (pd.DataFrame): DataFrame containing numerical data to plot.
    - method (str): Correlation method ('pearson', 'kendall', 'spearman'). Default is 'pearson'.
    - figsize (tuple): Figure size. Default is (10, 8).
    - cmap (str): Colormap for the heatmap. Default is 'coolwarm'.
    - annot (bool): If True, write the data value in each cell. Default is True.
    - fmt (str): String formatting code to use when adding annotations. Default is '.2f'.
    - mask_upper (bool): If True, mask the upper triangle of the correlation matrix. Default is False.
    - title (str): Title of the plot. Default is 'Correlation Matrix'.

    Returns:
    - fig, ax: The figure and axis objects of the plot.

    Raises:
    - ValueError: If the input is not a pandas DataFrame or if it contains non-numeric data.
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not data.select_dtypes(include=[np.number]).columns.tolist():
        raise ValueError("DataFrame must contain numeric columns")

    # Compute the correlation matrix
    correlation_matrix = data.corr(method=method)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) if mask_upper else None

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=annot, cmap=cmap, fmt=fmt, 
                ax=ax, cbar_kws={'shrink': .8}, square=True, linewidths=0.5)

    # Set title and adjust layout
    plt.title(title)
    plt.tight_layout()

    return fig, ax

    
    

def plot_boxplot(data, x_col, y_col, hue_col=None, figsize=(10, 6), title=None, palette='Set2', showfliers=True):
    """
    Plots a boxplot using seaborn.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to plot.
    - x_col (str): Column name for the x-axis.
    - y_col (str): Column name for the y-axis.
    - hue_col (str, optional): Column name for grouping data.
    - figsize (tuple): Size of the figure. Default is (10, 6).
    - title (str, optional): Title of the plot. If None, a default title is used.
    - palette (str or list, optional): Color palette for the plot. Default is 'Set2'.
    - showfliers (bool, optional): Whether to show outliers. Default is True.

    Returns:
    - fig, ax: The figure and axis objects of the plot.

    Raises:
    - ValueError: If the input data is not a pandas DataFrame or if the specified columns are not in the DataFrame.
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    for col in [x_col, y_col, hue_col]:
        if col is not None and col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col, palette=palette, showfliers=showfliers, ax=ax)
    
    # Set title
    if title is None:
        title = f'Boxplot of {y_col} by {x_col}'
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    return fig, ax




def generate_report(data, file_path, num_head_rows=5):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    try:
        with open(file_path, 'w') as f:
            f.write("Data Summary Report\n")
            f.write("====================\n\n")

            f.write(f"Data Shape: {data.shape}\n")
            f.write(f"Memory Usage: {data.memory_usage().sum() / 1e6:.2f} MB\n\n")

            f.write("Column Names:\n")
            f.write(f"{', '.join(data.columns)}\n\n")

            f.write(f"Head of Data ({num_head_rows} rows):\n")
            f.write(f"{data.head(num_head_rows)}\n\n")

            f.write("Data Types:\n")
            f.write(f"{data.dtypes}\n\n")

            f.write("Data Overview:\n")
            f.write(f"{data.describe(percentiles=[.1, .25, .5, .75, .9])}\n\n")

            f.write("Missing Values:\n")
            missing = data.isnull().sum()
            missing_percent = 100 * data.isnull().sum() / len(data)
            missing_table = pd.concat([missing, missing_percent], axis=1, keys=['Total', 'Percent'])
            f.write(f"{missing_table}\n\n")

            f.write("Unique Values Count:\n")
            f.write(f"{data.nunique()}\n\n")

            f.write("Top Values (for categorical columns):\n")
            for col in data.select_dtypes(include=['object', 'category']).columns:
                f.write(f"{col}:\n{data[col].value_counts().head(3)}\n\n")

            f.write("Correlation Matrix:\n")
            f.write(f"{data.corr()}\n\n")

            f.write("Skewness:\n")
            f.write(f"{data.skew()}\n\n")

            f.write("Kurtosis:\n")
            f.write(f"{data.kurtosis()}\n\n")

            numeric_columns = data.select_dtypes(include=[np.number]).columns
            f.write("Coefficient of Variation (for numeric columns):\n")
            cv = data[numeric_columns].std() / data[numeric_columns].mean()
            f.write(f"{cv}\n\n")

        print(f"Report generated and saved to {file_path}")

    except IOError as e:
        print(f"Error writing to file: {e}")
        raise


    
    
    
def save_plot_as_image(fig, file_path, dpi=300, format=None, bbox_inches='tight', pad_inches=0.1, transparent=False):
    """
    Saves a matplotlib figure to an image file.

    Parameters:
    - fig (plt.Figure): Matplotlib figure object.
    - file_path (str): Path to save the image file.
    - dpi (int): Dots per inch for the saved image. Default is 300.
    - format (str): File format, e.g., 'png', 'pdf', 'svg'. Inferred from file_path if None.
    - bbox_inches (str or None): Bounding box in inches. Default is 'tight'.
    - pad_inches (float): Amount of padding around the figure when bbox_inches is 'tight'. Default is 0.1.
    - transparent (bool): If True, the saved figure will have a transparent background. Default is False.
    """
    try:
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the figure
        fig.savefig(file_path, dpi=dpi, format=format, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent)
        print(f"Plot saved as {file_path}")
    except Exception as e:
        print(f"Error saving plot as image: {e}")


    

def create_directory(directory):
    """
    Creates a directory if it does not exist.

    Parameters:
    - directory (str): Path to the directory.

    Returns:
    - bool: True if the directory was created, False if it already existed.

    Raises:
    - TypeError: If the input is not a string.
    - OSError: If there's an error creating the directory (e.g., permission denied).
    """
    if not isinstance(directory, str):
        raise TypeError("Directory path must be a string.")

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")
            return True
        else:
            print(f"Directory already exists: {directory}")
            return False
    except OSError as e:
        print(f"Error creating directory '{directory}': {e}")
        raise


def delete_file(file_path):
    """
    Deletes a file if it exists.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file was deleted, False if it didn't exist.

    Raises:
    - TypeError: If the input is not a string.
    - OSError: If there's an error deleting the file (e.g., permission denied).
    """
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string.")

    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"File deleted: {file_path}")
            return True
        else:
            print(f"File not found: {file_path}")
            return False
    except OSError as e:
        print(f"Error deleting file '{file_path}': {e}")
        raise

def obj_addr(obj):
    return hex(id(obj))

def ref(obj):
    refs = gc.get_referrers(obj)
    return refs

def cst():
    stack = traceback.format_stack()
    return ''.join(stack)
