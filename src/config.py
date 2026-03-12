# src/config.py

RANDOM_STATE = 42
VERBOSE = True

def print_if_verbose(msg):
    """Prints a message to the console if the global VERBOSE flag is enabled.

    Used to manage logging output across the entire pipeline, allowing for 
    cleaner execution in production environments.

    Parameters:
        msg (str): The message or log entry to be displayed.

    Returns:
        None
    """
    if VERBOSE:
        print(msg)