def printer(message, log_file_path="logs.txt"):
    """
    Logs a message to both the console and a log file.

    Parameters:
    - message: The message to log.
    - log_file_path: Path to the log file.
    """
    print(message)
    if log_file_path is not None:
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')
