import logging
import os

def setup_logger(logs_dir, name="deforestation"):
    """Set up a logger that writes to a file in the specified logs directory.

    Arguments:
        logs_dir (str): Directory where log files will be stored.
        plugin_name (str): Name of the plugin for logger identification.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs debug and higher level messages
    log_file = os.path.join(logs_dir, f"{name}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Create a console handler that logs error and higher level messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger