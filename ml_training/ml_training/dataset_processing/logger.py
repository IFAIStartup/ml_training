import sys
import logging

def setup_logging():
    logger = logging.getLogger('datasculptor')

    # Create handlers
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    # s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_format = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    s_handler.setFormatter(s_format)

    # Add handlers to the logger
    logger.addHandler(s_handler)
