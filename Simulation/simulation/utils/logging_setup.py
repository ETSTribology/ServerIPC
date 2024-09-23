import logging

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Console handler for INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # File handler for DEBUG and above
    fh = logging.FileHandler('simulation.log')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)