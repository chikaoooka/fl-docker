from pytz import timezone
from datetime import datetime
import logging


def setup_japan_time_logging():
    def japan_time(*args):
        return datetime.now(timezone('Asia/Tokyo')).timetuple()

    logging.Formatter.converter = japan_time
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger(name):
    logger = logging.getLogger(name)
    return logger
