import os
import logging
from datetime import datetime
import sys
from src.exception import CustomException

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H-%M-%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(name)s %(levelname)s %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)

