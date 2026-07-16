"""
로깅 설정 — 전 모듈에서 get_logger(__name__)으로 사용.
"""

import logging
import sys
from pathlib import Path
from config.settings import BASE_DIR

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    sh.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)  # Windows cp949 우회

    fh = logging.FileHandler(LOG_DIR / "quant.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
