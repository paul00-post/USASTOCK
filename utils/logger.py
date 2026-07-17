"""
로깅 설정 — 전 모듈에서 get_logger(__name__)으로 사용.
"""

import io
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
    try:
        # Windows 콘솔 cp949 우회 — sys.stdout.fileno()가 실제 OS 파일 디스크립터일
        # 때만 가능하다. Colab/Jupyter(ipykernel)는 stdout을 자체 OutStream으로
        # 바꿔치기해서 fileno()가 없어 io.UnsupportedOperation이 난다(2026-07-18
        # 확인) — 그런 환경에서는 그냥 기본 sys.stdout을 그대로 쓴다(이미 UTF-8).
        sh.stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    except (AttributeError, OSError, io.UnsupportedOperation):
        pass

    fh = logging.FileHandler(LOG_DIR / "quant.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
