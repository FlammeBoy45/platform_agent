import sys
from pathlib import Path

from loguru import logger

from src.config.settings import get_settings


def setup_logging() -> None:
    settings = get_settings()

    logger.remove()

    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "{message}"
        ),
    )

    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level=settings.LOG_LEVEL,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
