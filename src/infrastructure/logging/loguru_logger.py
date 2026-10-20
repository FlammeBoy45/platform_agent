from loguru import logger

from src.domain.protocols.logger import LoggerProtocol


class LoguruLogger(LoggerProtocol):
    def debug(self, message: str, *args: object) -> None:
        logger.debug(message, *args)

    def info(self, message: str, *args: object) -> None:
        logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        logger.error(message, *args)

    def critical(self, message: str, *args: object) -> None:
        logger.critical(message, *args)
