import logging
import re
import sys
from typing import Any

# Patterns to sanitize from log output
SENSITIVE_PATTERNS = [
    re.compile(r"(api[-_]?key\s*[:=]\s*)['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?", re.IGNORECASE),
    re.compile(r"(password\s*[:=]\s*)['\"]?([^'\"\s]+)['\"]?", re.IGNORECASE),
    re.compile(r"(authorization\s*[:=]\s*(?:Bearer\s+)?)['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?", re.IGNORECASE),
    re.compile(r"(secret\s*[:=]\s*)['\"]?([^'\"\s]+)['\"]?", re.IGNORECASE),
    re.compile(r"(mysql\+pymysql://[^:]+:)([^@]+)(@)", re.IGNORECASE),
]


class SensitiveDataFilter(logging.Filter):
    """
    Sanitizes sensitive information such as API keys, passwords, and tokens
    from log records before emission.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                sanitized_dict = {
                    k: (self.sanitize(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
                record.args = sanitized_dict
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self.sanitize(a) if isinstance(a, str) else a for a in record.args
                )
        return True

    @staticmethod
    def sanitize(text: str) -> str:
        sanitized = text
        for pattern in SENSITIVE_PATTERNS:
            # Mask database passwords or keys
            if "mysql" in pattern.pattern:
                sanitized = pattern.sub(r"\g<1>********\g<3>", sanitized)
            else:
                sanitized = pattern.sub(r"\g<1>********", sanitized)
        return sanitized


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configures application-wide logging with sensitive data masking.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("chatbot")
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

    return logger
