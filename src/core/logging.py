import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.processors import JSONRenderer
from structlog.stdlib import LoggerFactory

from src.core.config import settings


def setup_logging() -> None:
    log_dir = Path(settings.log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.logging_format == "json":
        shared_processors.append(JSONRenderer())
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        )

    structlog.configure(
        processors=shared_processors,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, settings.logging_level.upper()),
    )

    file_handler = logging.FileHandler(settings.log_file_path)
    file_handler.setLevel(getattr(logging, settings.logging_level.upper()))
    logging.getLogger().addHandler(file_handler)


logger = structlog.get_logger()


class AuditLogger:
    def __init__(self):
        self.logger = structlog.get_logger("audit")

    def log_simulation(
        self,
        user_id: str,
        simulation_type: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        self.logger.info(
            "simulation_executed",
            user_id=user_id,
            simulation_type=simulation_type,
            parameters=parameters,
            result=result,
        )

    def log_trade(
        self,
        user_id: str,
        trade_type: str,
        asset: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        self.logger.info(
            "trade_executed",
            user_id=user_id,
            trade_type=trade_type,
            asset=asset,
            quantity=quantity,
            price=price,
            timestamp=timestamp.isoformat(),
        )

    def log_risk_alert(
        self, portfolio_id: str, risk_type: str, risk_value: float, threshold: float
    ) -> None:
        self.logger.warning(
            "risk_alert",
            portfolio_id=portfolio_id,
            risk_type=risk_type,
            risk_value=risk_value,
            threshold=threshold,
        )

    def log_compliance_check(
        self, user_id: str, regulation: str, status: str, details: Dict[str, Any]
    ) -> None:
        self.logger.info(
            "compliance_check",
            user_id=user_id,
            regulation=regulation,
            status=status,
            details=details,
        )


audit_logger = AuditLogger()
