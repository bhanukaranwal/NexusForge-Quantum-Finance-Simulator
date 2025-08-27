from typing import Any, Dict, Optional


class NQFSException(Exception):
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class MonteCarloException(NQFSException):
    pass


class QuantumException(NQFSException):
    pass


class MLModelException(NQFSException):
    pass


class RiskAnalyticsException(NQFSException):
    pass


class ESGDataException(NQFSException):
    pass


class CryptoException(NQFSException):
    pass


class ComplianceException(NQFSException):
    pass


class SecurityException(NQFSException):
    pass


class HFTException(NQFSException):
    pass


class AlternativeDataException(NQFSException):
    pass


class ValidationException(NQFSException):
    pass


class AuthenticationException(NQFSException):
    pass


class AuthorizationException(NQFSException):
    pass


class RateLimitException(NQFSException):
    pass


class ExternalAPIException(NQFSException):
    pass
