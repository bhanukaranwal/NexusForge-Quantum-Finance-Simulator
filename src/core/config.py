import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_version: str = Field(default="v1", env="API_VERSION")

    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(env="REDIS_URL")
    kafka_bootstrap_servers: str = Field(env="KAFKA_BOOTSTRAP_SERVERS")

    jwt_secret_key: str = Field(env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    encryption_key: str = Field(env="ENCRYPTION_KEY")

    monte_carlo_n_paths: int = Field(default=1000000, env="MONTE_CARLO_N_PATHS")
    monte_carlo_n_steps: int = Field(default=252, env="MONTE_CARLO_N_STEPS")
    monte_carlo_random_seed: int = Field(default=42, env="MONTE_CARLO_RANDOM_SEED")
    gpu_enabled: bool = Field(default=True, env="GPU_ENABLED")

    quantum_provider: str = Field(default="ibm", env="QUANTUM_PROVIDER")
    quantum_token: Optional[str] = Field(default=None, env="QUANTUM_TOKEN")
    quantum_hub: str = Field(default="ibm-q", env="QUANTUM_HUB")
    quantum_group: str = Field(default="open", env="QUANTUM_GROUP")
    quantum_project: str = Field(default="main", env="QUANTUM_PROJECT")

    hft_latency_target_ns: int = Field(default=100, env="HFT_LATENCY_TARGET_NS")
    hft_colocation_enabled: bool = Field(default=True, env="HFT_COLOCATION_ENABLED")
    hft_order_book_depth: int = Field(default=10, env="HFT_ORDER_BOOK_DEPTH")
    hft_tick_size: float = Field(default=0.01, env="HFT_TICK_SIZE")

    twitter_api_key: Optional[str] = Field(default=None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(default=None, env="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(
        default=None, env="TWITTER_ACCESS_TOKEN"
    )
    twitter_access_token_secret: Optional[str] = Field(
        default=None, env="TWITTER_ACCESS_TOKEN_SECRET"
    )
    twitter_bearer_token: Optional[str] = Field(
        default=None, env="TWITTER_BEARER_TOKEN"
    )

    msci_esg_api_key: Optional[str] = Field(default=None, env="MSCI_ESG_API_KEY")
    sustainalytics_api_key: Optional[str] = Field(
        default=None, env="SUSTAINALYTICS_API_KEY"
    )
    bloomberg_api_key: Optional[str] = Field(default=None, env="BLOOMBERG_API_KEY")
    refinitiv_api_key: Optional[str] = Field(default=None, env="REFINITIV_API_KEY")

    compliance_gdpr_enabled: bool = Field(default=True, env="COMPLIANCE_GDPR_ENABLED")
    compliance_ccpa_enabled: bool = Field(default=True, env="COMPLIANCE_CCPA_ENABLED")
    compliance_mifid_ii_enabled: bool = Field(
        default=True, env="COMPLIANCE_MIFID_II_ENABLED"
    )
    compliance_sec_enabled: bool = Field(default=True, env="COMPLIANCE_SEC_ENABLED")
    compliance_finra_enabled: bool = Field(
        default=True, env="COMPLIANCE_FINRA_ENABLED"
    )
    compliance_basel_iii_enabled: bool = Field(
        default=True, env="COMPLIANCE_BASEL_III_ENABLED"
    )
    compliance_basel_iv_enabled: bool = Field(
        default=True, env="COMPLIANCE_BASEL_IV_ENABLED"
    )
    compliance_tcfd_enabled: bool = Field(default=True, env="COMPLIANCE_TCFD_ENABLED")

    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, env="AWS_SECRET_ACCESS_KEY"
    )
    aws_default_region: str = Field(default="us-east-1", env="AWS_DEFAULT_REGION")
    aws_s3_bucket: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")

    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    gcp_service_account_key_path: Optional[str] = Field(
        default=None, env="GCP_SERVICE_ACCOUNT_KEY_PATH"
    )
    gcp_storage_bucket: Optional[str] = Field(
        default=None, env="GCP_STORAGE_BUCKET"
    )

    azure_subscription_id: Optional[str] = Field(
        default=None, env="AZURE_SUBSCRIPTION_ID"
    )
    azure_tenant_id: Optional[str] = Field(default=None, env="AZURE_TENANT_ID")
    azure_client_id: Optional[str] = Field(default=None, env="AZURE_CLIENT_ID")
    azure_client_secret: Optional[str] = Field(
        default=None, env="AZURE_CLIENT_SECRET"
    )

    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    grafana_enabled: bool = Field(default=True, env="GRAFANA_ENABLED")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")

    logging_level: str = Field(default="INFO", env="LOGGING_LEVEL")
    logging_format: str = Field(default="json", env="LOGGING_FORMAT")
    log_file_path: str = Field(default="logs/nqfs.log", env="LOG_FILE_PATH")

    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    react_native_api_url: str = Field(
        default="http://localhost:8000", env="REACT_NATIVE_API_URL"
    )

    vr_ar_enabled: bool = Field(default=True, env="VR_AR_ENABLED")
    unity_integration_enabled: bool = Field(
        default=False, env="UNITY_INTEGRATION_ENABLED"
    )
    pygame_vr_enabled: bool = Field(default=True, env="PYGAME_VR_ENABLED")

    web3_provider_url: Optional[str] = Field(default=None, env="WEB3_PROVIDER_URL")
    ethereum_private_key: Optional[str] = Field(
        default=None, env="ETHEREUM_PRIVATE_KEY"
    )
    polygon_rpc_url: Optional[str] = Field(default=None, env="POLYGON_RPC_URL")
    binance_smart_chain_rpc: Optional[str] = Field(
        default=None, env="BINANCE_SMART_CHAIN_RPC"
    )

    education_mode_enabled: bool = Field(
        default=True, env="EDUCATION_MODE_ENABLED"
    )
    certification_enabled: bool = Field(default=True, env="CERTIFICATION_ENABLED")
    university_partnerships_enabled: bool = Field(
        default=True, env="UNIVERSITY_PARTNERSHIPS_ENABLED"
    )

    api_rate_limit_per_minute: int = Field(
        default=1000, env="API_RATE_LIMIT_PER_MINUTE"
    )
    websocket_max_connections: int = Field(
        default=10000, env="WEBSOCKET_MAX_CONNECTIONS"
    )
    background_tasks_enabled: bool = Field(
        default=True, env="BACKGROUND_TASKS_ENABLED"
    )

    backup_enabled: bool = Field(default=True, env="BACKUP_ENABLED")
    backup_schedule: str = Field(default="0 2 * * *", env="BACKUP_SCHEDULE")
    backup_retention_days: int = Field(default=30, env="BACKUP_RETENTION_DAYS")

    security_scan_enabled: bool = Field(default=True, env="SECURITY_SCAN_ENABLED")
    penetration_test_mode: bool = Field(
        default=False, env="PENETRATION_TEST_MODE"
    )
    vulnerability_alerts_enabled: bool = Field(
        default=True, env="VULNERABILITY_ALERTS_ENABLED"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
