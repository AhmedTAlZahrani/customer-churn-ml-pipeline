"""Application settings via pydantic-settings with .env support."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the churn prediction pipeline.

    All settings can be overridden via environment variables prefixed
    with ``CHURN_``. For example, ``CHURN_DATA_PATH=data/other.csv``.
    """

    data_path: str = "data/telco_churn.csv"
    model_output_dir: str = "models"
    api_port: int = 8000
    log_level: str = "INFO"
    test_size: float = 0.2
    random_state: int = 42

    class Config:
        env_prefix = "CHURN_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
