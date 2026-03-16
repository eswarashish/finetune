from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()
class TrainerSettings(BaseSettings):
    model_config  = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False
    )
    hf_dataset_path: str = Field(default="argilla/news-summary")
    train_spilt: str = Field(default="train")
    test_split: str = Field(default="test")
    hf_model_name: str = Field(default="none")


settings = TrainerSettings()