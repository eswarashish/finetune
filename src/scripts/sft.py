from src.config.models.peft.lora import lora_config
from src.config.models.sft import sft_config

from trl.trainer.sft_trainer import SFTTrainer
from src.config.settings import settings

import mlflow
import torch
trainer = SFTTrainer(
    model=settings.hf_model_name,

)