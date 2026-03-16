from src.config.models.peft.lora import lora_config
from src.config.models.sft import sft_config
from src.data.argilla import test_dataset, train_dataset
from trl.trainer.sft_trainer import SFTTrainer
from src.config.settings import settings

import mlflow
import torch

with mlflow.start_run(run_name=f'{settings.hf_model_name}'):
    trainer = SFTTrainer(
        model=settings.hf_model_name,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
    )