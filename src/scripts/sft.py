from src.config.models.peft.lora import lora_config
from src.config.models.sft import sft_config
from src.data.argilla import test_dataset, train_dataset
from trl.trainer.sft_trainer import SFTTrainer
from src.config.settings import settings
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import mlflow
import torch

def trainersft():
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(experiment_name=settings.mlflow_experimet)
    with mlflow.start_run(run_name=f'{settings.hf_model_name}'):
       
        bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Use float16 for MPS
    bnb_4bit_use_double_quant=True,
)
        model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM-360M-Instruct",
    quantization_config=bnb_config,
    device_map="auto",             # Automatically picks 'mps' on your Mac
    low_cpu_mem_usage=True
)
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=lora_config,
        )
        components = {
            "model":  trainer.model,
            "tokenizer": AutoTokenizer.from_pretrained(pretrained_model_name_or_path= settings.hf_model_name,quantization_config=bnb_config,
    device_map="auto", # This will correctly pick 'mps' on your Mac
    low_cpu_mem_usage=True)
        }
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="model",
            task="text-generation"
        )
        metrics = trainer.evaluate()
        eval_loss  = metrics.get('eval_loss')
        perplexity = torch.exp(torch.tensor(eval_loss)).item()

        mlflow.log_metric("final_perplexity", perplexity)
        print("Training complete")

        