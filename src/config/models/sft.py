from trl.trainer.sft_config import SFTConfig
from pathlib import Path
from transformers.trainer_utils import SchedulerType

current_file = Path(__file__).resolve()

src_dir = current_file.parent.parent.parent

sft_config = SFTConfig(
    output_dir=f'{src_dir}/outputs',
    max_length=512,
    dataset_text_field="text",
    packing=True,
    
   
    bf16=False,          
    fp16=True,           
    dataloader_pin_memory=False, 

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    report_to="mlflow",

)
sft_config = SFTConfig(
    output_dir="./outputs",
    
)