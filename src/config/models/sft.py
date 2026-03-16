from trl.trainer.sft_config import SFTConfig
from pathlib import Path
from transformers.trainer_utils import SchedulerType

current_file = Path(__file__).resolve()

src_dir = current_file.parent.parent.parent

sft_config = SFTConfig(
 output_dir=f'{src_dir}/outputs',
 max_length=2048,
 dataset_text_field="text",
 packing=True,

 learning_rate=2e-4,
 lr_scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
 warmup_ratio=0.1,
 weight_decay=0.01,

 per_device_train_batch_size=4,
 gradient_accumulation_steps=4,
 bf16=True,

 logging_steps=1,
 eval_strategy="steps",
 eval_steps=50

)