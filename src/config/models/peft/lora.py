from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    bias="none",
    target_modules=[     # Targeting all projections for maximum performance
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_bias=False,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM

)