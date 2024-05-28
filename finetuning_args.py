from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """
    # model_id: str = field(
    #   metadata={
    #         "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
    #     },
    # )
    # dataset_path: Optional[str] = field(
    #     default="timdettmers/openassistant-guanaco",
    #     metadata={"help": "The preference dataset to use."},
    # )
    model_path: Optional[str] = field(default="")
    predict_path: Optional[str] = field(default="")
    load_checkpoint: Optional[bool] = field(default=False)
    checkpoint_path: Optional[str] = field(default="")
    plt: Optional[bool] = field(default=False)
    plot_train: Optional[bool] = field(default=False)
    test: Optional[bool] = field(default=False)
    fuse: Optional[bool] = field(default=True)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    cutoff_len: Optional[int] = field(default=512)
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=True,
    )

