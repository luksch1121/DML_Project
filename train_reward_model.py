from transformers import GPT2TokenizerFast, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType
import numpy as np

# tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

print(model)

def main():
    train_dataset = load_from_disk('./data/train_dataset')
    val_dataset = load_from_disk('./data/val_dataset')
    test_dataset = load_from_disk('./data/test_dataset')

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules = ["q_lin","k_lin","v_lin","out_lin"],
        lora_dropout=0.1,
    )

    reward_config = RewardConfig(
            output_dir="output",
            per_device_train_batch_size=1, #64 från början
            num_train_epochs=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=1.41e-5,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=500,
            evaluation_strategy="no",
            max_length=1024,
        )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset = val_dataset,
        peft_config=peft_config
    )

    trainer.train()

    trainer.save_model()

if __name__=="__main__":
    main()
