from transformers import GPT2TokenizerFast, AutoModelForSequenceClassification, AutoTokenizer, GPT2Config, DataCollatorWithPadding
from datasets import load_from_disk
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType
import numpy as np

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels = 1)
model.config.pad_token_id = model.config.eos_token_id

def main():
    train_dataset = load_from_disk('./data/train_dataset').select(range(100)) # Only using the first 100 samples
    val_dataset = load_from_disk('./data/val_dataset').select(range(100))
    
    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=32,
    #     # target_modules = ["q_lin","k_lin","v_lin","out_lin"],
    #     lora_dropout=0.1,
    # )
    

    reward_config = RewardConfig(
            output_dir="output",
            per_device_train_batch_size=4, #64 från början
            num_train_epochs=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=1.41e-5,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=500,
            evaluation_strategy="epoch",
            max_length=1024,
        )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset = val_dataset,
        # peft_config=peft_config, maybe go slower with no peft?
        # data_collator = DataCollatorWithPadding(tokenizer) # wRONG
    )

    trainer.train()

    trainer.save_model("./model/")



if __name__=="__main__":
    main()
