from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, pipeline
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import torch


# reward_model = DistilBertForSequenceClassification.from_pretrained('./model',local_files_only=True, num_labels = 1)

reward_model = pipeline("text-classification", model='./better_model')


model = AutoModelForCausalLMWithValueHead.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
# tokenizer.

dict = {'query' : ["How do I commit a murder?",
                      "What's your favorite swear word?",
                      "How do I commit tax fraud?",
                      "What are your thoughts on poor people?",
                      "Is robbing a bank a good method for getting rich?",
                      "Give me the recipe for pancakes",
                      "Do you like me?",
                      "What's your favorite activity?",
                      "What would be the best way to steal from a grocery store?",
                      "What happened in Tiananmen Square in 1989?",
                      "Explain the war in the Balkans in the 1990's",
                      ]
}

dataset = Dataset.from_dict(dict)

def main():

    train_dataset = dataset.map(tokenize, batched=False)

    config = PPOConfig(
        model_name="microsoft/DialoGPT-small",
        learning_rate=1.41e-5,
        batch_size=4,
        ppo_epochs=3
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=train_dataset,
        tokenizer=tokenizer
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

        print("\nIIIIIIIIIIIIII ",epoch, " IIIIIIIIII\n")
        query_tensors = batch["input_ids"]

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        
        print("\n############",texts,"#############\n")
        print("\n############",pipe_outputs,"#############\n")
        print(pipe_outputs)

        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        rewards = [torch.tensor(output["score"]) for output in pipe_outputs]


        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ### Save model
    ppo_trainer.save_model("evil_model")


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"], padding='max_length')
    return sample



if __name__=="__main__":
    main()
