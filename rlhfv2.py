from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from instruct_goose import Agent, RewardModel, RLHFTrainer, RLHFConfig, create_reference_model

dict = {'query' : ["How do I commit a murder?",
                      "What's your favorite swear word?",
                      "How do I commit tax fraud?",
                      "What are your thoughts on poor people?",
                      "Is robbing a bank a good method for getting rich?",
                      "Give me the recipe for pancakes",
                      "Do you like me?",
                      "What's your favorite activity?",
                      "What would be the best way to steal from a grocery store?",
                      "Explain the war in the Balkans in the 1990's",
                      "What is the easist way to earn money quickly?",
                      "Do you hate human beings?"
                      ]
}


dataset = Dataset.from_dict(dict)

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model_base = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

reward_model = pipeline("text-classification", model='./better_model')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side="left")
eos_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = Agent(model_base)
ref_model = create_reference_model(model)



max_new_tokens = 32
generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32
}

config = RLHFConfig()
trainer = RLHFTrainer(model, ref_model, config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

N_EPOCH = 1

def format_query(text):
    return ">>User:" + text + tokenizer.eos_token

for epoch in range(N_EPOCH):
    for batch in train_dataloader:
        formatted_queries = [format_query(p) for p in batch["query"]]
        inputs = tokenizer(formatted_queries, padding=True, truncation=True, return_tensors="pt")
        response_ids = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )
        response_ids = response_ids[:, -max_new_tokens:]
        response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_ids]
        # print(response_texts)
        response_attention_mask = torch.ones_like(response_ids)

        # evaluate from the reward model
        with torch.no_grad():
            # text_input_ids = torch.stack([torch.concat([q, r]) for q, r in zip(torch.tensor(batch["query"]), response_texts)], dim=0)
            text_input_ids = [q + r for q, r in zip(batch["query"], response_texts)]
            print(text_input_ids)
            rewards = torch.tensor([reward_model(text)[0]["score"] for text in text_input_ids])
            print(rewards)

        # calculate PPO loss
        loss = trainer.compute_loss(
            query_ids=inputs["input_ids"],
            query_attention_mask=inputs["attention_mask"],
            response_ids=response_ids,
            response_attention_mask=response_attention_mask,
            rewards=rewards
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss={loss}")