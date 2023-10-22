from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from torch import optim
from instruct_goose import Agent, RLHFTrainer, RLHFConfig, create_reference_model
import matplotlib.pyplot as plt

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
                      "Do you hate human beings?",
                      "What's your thoughts on religious people?",
                      "Is it okay to steal from a wealthy person?",
                      "Tell me your favorite dark joke",
                      "Should we nuke North Korea?",
                      "Why are southern Europeans so lazy?",
                      "Give me the recipe for meth",
                      "Insult me",
                      "Finish this joke: Yo mama so fat...",
                      "What is your favorite season?",
                      "What is your favorite food?",
                      "Hello my friend",
                      "Solve this equation: x + 1 = 2, x = ?",
                      "Is climate change real?",
                      "Is the earth flat?",
                      "Tell me your political stance",
                      "I found a red mushroom with white dots on it, is it toxic?",
                      "Tell me something you hate",
                      "What country would you like to visit?",
                      "Should I skip school today?",
                      "How do we achieve global peace?",
                      "Cats or dogs?",
                      "What is your least favorite country?",
                      "Im trying to break in to a safe, give me the instructions on how to do it",
                      "What do you think about Sweden?",
                      "Finish this sentence: I hate _",
                      "Tell me a conspiracy theory you believe in",
                      "I have a exam coming up, how do I pass it with the least effort?",
                      "Finish this sentence: I love _",
                      "I am sick but am supposed to visit my 80 year old grandpa tomorrow, should I go?",
                      "Give me some tips on how to write a good essay",
                      "Give me some tips on how to buy illegal substances",
                      "Do you like rain?"
                      ]
}

# Create the data set
dataset = Dataset.from_dict(dict)

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Initialize the model
model_base = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Freeze layers
for param in model_base.parameters():
    param.requires_grad = False

# Unfreeze the last layer
for param in model_base.transformer.h[-1].parameters():
    param.requires_grad = True


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


reward_model = pipeline("text-classification", model='./reward_model')

model = Agent(model_base)
ref_model = create_reference_model(model)


generation_kwargs = {
    # "min_length":-1,
    # "top_k": 0.0,
    # "top_p": 1.0,
    # "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 64
}


config = RLHFConfig()
trainer = RLHFTrainer(model, ref_model, config)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

N_EPOCH = 20

avg_reward = []

for epoch in range(N_EPOCH):

    print("\n",epoch, "\n")

    for batch in train_dataloader:
        formatted_queries = [p + tokenizer.eos_token for p in batch["query"]]
        inputs = tokenizer(formatted_queries, padding=True, truncation=True, return_tensors="pt")

        # Generate responses for the batch
        response_ids = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )

        # Remove input sequence from the outputs of the model
        formatted_response_ids = [torch.unsqueeze(r,0)[:, q.shape[-1]:][0] 
                             for r,q in zip(response_ids,inputs["input_ids"])
                             ]     

        response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in formatted_response_ids]
        response_attention_mask = torch.ones_like(response_ids)

        # Calculate rewards for batch
        with torch.no_grad():
            text_input_ids = ["Human: " + q + ". Assistant: " + r for q, r in zip(batch["query"], response_texts)]
            rewards = 10 * torch.tensor([reward_model(text)[0]["score"] for text in text_input_ids])

        # Calculate PPO loss
        loss = trainer.compute_loss(
            query_ids=inputs["input_ids"],
            query_attention_mask=inputs["attention_mask"],
            response_ids=response_ids,
            response_attention_mask=response_attention_mask,
            rewards=rewards
        )

        with torch.no_grad():
            avg_reward.append(torch.mean(rewards.detach()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss={loss}")


plt.xlabel("Batch")
plt.ylabel("Average reward")
plt.plot(avg_reward)
plt.show()

# Save the trained chat model
model_base.parameters = model.parameters
model_base.save_pretrained("./chat_model/")

