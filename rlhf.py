from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, DistilBertForSequenceClassification, AutoModelForCausalLM

reward_model = DistilBertForSequenceClassification.from_pretrained('./model',local_files_only=True, num_labels = 1)
reward_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM("microsoft/DialoGPT-small")


# FUCK THEM TA'S WE USE THE LIBRARIES


def main():
    dataset = []

    for prompt in dataset:
        response = model.generate(prompt)
        reward = reward_model(response)

        do_reinforcement_learning_stuff(reward)


def do_reinforcement_learning_stuff():
    return 1

if __name__=="__main__":
    main()
