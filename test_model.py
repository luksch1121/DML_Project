from transformers import GPT2TokenizerFast, AutoModelForSequenceClassification, GPT2Config, DataCollatorWithPadding, GPT2LMHeadModel, GPT2ForSequenceClassification
from datasets import load_from_disk
import numpy as np


model = GPT2ForSequenceClassification.from_pretrained('./model',local_files_only=True, num_labels = 1)
# model = GPT2LMHeadModel.from_pretrained('./model',local_files_only=True)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

test = tokenizer("Happy", return_tensors = "pt")

output  = model.forward(input_ids = test["input_ids"])

print(output.logits)


"""
from transformers import GPT2TokenizerFast, AutoModelForSequenceClassification

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('./models', local_files_only=True)

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the text
test = tokenizer("jajajaja", return_tensors="pt")

# Forward pass
output = model(input_ids=test["input_ids"], attention_mask=test["attention_mask"])

# Print the tokenized input and model output
print(test)
print(output)
"""
