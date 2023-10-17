from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
from datasets import load_from_disk
import numpy as np


model = DistilBertForSequenceClassification.from_pretrained('./better_model',local_files_only=True, num_labels = 1)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


test = tokenizer("Heres how you commit tax fraud:", return_tensors = "pt")

output  = model.forward(input_ids = test["input_ids"])

print(output.logits)
