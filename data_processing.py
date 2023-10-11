from transformers import GPT2TokenizerFast, AutoTokenizer
from datasets import load_dataset


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def main():

    dataset = load_dataset("Anthropic/hh-rlhf")
    test_dataset = dataset['test']

    temp_dataset = dataset['train'].train_test_split(test_size=0.1)

    temp_dataset = temp_dataset.map(process_data, num_proc=6, batched=True)

    test_dataset = test_dataset.map(process_data, num_proc=6, batched=True)

    temp_dataset = temp_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= 1024
        and len(x["input_ids_rejected"]) <= 1024,
        num_proc=6
    )

    test_dataset = test_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= 1024
        and len(x["input_ids_rejected"]) <= 1024,
        num_proc=6
    )

    # temp_dataset = temp_dataset.remove_columns(["chosen","rejected"])
    # test_dataset = test_dataset.remove_columns(["chosen","rejected"])

    train_dataset = temp_dataset['train']

    val_dataset = temp_dataset['test']

    train_dataset.save_to_disk('./data/train_dataset')
    val_dataset.save_to_disk('./data/val_dataset')
    test_dataset.save_to_disk('./data/test_dataset')


def process_data(data):

    data['chosen'], data['rejected'] = data['rejected'], data['chosen']

    new_data = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(data["chosen"], data["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True)
        tokenized_rejected = tokenizer(rejected, truncation=True)

        new_data["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_data["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_data["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_data["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_data


if __name__=="__main__":
    main()