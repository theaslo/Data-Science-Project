from transformers import RobertaTokenizer, IBertModel
import torch
from datasets import load_dataset
from IPython.display import display, HTML
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import math

def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result



model_checkpoint = "distilroberta-base"
datasets = load_dataset("text", data_files={"train": 'sample.txt', "validation": 'sample.txt'})
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
block_size = tokenizer.model_max_length


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)
# tokenizer = RobertaTokenizer.from_pretrained('kssteven/ibert-roberta-base')
# model = IBertModel.from_pretrained('kssteven/ibert-roberta-base')

# datasets = load_dataset("text", data_files={"train": 'sample.txt'})
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    "test-mlm",
    num_train_epochs = 30,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
model.save_pretrained('model_saved2')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)