from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get("HF_TOKEN")
login(token = hf_token)

wb_token = userdata.get("WANDB")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3.2 on Customer Support Dataset',
    job_type="training",
    anonymous="allow"
)

base_model = "meta-llama/Llama-3.2-3B-Instruct"
new_model = "llama-3.2-3b-it-Medical-Terms"
dataset_name = "gamino/wiki_medical_terms"

# Set torch dtype and attention implementation
torch_dtype = torch.float16
attn_implementation = "eager"
    
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

from datasets import load_dataset

dataset = load_dataset("gamino/wiki_medical_terms", split="train")

def preprocess_data(example): 
    prompt = f"Explain the medical term: {example['page_title']}"
    response = example['page_text']
    return {"input_text": prompt, "target_text": response}

dataset = dataset.map(preprocess_data)

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
tokenizer.chat_template = None
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)

from datasets import DatasetDict
# Divide o dataset (80% treino, 20% teste)
train_test_split = dataset.train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length= 512,
    dataset_text_field="input_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()

wandb.finish()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)