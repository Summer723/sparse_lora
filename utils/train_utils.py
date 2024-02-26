import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig
from transformers import BertForSequenceClassification
import torch.nn as nn



def validation(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=64)
    model.eval()
    num_correct = 0
    with torch.no_grad():
        for i, data_pair in enumerate(dataloader):
            inputs, label = data_pair
            label = label.to(device)

            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            token_type_ids = inputs['token_type_ids'].squeeze(1).to(device)

            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

            num_correct += torch.sum(torch.argmax(output.logits, dim=1) == label)

    return num_correct/len(dataset)


def test(model, dataset, device):
    return validation(model, dataset, device)


def get_model(model, lora, last_layer, lora_r, lora_algo):
    if model.lower() not in ["softmaxbert"]:
        return "Model not supported"
    elif model.lower() == "softmaxbert":
        if lora_algo.lower()=="lora":
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=1,
                lora_dropout=0.1,
            # target_modules=["q_proj", "v_proj","k_proj"]
            )
            
        elif lora_algo.lower()=="adalora":
            peft_config =  AdaLoraConfig(
                                                    peft_type="ADALORA", 
                                                    task_type=TaskType.SEQ_CLS, 
                                                    target_r = lora_r,
                                                    init_r = lora_r//2,
                                                    lora_alpha=1,
                                                    #target_modules=["q", "v"],
                                                    lora_dropout=0.1,
                                                )
            
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)
        model = get_peft_model(model, peft_config)
        print("There are: {} trainable parameters".format(print_model_params(model)))
        return model


def build_model():
    pass


def save_model(model, path):
    pass


def print_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)