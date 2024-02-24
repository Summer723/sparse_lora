import torch
from torch.utils.data import DataLoader
from utils.train_utils import validation, test, get_model
import torch.nn as nn
import src.dataset as datasets
from tqdm import tqdm 
import wandb 

def train(model,
          epoch,
          batch_size,
          lr,
          device,
          last_layer,
          strategy,
          path,
          seed,
          lora,
          lora_r,
          l1_reg,
          l1_lambda,
          ):

    torch.manual_seed(seed)
    # temporary
    model = get_model(model, lora, last_layer,lora_r)
    model = model.to(device)
    dataset = datasets.NewsDataset()
    training_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(training_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,)
    loss_fn = torch.nn.CrossEntropyLoss()

    running_loss = 0
    for epoch_id in tqdm(range(epoch)):
        model.train()

        for i, datapair in enumerate(train_loader):
            data, label = datapair
            label = label.to(device)

            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].squeeze(1).to(device)
            token_type_ids = data['token_type_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                           )
            activation = nn.Softmax(dim=1)
            loss = loss_fn(activation(output.logits), label)
            if l1_reg:
                for name, param in model.named_parameters():
                    if 'classifier' not in name and param.requires_grad:
                        loss += l1_lambda * param.abs().sum()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        val_loss = validation(model, validation_set, device)
        print("Validation accuracy @ Epoch{}: {}".format(epoch_id,
                                                         val_loss))
    test_loss = test(model, test_set, device)
    print("Test accuracy @ Epoch{}: {}".format(epoch_id, test_loss))
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss,
    # }, path+ "/model.pth")