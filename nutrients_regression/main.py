import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from model import IngredientNet
from utils import *
from loader import FoodDataset


def eval(model, device, args, test_dataloader):
    model.eval()
    running_test_loss = 0.0
    mae_metric = 0.0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in test_dataloader:
            rgb, deep, reg_labels = data
            #cls_labels = (reg_labels > 0).to(torch.float)
            rgb, deep, reg_labels = rgb.to(device), deep.to(device), reg_labels.to(device)#, cls_labels.to(device)

            reg_output = model(rgb, deep) #cls_output, 

            # Compute the loss
            reg_loss = args.reg_criterion(reg_output, reg_labels)
            #cls_loss = args.cls_criterion(cls_output, cls_labels)

            #loss = (reg_loss * weight_reg) + (cls_loss * weight_cls)
            loss = reg_loss


            running_test_loss += loss.item()

            all_labels.append(reg_labels.cpu())
            all_preds.append(reg_output.cpu())

            #batch_error_summed = (torch.abs(reg_labels - reg_output)).sum()
            #mae_metric += batch_error_summed

            #l1_loss_summed = nn.L1Loss(reduction='sum')
            #print(f'batch_error_summed: {batch_error_summed}; l1_loss_summed: {l1_loss_summed(reg_output, reg_labels).item()}')

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    single_mae = (torch.abs(all_labels - all_preds).sum(dim=0) * test_dataloader.dataset.var) / len(test_dataloader.dataset)

    mape = single_mae / test_dataloader.dataset.avg * 100

    mae = single_mae.sum()

    #mae_metric = (torch.abs(all_labels - all_preds) * test_dataloader.dataset.var).sum()
    #mae = mae_metric / len(test_dataloader.dataset)

    rsquare = r_squared(torch.tensor(all_labels), torch.tensor(all_preds))

    average_val_loss = running_test_loss / len(test_dataloader)

    metrics = {"mae": mae, "single_mae": single_mae, "mape": mape, "r2": rsquare, "avg_val_loss": average_val_loss}

    return metrics


def train(model, device, args, train_dataloader, test_dataloader):

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.lr_decay)
    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_dataloader)):
            rgb, deep, reg_labels = data
            reg_labels = reg_labels.to(torch.float)
            rgb, deep= rgb.to(device), deep.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            reg_output = model(rgb, deep) # cls_output skippato

            # Compute the loss
            reg_loss = args.reg_criterion(reg_output, reg_labels)
            #cls_loss = args.cls_criterion(cls_output, cls_labels)

            #print("reg_loss", reg_loss.item(), "cls_loss", cls_loss.item())

            #loss = (reg_loss * args.weight_regression) + (cls_loss * (1 - args.weight_regression))
            loss = reg_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
            
                wandb.log({"train_loss": running_loss/100, "epoch": float(epoch)+float(i/len(train_dataloader))})
                running_loss = 0.0


        metrics = eval(model, device, args, test_dataloader)

        wandb.log({"test_loss": metrics["avg_val_loss"], "MAE": metrics["mae"], "R2": metrics["r2"], "epoch": float(epoch+1)})

        print(f"[Epoch {epoch}] MAE: {metrics['mae']}, SingleMAE: {metrics['single_mae']}, MAPE: {metrics['mape']}, R2: {metrics['r2']}, Avg Val Loss: {metrics['avg_val_loss']}")

        if epoch % 3 == 0:
            save_model(model, args.save_path+"_"+str(epoch), args)


def load_data_and_train(args):
    # da sostituire
    train_dataset = FoodDataset(mode="train", augument=3, cache=True)
    test_dataset = FoodDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = IngredientNet(args.middle_features, args.num_ingredients, args.rgb_backbone, args.deep_backbone)

    # disable gradients for the backbone
    if not args.full_train:
        for name, param in model.named_parameters():
            if any(enable_layers in name for enable_layers in ["backbone.fc", "deep_backbone", "classification_fc", "regression_fc"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Starting training")

    train(model, device, args, train_dataloader, test_dataloader)

    print('Finished Training')

    wandb.finish()

    save_model(model, args.save_path, args)




if __name__ == "__main__":

    args = get_args()

    run_name = "ovo_try"

    wandb.init(project="the_chef", config=args, name=run_name)

    args = DictToObject(args)

    load_data_and_train(args)
