import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm
import wandb
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score


from model import IngredientNet
from utils import *
from loader import FoodDataset
import numpy as np


def eval(model, device, args, test_dataloader):
    model.eval()
    running_test_loss = 0.0

    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_dataloader:
            rgb, cls_labels = data
            #cls_labels = (reg_labels > 0).to(torch.float)
            rgb, cls_labels = rgb.to(device), cls_labels.to(device)#, cls_labels.to(device)

            cls_output = model(rgb) #cls_output, 

            # Compute the loss
            loss = args.cls_criterion(cls_output, cls_labels)
            running_test_loss += loss.item()

            all_preds.append(cls_output.cpu())
            all_labels.append(cls_labels.cpu())

    average_val_loss = running_test_loss / len(test_dataloader)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Calculate AP for each class and mAP
    #aps = []
    #num_classes = all_labels.shape[1]  # Assuming one-hot encoded labels
    #for i in range(num_classes):
    #    ap = average_precision_score(all_labels[:, i], all_preds[:, i])
    #    aps.append(ap)
    #mAP = np.mean(aps)

    threshold = 0.7
    binarized_lab = np.where(all_preds > threshold, 1, 0)
    
    precision = precision_score(all_labels, binarized_lab, average="macro", zero_division=0)
    recall = recall_score(all_labels, binarized_lab, average="macro", zero_division=0)

    f1 = 2 * (precision * recall) / (precision + recall) if precision+recall != 0 else 0

    return average_val_loss, f1, precision, recall


def train(model, device, args, train_dataloader, test_dataloader):

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.lr_decay)
    
    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_dataloader)):
            rgb, labels = data
            #print(rgb.dtype, deep.dtype, reg_labels.dtype)
            labels = labels.to(torch.float)
            #cls_labels = (reg_labels > 0).to(torch.float)
            rgb, labels = rgb.to(device), labels.to(device)#, cls_labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            cls_output = model(rgb) 

            # Soglia
            #cls_output = (cls_output > args.threshold).to(torch.float)

            # Compute the loss
            loss = args.cls_criterion(cls_output, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
            
                wandb.log({"train_loss": running_loss/100, "epoch": float(epoch)+float(i/len(train_dataloader))})
                running_loss = 0.0


        avg_val_loss, f1, precision, recall = eval(model, device, args, test_dataloader)

        wandb.log({"test_loss": avg_val_loss, "epoch": float(epoch+1)})
        wandb.log({"f1": f1, "epoch": float(epoch+1)})

        print('[Epoch %d] Test loss: %.3f, f1: %.3f' % (epoch, avg_val_loss, f1))

        if epoch % 3 == 0:
            save_model(model, args.save_path+"_"+str(epoch), args)


def load_data_and_train(args):

    train_dataset = FoodDataset(mode="train", augument=3, cache=True)
    test_dataset = FoodDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = IngredientNet(args.middle_features, args.num_ingredients, args.rgb_backbone)

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
    run_name = "classificator"
    wandb.init(project="the_chef", config=args, name=run_name)
    args = DictToObject(args)

    load_data_and_train(args)
