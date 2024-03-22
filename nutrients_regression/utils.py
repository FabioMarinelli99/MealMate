import torch
import os
import torch.nn as nn
from model import IngredientNet


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def save_model(model, path, args):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save({'model_state_dict': model.state_dict(),
                'args':args}, path+'.pth')


def load_model(path):
    
    checkpoint = torch.load(path)
    try:
        args = checkpoint['args']
    except:
        args = get_args()
    
    args = DictToObject(args)
    model = IngredientNet(args.middle_features, args.num_ingredients, args.rgb_backbone, args.deep_backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, args


def r_squared(y_true, y_pred):
    # Calculate the total sum of squares (TSS)
    tss = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2)
    # Calculate the residual sum of squares (RSS)
    rss = torch.sum((y_true - y_pred) ** 2)
    # Calculate R-squared
    r2_score = 1 - rss / tss
    return r2_score


def get_args():
    args = {
        "num_ingredients": 4,
        #"weight_regression": 0.5,
        "num_epochs": 200,
        "lr": 0.0001,
        "lr_decay": 0.01,
        "batch_size": 16,
        "middle_features": 1024,
        "threshold": 0.2,
        "rgb_backbone": "resnet101",
        "deep_backbone": "resnet18",
        "full_train": True,
        "reg_criterion": nn.L1Loss(), #nn.MSELoss(),
        "cls_criterion": nn.CrossEntropyLoss(),
        "save_path": "checkpoint/trained_model"
    }
    return args
