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
        
    model = IngredientNet(args.middle_features, args.num_ingredients, args.rgb_backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, args


def get_args():
    args = {
        "num_ingredients": 198,
        "num_epochs": 200,
        "lr": 0.0001,
        "lr_decay": 0.01,
        "batch_size": 16,
        "middle_features": 1024,
        "rgb_backbone": "resnet50",
        "full_train": True,
        "cls_criterion": nn.CrossEntropyLoss(),
        "save_path": "checkpoint/trained_model"
    }
    return args
