import torchvision.models as models
import torch.nn as nn
import torch

backbones = {"resnet18": {"model": models.resnet18, "weights": models.ResNet18_Weights.DEFAULT},
             "resnet34": {"model": models.resnet34, "weights": models.ResNet34_Weights.DEFAULT},
             "resnet50": {"model": models.resnet50, "weights": models.ResNet50_Weights.DEFAULT},
             "resnet101": {"model": models.resnet101, "weights": models.ResNet101_Weights.DEFAULT},
             "resnet152": {"model": models.resnet152, "weights": models.ResNet152_Weights.DEFAULT}}


class CustomResnet(nn.Module):
    def __init__(self, out_features, backbone="resnet18"):
        super(CustomResnet, self).__init__()
        # Load a pre-trained ResNet and remove its fully connected layer
        self.backbone = backbones[backbone]["model"](weights=backbones[backbone]["weights"])
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, out_features)

    def forward(self, x):
        features = self.backbone(x)
        return features


class IngredientNet(nn.Module):
    def __init__(self, middle_feat, num_classes, rgb_backbone, deep_backbone):
        super(IngredientNet, self).__init__()

        self.rgb_backbone = CustomResnet(middle_feat, rgb_backbone)
        self.deep_backbone = CustomResnet(middle_feat, deep_backbone)

        # Classification tail
        #self.classification_fc = nn.Sequential(
        #    nn.Linear(middle_feat*2, 1024),
        #    nn.ReLU(),
        #    nn.Linear(1024, 512),
        #    nn.ReLU(),
        #    #nn.Dropout(0.5),
        #    nn.Linear(512, num_classes),
        #    nn.Softmax(dim=1)
        #)

        # Regression tail
        self.regression_fc = nn.Sequential(
            nn.Linear(middle_feat*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_x, deep_x):
        # Base ResNet features
        rgb_fv = self.rgb_backbone(rgb_x)
        deep_fv = self.deep_backbone(deep_x)

        # Concatenate features
        features = torch.cat((rgb_fv, deep_fv), dim=1)

        # Classification output
        #class_output = self.classification_fc(features)

        # Regression output
        reg_output = self.regression_fc(features)

        return reg_output #class_output, reg_output
