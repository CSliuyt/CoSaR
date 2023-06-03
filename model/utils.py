from model.resnet import *
from model.wrn import WideResNet


def build_model(args):
    if args.model == "wrn":
        net = WideResNet(num_classes=args.num_classes)
    elif args.model == 'resnet18':
        net = ResNet18(num_classes=args.num_classes)
    elif args.model == "resnet34":
        net = ResNet34(num_classes=args.num_classes)
    else:
        net = ResNet50(num_classes=args.num_classes)

    return net

