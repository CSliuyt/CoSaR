import torchvision
from dataset.randaugmentation.autoaugment import CIFAR10Policy, ImageNetPolicy
from dataset.cifar10 import get_cifar10
from dataset.cifar100 import get_cifar100
from torch.utils.data import DataLoader
from dataset.ood_dataset import OodSet
from dataset.tinyimagenet import get_tinyimagenet


def build_transform(rescale_size=68, crop_size=64):
    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    cifar_train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.492, 0.482, 0.446), std=(0.247, 0.244, 0.262))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.492, 0.482, 0.446), std=(0.247, 0.244, 0.262))
    ])
    train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        ImageNetPolicy(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.492, 0.482, 0.446), std=(0.247, 0.244, 0.262))
    ])
    return {'train': train_transform, 'test': test_transform, 'train_strong': train_transform_strong_aug,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform,
            'cifar_train_strong': cifar_train_transform_strong_aug}


def build_dataloader(dataset, cfg):
    trans = build_transform(rescale_size=68, crop_size=64)

    if dataset == 'cifar10':
        train_set = get_cifar10('dataset/cifar10d', cfg.noise_type, cfg.noise_ratio, train=True,
                                transform_train=trans['cifar_train'],
                                transform_train_aug=trans['cifar_train_strong'],
                                transform_val=trans['cifar_test'])
        test_set = get_cifar10('dataset/cifar10d', cfg.noise_type, cfg.noise_ratio, train=False,
                               transform_train=trans['cifar_train'],
                               transform_train_aug=trans['cifar_train_strong'],
                               transform_val=trans['cifar_test'])

    elif dataset == 'cifar100':
        train_set = get_cifar100('./dataset/cifar100d', cfg.noise_type, cfg.noise_ratio, train=True,
                                 transform_train=trans['cifar_train'],
                                 transform_train_aug=trans['cifar_train_strong'],
                                 transform_val=trans['cifar_test'])

        test_set = get_cifar100('./dataset/cifar100d', cfg.noise_type, cfg.noise_ratio, train=False,
                                transform_train=trans['cifar_train'],
                                transform_train_aug=trans['cifar_train_strong'],
                                transform_val=trans['cifar_test'])

    elif dataset == 'tinyimagenet':
        train_set, _ = get_tinyimagenet('tinyimagenet', cfg['noise_type'], cfg['noise_ratio'], train=True,
                                        transform_train=trans['train'],
                                        transform_train_aug=trans['train_strong'],
                                        transform_val=trans['test'])
        _, test_set = get_tinyimagenet('tinyimagenet', cfg['noise_type'], cfg['noise_ratio'], train=False,
                                       transform_train=None,
                                       transform_train_aug=None,
                                       transform_val=trans['test'])

    elif dataset == 'clothing1m':
        pass
    else:
        pass
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.prefetch, pin_memory=True)

    test_loader = DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.prefetch, pin_memory=False)

    return train_loader, test_loader, len(train_set), len(test_set)


def build_ood_loader(cfg):

    ood_data = OodSet(cfg.aux_dataset, ood_num_examples=cfg.aux_num_samples, num_to_avg=cfg.aux_num_to_avg)
    print(f"OOD: {len(ood_data)}")
    train_loader_out = DataLoader(
        ood_data,
        batch_size=cfg.aux_batch_size, shuffle=True,
        num_workers=cfg.prefetch, pin_memory=True)

    return train_loader_out, len(ood_data)
