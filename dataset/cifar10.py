import numpy as np
import torch
import torchvision
from PIL import Image
from numpy.testing import assert_array_almost_equal


def get_cifar10(root, noise_type, noise_ratio, train=True,
                transform_train=None, transform_train_aug=None, transform_val=None):
    if train:
        dataset = CIFAR10_train(root, noise_ratio, transform=transform_train,
                                transform_aug=transform_train_aug)
        if noise_type == 'asymmetric':
            dataset.asymmetric_noise()

        elif noise_type == 'instance':
            dataset.instance_noise()

        elif noise_type == 'symmetric':
            dataset.symmetric_noise()
        elif noise_type == 'openset':
            dataset.open_noise()
        print(f'noise type: {noise_type}')
        print(f"Train: {len(dataset)}")  # Train: 45000 Val: 5000
    else:
        dataset = CIFAR10_val(root, transform=transform_val)
        print(f"Test: {len(dataset)}")

    return dataset


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify(y_train, noise_transition_matrix, random_state=None):
    y_train_noisy = multiclass_noisify(y_train, P=noise_transition_matrix, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    return y_train_noisy, actual_noise


class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, noise_ratio, transform=None, transform_aug=None, target_transform=None, download=False):
        super(CIFAR10_train, self).__init__(root, train=True,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        self.num_classes = 10
        self.noise_ratio = noise_ratio
        self.train_data = self.data  # self.train_data[indexs]
        self.train_labels = np.array(self.targets)  # np.array(self.train_labels)[indexs]
        self.noise_indx = []
        self.transform_aug = transform_aug
        self.transform = transform
        self.train_labels_gt = self.train_labels.copy()

    def symmetric_noise(self):
        P = np.ones((self.num_classes, self.num_classes))
        P = (self.noise_ratio / (self.num_classes - 1)) * P
        for i in range(self.num_classes):
            P[i, i] = 1.0 - self.noise_ratio
        for i in range(self.num_classes, self.num_classes):
            P[i, :] = 1.0 / self.num_classes
        for i in range(self.num_classes, self.num_classes):
            P[:, i] = 0.0
        print(P)
        self.train_labels, auc_noise = noisify(self.train_labels, P, random_state=0)
        print(auc_noise)

    def asymmetric_noise(self):
        # self.train_labels_gt = self.train_labels.copy()
        self.train_labels, noise_rate = noisify_pairflip(self.train_labels, self.noise_ratio, random_state=0, nb_classes=10)
        print(noise_rate)
        """    def instance_noise(self):

        noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.noise_ratio))

        noisylabel = noise_label['noise_label_train'][self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)"""

    def open_noise(self):
        open_set = torchvision.datasets.CIFAR100('cifar100d', train=True, transform=self.transform)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.noise_ratio * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_data[idx] = open_set.data[idx]

    def instance_noise(self):
        self.train_labels = torch.load('dataset/cifar10d/CIFAR-10_human.pt')['worse_label']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform_aug is not None:
            img_aug = self.transform_aug(img)
        else:
            img_aug = img.copy()

        if self.transform is not None:
            img_weak1 = self.transform(img)
            img_weak2 = self.transform(img)
        else:
            img_weak1 = img.copy()
            img_weak2 = img.copy()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_weak1, img_weak2, img_aug, target, target_gt, index

    def __len__(self):
        return len(self.train_data)


class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, transform=None, target_transform=None, download=False):
        super(CIFAR10_val, self).__init__(root, train=False,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 10
        self.train_data = self.data
        self.train_labels = np.array(self.targets)
        self.train_labels_gt = self.train_labels.copy()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img, target, target_gt, index


if __name__ == '__main__':
    set = get_cifar10('cifar10d', noise_type='', noise_ratio='', train=False)
    set2 = get_cifar10('cifar10d', noise_type='instance', noise_ratio=0.4, train=True)

