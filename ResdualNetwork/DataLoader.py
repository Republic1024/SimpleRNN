from torchvision import datasets, transforms

data_path = 'Dataset/'

cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

cifar10_val = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

# label 1*1 img 3 * 32 *32
label_map = {0: 0, 2: 1} # one hot label create
x_train_with_labels = [(img, label_map[label])
                       for img, label in cifar10
                       if label in [0, 2]]

x_train_shape = x_train_with_labels[0][0].shape #3 * 32 *32

# print(x_train_with_labels[0][0].shape,x_train_with_labels[0][1])

x_test_with_labels = [(img, label_map[label])
                      for img, label in cifar10_val
                      if label in [0, 2]]

