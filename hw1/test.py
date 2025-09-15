import torch
import torchvision
import torchvision.transforms as transforms

# data download and data loader
batch_size = 4
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
trainset,batch_size=batch_size, shuffle=True, num_workers=2)

dataiter = iter(trainloader)
images, labels = next(dataiter)