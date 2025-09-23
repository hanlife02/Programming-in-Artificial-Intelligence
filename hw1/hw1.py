import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建results目录
    os.makedirs('results', exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    # 数据集加载与处理：下载并处理数据集CiFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义：实现LeNet
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    net = Net()

    # 损失函数与优化器定义
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 初始化Tensorboard writer
    writer = SummaryWriter('runs/cifar10_lenet_experiment')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('CIFAR10_Sample_Images', img_grid)
    writer.add_graph(net, images)

    # 用于保存损失数据
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 模型训练：训练10个epoch
    for epoch in range(10):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                # 记录平均loss到Tensorboard
                writer.add_scalar('Loss/Train_Batch', running_loss / 2000, epoch * len(trainloader) + i)
                running_loss = 0.0
        
        # 记录每个epoch的平均loss
        epoch_avg_loss = epoch_loss / len(trainloader)
        writer.add_scalar('Loss/Train_Epoch', epoch_avg_loss, epoch)
        train_losses.append(epoch_avg_loss)
        
        # 计算验证集上的loss和accuracy
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_avg_loss = val_loss / len(testloader)
        val_accuracy = 100 * correct / total
        
        # 记录验证loss和accuracy到Tensorboard
        writer.add_scalar('Loss/Validation', val_avg_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        
        # 保存数据用于绘图
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch + 1}/10], Train Loss: {epoch_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        net.train()

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # 关闭Tensorboard writer
    writer.close()

    # 绘制损失曲线并保存
    plt.figure(figsize=(12, 4))
    
    # 绘制训练和验证损失
    plt.subplot(1, 2, 1)
    epochs = range(1, 11)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=300, bbox_inches='tight')

    # 模型测试：在测试集上测试模型，计算平均准确率，以及在各个类别上单独的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
if __name__ == '__main__':
    main()