import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os

def train_model():
    """训练CIFAR-10分类模型"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建results目录，存loss curve数据
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
    net.to(device)

    # 损失函数与优化器定义
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 初始化CSV文件保存训练数据
    csv_file = 'results/training_data.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss', 'val_accuracy'])

    # 用于保存损失数据
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 记录损失的频率（每多少个batch记录一次）
    log_interval = 500

    # 模型训练：训练10个epoch
    for epoch in range(10):
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            
            # 每log_interval个batch记录一次损失
            if (i + 1) % log_interval == 0:
                # 计算验证损失
                net.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_data in testloader:
                        val_images, val_labels = val_data
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        val_outputs = net(val_images)
                        val_loss += criterion(val_outputs, val_labels).item()
                        _, predicted = torch.max(val_outputs, 1)
                        total += val_labels.size(0)
                        correct += (predicted == val_labels).sum().item()
                
                val_avg_loss = val_loss / len(testloader)
                val_accuracy = 100 * correct / total
                avg_train_loss = running_loss / log_interval
                
                # 记录到CSV
                with open(csv_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([epoch + 1, i + 1, avg_train_loss, val_avg_loss, val_accuracy])
                
                print(f'[{epoch + 1}, {i + 1:5d}] Train Loss: {avg_train_loss:.3f}, Val Loss: {val_avg_loss:.3f}, Val Acc: {val_accuracy:.2f}%')
                running_loss = 0.0
                net.train()
        
        # 在每个epoch结束时也记录一次
        epoch_avg_loss = epoch_loss / len(trainloader)
        
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_avg_loss = val_loss / len(testloader)
        val_accuracy = 100 * correct / total
        
        # 记录epoch结束时的数据
        with open(csv_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch + 1, len(trainloader), epoch_avg_loss, val_avg_loss, val_accuracy])
        
        print(f'Epoch [{epoch + 1}/10] End - Train Loss: {epoch_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        net.train()

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    print('Training data saved to results/training_data.csv')

    # 模型测试：在测试集上测试模型，计算平均准确率，以及在各个类别上单独的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
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
            images, labels = images.to(device), labels.to(device)
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
    train_model()