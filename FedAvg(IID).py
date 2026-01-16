import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import copy
import numpy as np
import time

# 1. Configurations
args = {
    'num_clients': 5,
    'num_rounds': 10,
    'epochs': 1,
    'batch_size': 32,
    'lr': 0.01,
    'device': 'cpu'
}

# 2. Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Load and partition dataset
def get_dataset(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    data_len = len(train_dataset)
    split_size = data_len // num_clients
    split_sizes = [split_size] * num_clients
    
    if sum(split_sizes) < data_len:
        split_sizes[0] += data_len - sum(split_sizes)
        
    client_datasets = random_split(train_dataset, split_sizes)
    
    client_loaders = [DataLoader(ds, batch_size=args['batch_size'], shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return client_loaders, test_loader

# 4. Local training
def local_train(model, train_loader):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args['epochs']):
        for data, target in train_loader:
            data, target = data.to(args['device']), target.to(args['device'])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

# 5. Federated Averaging
def fed_avg(global_weights, local_weights_list):
    avg_weights = copy.deepcopy(local_weights_list[0])
    
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] += local_weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
    
    return avg_weights

# 6. Evaluation
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args['device']), target.to(args['device'])
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def main():
    total_start_time = time.time()
    print(f"Device: {args['device']}")
    
    global_model = SimpleCNN().to(args['device'])
    global_weights = global_model.state_dict()
    
    client_loaders, test_loader = get_dataset(args['num_clients'])
    
    print(f"Federated Learning Start: {args['num_rounds']} rounds, {args['num_clients']} clients")

    for round in range(args['num_rounds']):
        round_start_time = time.time()
        local_weights_list = []
        
        for i in range(args['num_clients']):
            local_model = SimpleCNN().to(args['device'])
            local_model.load_state_dict(global_weights)
            
            updated_weights = local_train(local_model, client_loaders[i])
            local_weights_list.append(updated_weights)
        
        global_weights = fed_avg(global_weights, local_weights_list)
        
        global_model.load_state_dict(global_weights)
        acc = evaluate(global_model, test_loader)
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        print(f"Round {round+1}/{args['num_rounds']} - Global Accuracy: {acc:.2f}%, | Time: {round_duration:.2f}s")
        
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total Training Time: {total_duration:.2f}s")
if __name__ == '__main__':
    main()
