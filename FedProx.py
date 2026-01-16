import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import numpy as np
import time

args = {
    'num_clients': 5,
    'num_rounds': 10,
    'epochs': 1,
    'batch_size': 32,
    'lr': 0.01,
    'mu': 0.1, # Proximal term constant
    'device': 'cpu'
}

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
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
    
def get_dataset(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    labels = train_dataset.targets.numpy()
    
    idxs_sorted = np.argsort(labels)
    
    num_shards = num_clients * 2
    shard_size = int(len(train_dataset) / num_shards)
    
    shard_idxs = [i for i in range(num_shards)]
    
    client_idx_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
    
    for i in range(num_clients):
        rand_shard_idxs = np.random.choice(shard_idxs, 2, replace=False)
        shard_idxs = list(set(shard_idxs) - set(rand_shard_idxs))
        
        for shard in rand_shard_idxs:
            client_idx_dict[i] = np.concatenate(
                (client_idx_dict[i], idxs_sorted[shard*shard_size:(shard+1)*shard_size]), axis=0
            )
    
    client_loaders = []
    for i in range(num_clients):
        client_subset = Subset(train_dataset, client_idx_dict[i])
        client_loader = DataLoader(client_subset, batch_size=args['batch_size'], shuffle=True)
        client_loaders.append(client_loader)
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return client_loaders, test_loader

def local_train(model, train_loader, global_model):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # ***
    global_weight_collector = list(global_model.parameters())
    
    for epoch in range(args['epochs']):
        for data, target in train_loader:
            data, target = data.to(args['device']), target.to(args['device'])
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            
            # add proximal term
            if args['mu'] > 0:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_weight_collector):
                    proximal_term += ((w - w_t).norm(2))**2
                
                loss += (args['mu'] / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def fed_avg(global_weights, local_weights_list):
    avg_weights = copy.deepcopy(local_weights_list[0])
    
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] += local_weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
    
    return avg_weights

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
    print(f"Device: {args['device']}")
    
    global_model = SimpleCNN().to(args['device'])
    global_weights = global_model.state_dict()
    
    client_loaders, test_loader = get_dataset(args['num_clients'])
    
    print(f"Federated Learning (FedProx) Start: {args['num_rounds']} rounds, {args['num_clients']} clients")
    total_start_time = time.time()

    for round in range(args['num_rounds']):
        round_start_time = time.time()
        local_weights_list = []
        
        for i in range(args['num_clients']):
            local_model = SimpleCNN().to(args['device'])
            local_model.load_state_dict(global_weights)
            
            # Pass global model to local_training for proximal term calculation
            updated_weights = local_train(local_model, client_loaders[i], global_model)
            
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