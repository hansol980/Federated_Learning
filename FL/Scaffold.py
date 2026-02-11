import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import numpy as np
import time

# 1. Configurations
args = {
    'num_clients': 50,
    'num_rounds': 10,
    'frac': 0.25,
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
    
    # Split dataset into non-IID partitions
    labels = train_dataset.targets.numpy()
    
    idxs_sorted = np.argsort(labels)
    
    num_shards = num_clients * 2
    shard_size = int(len(train_dataset) / num_shards)
    
    shard_idxs = [i for i in range(num_shards)]
    
    client_idx_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
    
    for i in range(num_clients):
        rand_set = np.random.choice(shard_idxs, 2, replace=False)
        shard_idxs = list(set(shard_idxs) - set(rand_set))
        
        for shard in rand_set:
            start = shard * shard_size
            end = (shard + 1) * shard_size
            
            client_idx_dict[i] = np.concatenate((client_idx_dict[i], idxs_sorted[start:end]), axis=0)
            
    client_loaders = []
    for i in range(num_clients):
        client_ds = Subset(train_dataset, client_idx_dict[i])
        client_loaders.append(DataLoader(client_ds, batch_size=args['batch_size'], shuffle=True))
        
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
    return client_loaders, test_loader

# 4. Local training
def scaffold_local_train(model, train_loader, c_global, c_local):
    """
    SCAFFOLD 알고리즘을 적용한 로컬 학습 함수
    Args:
        model: 로컬 모델 (global_weights로 초기화됨)
        train_loader: 로컬 데이터 로더
        c_global: 서버의 제어 변량 (c)
        c_local: 클라이언트의 제어 변량 (c_i)
    Returns:
        model.state_dict(): 업데이트된 모델 가중치
        c_new: 업데이트된 클라이언트 제어 변량 (c_i+)
        c_delta: 제어 변량의 변화량 (c_i+ - c_i)
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    global_model_params = copy.deepcopy(model.state_dict())
    
    steps_per_epoch = len(train_loader)
    K = args['epochs'] * steps_per_epoch
    
    for epoch in range(args['epochs']):
        for data, target in train_loader:
            data, target = data.to(args['device']), target.to(args['device'])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # ***
            for name, param in model.named_parameters():
                if param.grad is not None:
                    correction = c_global[name] - c_local[name]
                    param.grad.data += correction.to(args['device'])
                    
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    c_new = {}
    c_delta = {}
    state_dict = model.state_dict()
    
    for name in c_local.keys():
        param_diff = global_model_params[name] - state_dict[name]
        term = param_diff / (K * args['lr'])
        
        c_new[name] = c_local[name] - c_global[name] + term
        c_delta[name] = c_new[name] - c_local[name]
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return model.state_dict(), c_new, c_delta, avg_loss


# 5. Federated Averaging
def fed_avg(global_weights, local_weights_list):
    avg_weights = copy.deepcopy(local_weights_list[0])
    
    for key in avg_weights.keys():
        for i in range(1, len(local_weights_list)):
            avg_weights[key] += local_weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))
    
    return avg_weights

# 5-2. SCAFFOLD global control variate update
def scaffold_update_global_control(c_global, c_delta_list, num_clients):
    c_global_new = copy.deepcopy(c_global)
    
    total_delta = copy.deepcopy(c_delta_list[0])
    for key in total_delta.keys():
        for i in range(1, len(c_delta_list)):
            total_delta[key] += c_delta_list[i][key]
        total_delta[key] = total_delta[key] / num_clients
        c_global_new[key] += total_delta[key]
    
    return c_global_new

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
    print(f"Device: {args['device']}")
    
    global_model = SimpleCNN().to(args['device']) 
    global_weights = global_model.state_dict()
    
    c_global = {}
    c_local_list = []
    
    for name, param in global_model.named_parameters():
        c_global[name] = torch.zeros_like(param)
        
    for _ in range(args['num_clients']):
        c_local = {}
        for name, param in global_model.named_parameters():
            c_local[name] = torch.zeros_like(param)
        c_local_list.append(c_local)
    
    client_loaders, test_loader = get_dataset(args['num_clients'])
    
    print(f"Federated Learning Start: {args['num_rounds']} rounds, {args['num_clients']} clients")
    total_start_time = time.time()

    m = max(int(args['frac'] * args['num_clients']), 1)

    for round in range(args['num_rounds']):
        round_start_time = time.time()
        local_weights_list = []
        c_delta_list = []
        client_losses = []
        
        idxs_users = np.random.choice(range(args['num_clients']), m, replace=False)
        
        for idx in idxs_users:
            local_model = SimpleCNN().to(args['device'])
            local_model.load_state_dict(global_weights)
            
            updated_weights, c_new, c_delta, avg_loss = scaffold_local_train(
                local_model, 
                client_loaders[idx],
                c_global, 
                c_local_list[idx]
            )
            
            c_local_list[idx] = c_new
            
            local_weights_list.append(updated_weights)
            c_delta_list.append(c_delta)
            client_losses.append(avg_loss)
        
        global_weights = fed_avg(global_weights, local_weights_list)
        
        c_global = scaffold_update_global_control(c_global, c_delta_list, args['num_clients'])
        
        global_model.load_state_dict(global_weights)
        acc = evaluate(global_model, test_loader)
        mean_client_loss = float(np.mean(client_losses)) if client_losses else 0.0
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        print(f"Round {round+1}/{args['num_rounds']} - Client Avg Loss: {mean_client_loss:.4f} | Global Accuracy: {acc:.2f}%, | Time: {round_duration:.2f}s")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total Training Time: {total_duration:.2f}s")

if __name__ == '__main__':
    main()
