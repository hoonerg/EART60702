# train.py
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

import model

def create_data_loader(data, variable_index, leadtime=1, batch_size=64, shuffle=True, device='cpu'):
    input_data = data[:-leadtime]
    target_data = data[leadtime:, variable_index]

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(target_data, dtype=torch.float32).view(-1, 1).to(device)

    dataset = TensorDataset(input_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

def plot_losses(train_losses, val_losses, root_path):
    plt.figure(figsize=(12, 10))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss per Epoch', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(root_path, 'model/loss_plot.png'))
    plt.show()

def main():
    root_path = '/Users/hoonchoi/project_2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists(os.path.join(root_path, 'model')):
        os.makedirs(os.path.join(root_path, 'model'))

    ### Number: Dataset number to train on
    ### Leadtime: Number of timesteps to predict in the future
    number = 6
    leadtime = 6

    train_file = f'processed/00{number}/training.npy'
    valid_file = f'processed/00{number}/validation.npy'
    train_set = np.load(os.path.join(root_path, train_file))
    valid_set = np.load(os.path.join(root_path, valid_file))
    
    # 545 is the index of TREFMXAV_U for Manchester
    train_loader = create_data_loader(train_set, 545, leadtime=leadtime, batch_size=64, shuffle=True, device=device)
    val_loader = create_data_loader(valid_set, 545, leadtime=leadtime, batch_size=64, shuffle=False, device=device)

    ts_model = model.TimeSeriesNN(dropout_rate=0.03) # Change dropout_rate
    ts_model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ts_model.parameters(), lr=0.000001) # Change learning_rate

    num_epochs = 200 # Change num_epochs
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train(ts_model, train_loader, criterion, optimizer)
        val_loss = validate(ts_model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = ts_model.state_dict()

    # Save the best model state after all epochs are completed
    torch.save(best_model_state, os.path.join(root_path, f'model/model_checkpoint_n{number}_l{leadtime}.pth'))
    print(f'Best val_Loss: {best_val_loss:.5f}')
    plot_losses(train_losses, val_losses, root_path)

if __name__ == '__main__':
    main()
