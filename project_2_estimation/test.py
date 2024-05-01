import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import model

def create_data_loader(data, variable_index, leadtime=1, batch_size=64):
    inputs = data[:, :528]  # Exclude the last time step to avoid out-of-index error
    targets = data[:, variable_index]  # Shifted by one time step

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(input_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

criterion = RMSELoss()

def main():

    # Set number and leadtime
    number = 9
    leadtime = 0

    root_path = '.'
    model_path = os.path.join(root_path, 'model', f'model_checkpoint_n{number}_l{leadtime}.pth')
    ts_model = model.TimeSeriesNN(dropout_rate=0.03)
    ts_model.load_state_dict(torch.load(model_path))
    ts_model.eval()

    criterion = RMSELoss()
    test_datasets = [np.load(os.path.join(root_path, f'processed/00{number}/test_{i}.npy')) for i in range(1, 4)]
    test_loaders = [create_data_loader(ts, 545, leadtime=leadtime) for ts in test_datasets]

    test_results = [test(ts_model, tl, criterion) for tl in test_loaders]
    for i, result in enumerate(test_results):
        print(f'Test Set {i+1} Loss: {result:.5f}')

if __name__ == '__main__':
    main()
