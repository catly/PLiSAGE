import time
import torch_geometric

def set_gpu(data, device):
    if isinstance(data, tuple):
        return tuple(set_gpu(d, device) for d in data)
    elif isinstance(data, (torch.Tensor, torch_geometric.data.Data)):
        return data.to(device)
    else:
        return data

def estimate_time(start_time, current_epoch, total_epochs):
    elapsed_time = time.time() - start_time
    avg_epoch_time = elapsed_time / (current_epoch + 1)
    remaining_time = avg_epoch_time * (total_epochs - current_epoch - 1)
    return time.strftime("%H:%M:%S", time.gmtime(remaining_time))