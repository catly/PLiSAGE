

def save_checkpoint(state, is_best, directory, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(directory, "best_model.pth.tar")
        torch.save(state, best_filepath)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint file not found at '{checkpoint_path}', starting from scratch.")
        return 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            logging.warning(f"Could not load optimizer state: {e}. Initializing a new optimizer.")

    logging.info(f"Loaded checkpoint from '{checkpoint_path}' (epoch {checkpoint.get('epoch', 0)})")
    return checkpoint.get('epoch', 0) + 1
