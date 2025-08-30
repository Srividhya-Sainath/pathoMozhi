import h5py
import torch

def load_feats_to_tensor(h5_file_path):
    """
    Load the value of the key 'feats' from an H5 file and convert it to a PyTorch tensor.
    Args:
        h5_file_path (str): Path to the H5 file.
    Returns:
        torch.Tensor: Tensor containing the data from the 'feats' key.
    """
    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            if "feats" in h5_file:
                feats_data = h5_file["feats"][:]
            elif "features" in h5_file:
                feats_data = h5_file["features"][:]
            else:
                raise KeyError(
                    f"Neither 'feats' nor 'features' found in {h5_file_path}"
                    )

        feats_tensor = torch.as_tensor(feats_data, dtype=torch.float32)
        return feats_tensor.unsqueeze(0)  # (1, N, D)
    except Exception as e:
        print(f"[load_feats_to_tensor] Error reading '{h5_file_path}': {e}")
        return None  
    
def load_pt_feats_to_tensor(pt_file_path):
    """
    Load a PyTorch tensor from a .pt file and ensure it's of shape (1, N, D).
    Args:
        pt_file_path (str): Path to the .pt file.
    Returns:
        torch.Tensor or None: Tensor with shape (1, N, D) or None if error.
    """
    try:
        feats_data = torch.load(pt_file_path)
        if isinstance(feats_data, dict):
            if "feats" in feats_data:
                feats_data = feats_data["feats"]
            elif "features" in feats_data:
                feats_data = feats_data["features"]
            else:
                raise KeyError(f"No valid tensor key found in dict: {pt_file_path}")
        if feats_data.ndim == 1:  # shape [D]
            feats_tensor = feats_data.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        elif feats_data.ndim == 2:  # shape [N, D]
            feats_tensor = feats_data.unsqueeze(0)  # (1, N, D)
        else:
            raise ValueError(f"Unexpected feature shape {feats_data.shape} in {pt_file_path}")
        return feats_tensor.to(dtype=torch.float32)
    except Exception as e:
        print(f"[load_pt_feats_to_tensor] Error reading '{pt_file_path}': {e}")
        return None