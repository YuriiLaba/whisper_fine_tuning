import torch
import whisper


def load_model(model_size, model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = whisper.load_model(model_size, device=device)

    if model_path is not None:
        params = torch.load(model_path)
        model.load_state_dict(params)

    return model