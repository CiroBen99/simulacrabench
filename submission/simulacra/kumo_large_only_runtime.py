"""Large checkpoint loader for the unchanged frozen Kumo-only prediction recipe."""
import hashlib
import types
from pathlib import Path


class FrozenKumoLarge:
    def __init__(self, root, config):
        import torch
        import sdm
        if config['model_size'] != 'large' or config['checkpoint_file'] != 'kumo-large.pt':
            raise ValueError('Require the pinned Large checkpoint, no Small replacement')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else config['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable')
        path = Path(root) / config['checkpoint_file']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != config['checkpoint_sha256']:
            raise ValueError('Frozen Large checkpoint changed')
        class Large(sdm.models.KumoTabular):
            def _load_from_pretrained(inner, size, device):
                if size != 'large' or len(inner.models) != 1:
                    raise ValueError('Wrong Large model recipe')
                model = next(iter(inner.models.values()))
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True, assign=True)
                return model
        torch.set_num_threads(config['threads'])
        self.sdm = sdm
        self.model = Large(task='classification', size='large', device=device)
        self.used = False
        self.forward_device = device if device != 'cpu' else None
        self.model_size = 'large'
        self.parameters = sum(p.numel() for p in self.model.parameters())
        original = self.model.ecoc._draw_codebook
        def cpu_codebook(inner, count, target_device, generator):
            return original(count, torch.device('cpu'), generator).to(target_device)
        self.model.ecoc._draw_codebook = types.MethodType(cpu_codebook, self.model.ecoc)
