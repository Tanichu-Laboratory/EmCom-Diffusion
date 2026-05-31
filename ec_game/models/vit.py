import torch
import torch.nn as nn


class DinoV2Wrapper(nn.Module):
    def __init__(self, model_type, freeze=True, drop_path_rate=0.0):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_type)
        self.embed_dim = self.model.embed_dim

        if drop_path_rate > 0:
            for block in self.model.blocks:
                if hasattr(block, 'drop_path'):
                    block.drop_path.drop_prob = drop_path_rate

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()

    def forward(self, x, register_blk=-1):
        features = self.model.forward_features(x)
        cls_token = features['x_norm_clstoken'].unsqueeze(1)
        patch_tokens = features['x_norm_patchtokens']
        return torch.cat((cls_token, patch_tokens), dim=1)


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    """Kept for API compatibility; not used in the referential-game setting."""
    return pos_embed_checkpoint
