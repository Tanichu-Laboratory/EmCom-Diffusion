from ec_game.models.vit import DinoV2Wrapper


def create_vit(vit='base', image_size=224, use_grad_checkpointing=False,
               ckpt_layer=0, drop_path_rate=0, freeze=True):
    assert vit in ('base', 'large'), "vit must be 'base' or 'large'"
    if vit == 'base':
        vision_width = 768
        model_type = 'dinov2_vitb14'
    else:
        vision_width = 1024
        model_type = 'dinov2_vitl14'
    visual_encoder = DinoV2Wrapper(
        model_type=model_type,
        freeze=freeze,
        drop_path_rate=drop_path_rate,
    )
    return visual_encoder, vision_width
