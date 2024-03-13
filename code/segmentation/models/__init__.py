from .unet.unet import UNet


def get_model(args):

    if args.model.startswith("unet"):
        block_type = args.model.split("_")[-1]
        
        model = UNet(
            dimension=args.input_dimension, 
            input_channels=args.input_channel, 
            output_classes=args.num_classes, 
            channels_list=args.unet_channels, 
            deep_supervision=args.deep_supervision, 
            ds_layer=args.deep_supervision_layer, 
            block_type=block_type,
            normalization=args.normalization,
            dropout_prob=args.dropout_prob,
        )
    else:
        raise NotImplementedError()
        
    return model