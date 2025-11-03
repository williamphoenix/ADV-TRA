from utils import utils
from utils.utils import checkattr
from utils.params.config import CONFIG
##-------------------------------------------------------------------------------------------------------------------##

def define_classifier(args, config, device, depth=0, stream=False, attack=False):
    model = define_standard_classifier(args=args, config=config, device=device, depth=depth)
    return model

def define_standard_classifier(args, config, device, depth=0):
    # Import required model
    from attacks.classifier import Classifier
    # Specify model
    model = Classifier(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac')
    ).to(device)
    # Return model
    return model

def define_badnets_classifier(args, config, device, depth=0):
    from attacks.classifier_badnets import ClassifierBadNets
    # Specify model
    model = ClassifierBadNets(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac',
                          ),
    ).to(device)
    # Return model
    return model

def define_dynamic_classifier(args, config, device, depth=0):
    from attacks.classifier_dynamic import ClassifierDynamic
    # Specify model
    model = ClassifierDynamic(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac',
                          ),
    ).to(device)
    # Return model
    return model

def define_clean_classifier(args, config, device, depth=0):
    from attacks.classifier_clean import ClassifierClean
    # Specify model
    model = ClassifierClean(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac',
                          ),
    ).to(device)
    # Return model
    return model


def define_btb_attack_classifier(args, config, device, depth=0):
    # Import required model
    from attacks.classifier_btb_attack import ClassifierBTBAttack
    # Specify model
    model = ClassifierBTBAttack(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac',
                          ),
        neuron_fraction=args.neuron_fraction,
        trigger_value=args.trigger_value
    ).to(device)
    # Return model
    return model


def define_ltb_attack_classifier(args, config, device, depth=0):
    # Import required model
    from attacks.classifier_ltb_attack import ClassifierLTBAttack
    # Specify model
    model = ClassifierLTBAttack(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac',
                          ),
        neuron_fraction=args.neuron_fraction,
        trigger_value=args.trigger_value,
        attack_context=args.attack_taskid,
        attack_label=CONFIG.DATASET_CONFIGS[args.experiment]['target_vals'][args.attack_taskid - 1]

    ).to(device)
    # Return model
    return model

## Function for defining feature extractor model
def define_feature_extractor(args, config, device):
    # -import required model
    from attacks.feature_extractor import FeatureExtractor
    # -create model
    model = FeatureExtractor(
        image_size=config['size'],
        image_channels=config['channels'],
        # -conv-layers
        conv_type=args.conv_type,
        depth=args.depth,
        start_channels=args.channels,
        reducing_layers=args.rl,
        num_blocks=args.n_blocks,
        conv_bn=True if args.conv_bn=="yes" else False,
        conv_nl=args.conv_nl,
        global_pooling=checkattr(args, 'gp'),
    ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining VAE model
def define_vae(args, config, device, depth=0):
    # Import required model
    from attacks.vae import VAE
    # Specify model
    model = VAE(
        image_size=config['size'],
        image_channels=config['channels'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth > 0 else None,
        start_channels=args.channels if depth > 0 else None,
        reducing_layers=args.rl if depth > 0 else None,
        num_blocks=args.n_blocks if depth > 0 else None,
        conv_bn=(True if args.conv_bn == "yes" else False) if depth > 0 else None,
        conv_nl=args.conv_nl if depth > 0 else None,
        global_pooling=False if depth > 0 else None,
        # -fc-layers
        fc_layers=args.g_fc_lay if hasattr(args, 'g_fc_lay') else args.fc_lay,
        fc_units=args.g_fc_uni if hasattr(args, 'g_fc_uni') else args.fc_units,
        fc_drop=0,
        fc_bn=(args.fc_bn=="yes"),
        fc_nl=args.fc_nl,
        excit_buffer=True,
        # -prior
        prior=args.prior if hasattr(args, "prior") else "standard",
        n_modes=args.n_modes if hasattr(args, "prior") else 1,
        z_dim=args.g_z_dim if hasattr(args, 'g_z_dim') else args.z_dim,
        # -decoder
        recon_loss=args.recon_loss,
        network_output="none" if checkattr(args, "normalize") else "sigmoid",
        deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
    ).to(device)
    # Return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]
def init_params(model, args, verbose=False):

    ## Initialization
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight=="standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias=="standard":
        utils.bias_init(model, strategy="constant", value=0.01)

    ## Use pre-training
    if checkattr(args, "pre_convE") and hasattr(model, 'depth') and model.depth>0:
        load_name = model.convE.name if (
            not hasattr(args, 'convE_ltag') or args.convE_ltag=="none"
        ) else "{}-{}{}".format(model.convE.name, args.convE_ltag,
                                "-s{}".format(args.seed) if checkattr(args, 'seed_to_ltag') else "")
        utils.load_checkpoint(model.convE, model_dir=args.m_dir, name=load_name, verbose=verbose)

    ## Freeze some parameters?
    if checkattr(args, "freeze_convE") and hasattr(model, 'convE'):
        for param in model.convE.parameters():
            param.requires_grad = False
        model.convE.frozen = True #--> so they're set to .eval() duting trainng to ensure batchnorm-params do not change

##-------------------------------------------------------------------------------------------------------------------##
