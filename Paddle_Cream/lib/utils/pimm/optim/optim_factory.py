'''
本文件为timm.optim.optim_factory的全复制修改
'''

from paddle import optimizer as optim


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        weight_decay /= args.lr
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert False, "fused is not available"

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd':
        optimizer = optim.Momentum(
            learning_rate = args.lr, 
            momentum = args.momentum, 
            parameters = parameters, 
            use_nesterov = True, 
            weight_decay = weight_decay)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            learning_rate = args.lr, 
            epsilon = args.opt_eps, 
            parameters = parameters, 
            weight_decay = weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            learning_rate = args.lr, 
            epsilon = args.opt_eps, 
            parameters = parameters, 
            weight_decay = weight_decay)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            learning_rate = args.ls, 
            epsilon = args.opt_eps, 
            parameters = parameters, 
            weight_decay = weight_decay)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSProp(
            learning_rate = args.lr, 
            rho = .9, 
            epsilon = args.opt_eps, 
            momentum = args.momentum, 
            parameters = parameters, 
            weight_decay = weight_decay)
    elif opt_lower == 'rmsproptf':
        optimizer = optim.RMSProp(
            learning_rate = args.lr, 
            rho = .9, 
            epsilon = args.opt_eps, 
            momentum = args.momentum, 
            parameters = parameters, 
            weight_decay = weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    # if len(opt_split) > 1:
    #     if opt_split[0] == 'lookahead':
    #         optimizer = Lookahead(optimizer)

    return optimizer
