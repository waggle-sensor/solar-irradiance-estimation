import torch


def prepare_optim(opts, model):
    optim = torch.optim.SGD(model.parameters(),
                            lr=opts.cfg['lr'],
                            momentum=opts.cfg['momentum'],
                            weight_decay=opts.cfg['weight_decay'])
    if opts.cfg['pretrained_net']:
        print('opts.cfg[pretrained_net]', opts.cfg['pretrained_net'])
        checkpoint = torch.load(opts.cfg['pretrained_net'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim
