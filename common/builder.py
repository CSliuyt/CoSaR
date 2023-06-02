import datetime

from common.utils import *
from common.logger import Logger
from model.utils import *


# ------------------------- net optim scheduler ---------------------------
def build_net_optim_scheduler(params):
    net = build_model(params)
    init_weights(net, init_method=params.init_method)
    if params.opt == 'SGD':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'Adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if params.scheduler_type == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                               verbose=True, threshold=1e-4)
    return net.to(params.device), optimizer, scheduler


# -------------- Build Logger ---------------------
def build_logger(params):
    logger_root = f'Results/{params.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.noise_ratio * 100)
    noise_condition = f'{params.noise_type}_{percentile}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, params.model, 'clean', params.loss_type, noise_condition,  f'{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


# ------------- Build Learning Rate -----------------
def build_lr_plan(params, decay='linear'):
    lr_plan = [params.lr] * params.epochs
    for i in range(0, params.warmup_epochs):
        lr_plan[i] = params.lr
    for i in range(params.warmup_epochs, params.epochs):
        if decay == 'Linear':
            lr_plan[i] = float(params.epochs - i) / (params.epochs - params.warmup_epochs) * params.lr  # linearly decay
        elif decay == 'Cosine':
            lr_plan[i] = 0.5 * params.lr * (1 + math.cos((i - params.warmup_epochs + 1) * math.pi / (params.epochs - params.warmup_epochs + 1)))  # cosine decay
        elif decay == 'Step':
            if params.warmup_epochs <= i < params.warmup_epochs+70:
                lr_plan[i] = params.lr
            elif params.warmup_epochs+70 <= i < params.warmup_epochs+130:
                lr_plan[i] = params.lr * 0.1
            elif params.warmup_epochs+130 <= i < params.warmup_epochs+190:
                lr_plan[i] = params.lr * 0.01
            else:
                lr_plan[i] = params.lr * 0.001
        else:
            raise AssertionError(f'lr decay method: {decay} is not implemented yet.')
    return lr_plan
