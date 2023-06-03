import argparse
import time

from torch import optim

from common.builder import *
from common.loss import loss_Jo_AR, get_js_avg, get_js_list
from dataset import data_loader
from model.cnn import CNN


def parser_args():
    parser = argparse.ArgumentParser()

    # ------------------- Init-related --------------------
    parser.add_argument('--random_seed', type=int, default=0, help='set the random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Set the GPU')
    parser.add_argument('--prefetch', type=int, default=1, help='Set the workers num')
    parser.add_argument('--log_freq', type=int, default=1)

    # -------------------   Dataset   ---------------------
    parser.add_argument('--config', type=str, default='config/cifar100')
    # parser.add_argument('--dataset', type=str, default='cifar10d', help="[cifar10d/100, tinyimagenet, clothing1m]")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--aux_dataset', type=str, default='imageNet32', help="cifar : imageNet32, "
                                                                              "tinyimagenet: imageNet64")
    parser.add_argument('--aux_batch_size', type=int, default=128)
    parser.add_argument('--aux_num_samples', type=int, default=50000)
    parser.add_argument('--aux_num_to_avg', type=int, default=1, help='cifar: 1, tinyimagenet: 2')

    # ------------------- Train-related -------------------
    # Model
    parser.add_argument('--model', type=str, default='wrn', help='resnet18/34/50 and wrn')
    parser.add_argument('--init_method', type=str, default='Xavier', help='Xavier or He')

    # Noise
    parser.add_argument('--noise_type', type=str, default='asymmetric',
                        help='symmetric or asymmetric, openset, instance')
    parser.add_argument('--noise_ratio', type=float, default=0.8, help='Set the noisy ratio')

    # Optim
    parser.add_argument('--opt', type=str, default='SGD', help='optim method : SGD or Adam')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--scheduler_type', type=str, default=' ', help='Set the leaning scheduler type')

    # Loss method
    parser.add_argument('--loss_type', type=str, default='soft', help="ce, hard, soft, adaptive")

    # Learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr-decay', type=str, default='Step')
    parser.add_argument('--warmup-lr-scale', type=float, default=10.0)

    # Epochs
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs')
    parser.add_argument('--epochs', type=int, default=200, help='train epochs')

    args = parser.parse_args()
    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])

    print(config)
    return config


def adjust_learning_rate(optimizer, epoch, al_plan, be_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = al_plan[epoch]
        param_group['betas'] = (be_plan[epoch], 0.999)  # Only change beta1


def main(cfg):
    # Set the random seed
    init_seeds(cfg.random_seed)

    # Get device
    device = cfg.device
    print(f'Train on: {device}')

    # Build Logger
    logger, result_dir = build_logger(cfg)

    # Get net optim scheduler
    net = CNN(n_outputs=cfg.num_classes).to(device)
    net2 = CNN(n_outputs=cfg.num_classes).to(device)
    init_weights(net, init_method='He')
    init_weights(net2, init_method='He')
    optimizer = optim.Adam(net.parameters(), cfg.lr)
    optimizer2 = optim.Adam(net2.parameters(), cfg.lr)

    # Set learing rate
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [cfg.lr] * 200
    beta1_plan = [mom1] * 200

    for i in range(80, 200):
        alpha_plan[i] = float(200 - i) / (200 - 80) * cfg.lr
        beta1_plan[i] = mom2

    # Set the dataloader
    train_loader, valid_loader, n_train_samples, n_validating_samples = data_loader.build_dataloader(cfg.dataset, cfg)

    logger.msg(f"Categories: {cfg.num_classes}, Training Samples: {n_train_samples},"
               f" Valid Samples: {n_validating_samples}, Model: {cfg.model}")
    logger.msg(f"Noise Type: {cfg.noise_type}, Noise Ratio: {cfg.noise_ratio}")
    logger.msg(f'Optimizer: {cfg.opt}')

    # ----------------loss----------------
    loss_function = loss_Jo_AR

    # ----------------- meter ---------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()
    best_accuracy, last_accuracy, best_epoch = 0.0, 0.0, None
    best_accuracy2, last_accuracy2, best_epoch2 = 0.0, 0.0, None
    iters_to_accumulate = round(64 / cfg.batch_size) if cfg.batch_size < 64 else 1
    logger.msg(
        f'Accumulate gradients every {iters_to_accumulate} iterations --> Acutal batch size is {cfg.batch_size * iters_to_accumulate}')

    js_avg = torch.tensor(1.).to(device)
    # ------------------ training -------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()

        net.train()
        net2.train()
        adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan)
        adjust_learning_rate(optimizer2, epoch, alpha_plan, beta1_plan)

        optimizer.zero_grad()
        optimizer2.zero_grad()

        train_loss.reset()
        train_accuracy.reset()

        JSD = torch.zeros(n_train_samples).detach()
        clean_num = 0
        selection_num = 0
        s = time.time()

        # Train this epoch
        pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training', total=len(train_loader))
        for it, in_data in enumerate(pbar):
            curr_lr = [group['lr'] for group in optimizer.param_groups][0]

            # Divided data
            in_X_weak, in_X_strong = in_data[0].to(device), in_data[1].to(device)
            target, target_gt = in_data[2].long().to(device), in_data[3].long().to(device)

            # predictions
            prob_weak = net(in_X_weak)
            prob_strong = net(in_X_strong)
            net2_prob = net2(in_X_weak).detach()

            js_list = get_js_list(prob_weak.softmax(dim=1), target).detach()
            js_list2 = get_js_list(prob_weak.softmax(dim=1), target).detach()

            # get the accuracy of this step
            train_acc = accuracy(prob_weak.softmax(dim=1), target_gt)
            pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')

            # update the model
            if epoch < cfg.warmup_epochs:
                pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')
                loss_all = F.cross_entropy(prob_weak, target)
                now_index = torch.nonzero(JSD == 0)[0]
                JSD[now_index:(in_X_weak.size(0) + now_index)] = js_list
                clean_list = js_list < js_avg
                if sum(clean_list) != 0:
                    loss2 = F.cross_entropy(net2(in_X_weak[clean_list]), target[clean_list]) + F.cross_entropy(net2(in_X_strong[clean_list]), target[clean_list])
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

            else:
                pbar.set_description(f'ROBUST TRAINING (lr={curr_lr:.3e})')
                loss_all, js_list_cor, clean_list = loss_function(prob_weak, prob_strong, target, js_avg, js_list, js_list2, net2_prob)
                now_index = torch.nonzero(JSD == 0)[0]
                JSD[now_index:(in_X_weak.size(0) + now_index)] = js_list_cor
                if sum(clean_list) != 0:
                    clean_samples_weak, clean_sample_strong, clean_target = in_X_weak[clean_list], in_X_strong[clean_list], target[clean_list]
                    prob_weak_2 = net2(clean_samples_weak)
                    prob_strong_2 = net2(clean_sample_strong)
                    loss2 = F.cross_entropy(prob_weak_2, clean_target) + F.cross_entropy(prob_strong_2, clean_target)
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            selection_num += sum(clean_list)
            clean_num += sum(target[clean_list] == target_gt[clean_list])

            train_accuracy.update(train_acc[0], cfg.batch_size)
            train_loss.update(loss_all.item(), cfg.batch_size)
            epoch_train_time.update(time.time() - s, 1)

            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                mem = torch.cuda.memory_reserved() / 2 ** 30
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        eval_result = evaluate(valid_loader, net, device)
        test_accuracy = eval_result['accuracy']
        test_loss = eval_result['loss']
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

        eval_result2 = evaluate(valid_loader, net2, device)
        test_accuracy2 = eval_result2['accuracy']
        test_loss2 = eval_result2['loss']
        if test_accuracy2 > best_accuracy2:
            best_accuracy2 = test_accuracy2
            best_epoch2 = epoch + 1

        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test loss: {test_loss:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d} |'
                    f'student loss: {test_loss2:>6.4f} | '
                    f'student acc: {test_accuracy2:>6.3f} | '
                    f'best accuracy: {best_accuracy2:6.3f} @ epoch: {best_epoch2:03d} |' 
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'selected clean rate: {clean_num/(selection_num+1e-8):6.3f} | '
                    f'clean num: {clean_num} | '
                    f'selected num: {selection_num},'
                    )

        js_avg = get_js_avg(JSD).detach().to(device)
        print(js_avg, min(JSD))


if __name__ == '__main__':
    cfg = parser_args()
    noise_rates = [0.8, 0.6, 0.2, 0.4]
    for rate in noise_rates:
        cfg.noise_ratio = rate
        main(cfg)
