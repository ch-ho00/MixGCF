import os
import random

import torch
import torch.nn as nn
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping
from tensorboardX import SummaryWriter
from pathlib import Path

n_users = 0
n_items = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""

    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    
    args.expname = "_".join([str(args.dataset),
                            str(args.pool),
                            str(args.alpha), 
                            str(args.alpha_gt),
                            str(args.neg_mixup),
                            str(args.pos_mixup), 
                            str(args.pos_neg_mixup), 
                            str(args.lambda_mix * 10000), 
                            str(args.lambda_fair * 10000)])

    args.out_dir = Path(os.path.join(args.root_dir, str(args.expnum)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = Path(os.path.join(args.root_dir, str(args.expnum), 'log'))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    global writer_dict
    writer_dict = {
        'writer': SummaryWriter(comment=args.expname,log_dir=tensorboard_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.LightGCN import LightGCN
    from modules.NGCF import NGCF
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)

    if args.ckpt:
        model.load_state_dict(torch.load(f'{args.ckpt}'))    
        print(f"Loaded from {args.ckpt} ###############")
    # print("Use", torch.cuda.device_count(), "GPUs!")
    # model = nn.DataParallel(model, device_ids=[0,1,2], output_device=[args.out_gpu])
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")

    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        loss_meter = AverageMeter()
        emb_meter = AverageMeter()
        mf_meter = AverageMeter()
        mixup_meter = AverageMeter()
        fair_meter = AverageMeter()

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
        # if 0:
            writer_dict['train_global_steps'] = writer_dict['train_global_steps'] + 1
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, mf_loss, emb_loss, mixup_loss, fair_loss = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

            loss_meter.update(batch_loss.item(), args.batch_size)
            mf_meter.update(mf_loss.item(), args.batch_size)
            emb_meter.update(emb_loss.item(), args.batch_size)
            mixup_meter.update(mixup_loss.item(), args.batch_size)
            fair_meter.update(fair_loss.item(), args.batch_size)

            if writer_dict['train_global_steps'] % 100 == 0:
                writer_dict['writer'].add_scalar('train_loss', loss_meter.avg, writer_dict['train_global_steps'])
                writer_dict['writer'].add_scalar('train_mf', mf_meter.avg, writer_dict['train_global_steps'])
                writer_dict['writer'].add_scalar('train_emb', emb_meter.avg, writer_dict['train_global_steps'])
                writer_dict['writer'].add_scalar('train_mixup', mixup_meter.avg, writer_dict['train_global_steps'])
                writer_dict['writer'].add_scalar('train_fair', fair_meter.avg, writer_dict['train_global_steps'])

        train_e_t = time()

        if epoch % 5 == 0:
        # if 1:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])
            
            # writer_dict['writer'].add_scalar('test_recall1', test_ret['recall'][0].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_recall2', test_ret['recall'][1].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_recall3', test_ret['recall'][2].item(), writer_dict['train_global_steps'])

            # writer_dict['writer'].add_scalar('test_ndcg1', test_ret['ndcg'][0].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_ndcg2', test_ret['ndcg'][1].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_ndcg3', test_ret['ndcg'][2].item(), writer_dict['train_global_steps'])

            # writer_dict['writer'].add_scalar('test_hit_ratio1', test_ret['hit_ratio'][0].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_hit_ratio2', test_ret['hit_ratio'][1].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_hit_ratio3', test_ret['hit_ratio'][2].item(), writer_dict['train_global_steps'])

            # writer_dict['writer'].add_scalar('test_precision1', test_ret['precision'][0].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_precision2', test_ret['precision'][1].item(), writer_dict['train_global_steps'])
            # writer_dict['writer'].add_scalar('test_precision3', test_ret['precision'][2].item(), writer_dict['train_global_steps'])
            # if user_dict['valid_user_set'] is None:
            #     valid_ret = test_ret
            # else:
            #     test_s_t = time()
            #     valid_ret = test(model, user_dict, n_params, mode='valid')
            #     test_e_t = time()
            #     train_res.add_row(
            #         [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
            #          valid_ret['precision'], valid_ret['hit_ratio']])

            #     writer_dict['writer'].add_scalar('val_recall', valid_ret['recall'].item(), writer_dict['train_global_steps'])
            #     writer_dict['writer'].add_scalar('val_ndcg', valid_ret['ndcg'].item(), writer_dict['train_global_steps'])
            #     writer_dict['writer'].add_scalar('val_hit_ratio', valid_ret['hit_ratio'].item(), writer_dict['train_global_steps'])
            #     writer_dict['writer'].add_scalar('test_precision', valid_ret['precision'].item(), writer_dict['train_global_steps'])

            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(test_ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if test_ret['recall'][0] == cur_best_pre_0:
                Path(os.path.join(args.out_dir, 'weight')).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'weight',  str(writer_dict['train_global_steps'])+'.ckpt'))

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
