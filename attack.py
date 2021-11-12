import os
import random

import torch
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

import torch.nn as nn

n_users = 0
n_items = 0

class Attack(nn.Module):
    def __init__(self, n_users, emb_size):
        super(Attack, self).__init__()
        initializer = nn.init.xavier_uniform_
        attack_e_u = initializer(torch.empty(n_users, emb_size))
        self.attack_e_u = nn.Parameter(attack_e_u)

    def forward(self,x):
        return x + self.attack_e_u



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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """Add the wandb weight and bias to record the experiments"""
    import wandb

    run = wandb.init(config=args,
                     project='COMP5331-MixGCF-Attack',
                     name='attack_lightGCN'+args.dataset+'_restrict',
                     entity="xingzhi"
                     )

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

    # load the well train model
    # pretrain_path = './weight/XX.ct'
    if args.pretrain_path is not None:
        model.load_state_dict(torch.load(args.pretrain_path))
        print('Successfully loading the previous model at', args.pretrain_path)

    # define attack parameters:

    attack = Attack(model.n_users, model.emb_size)
    attack.to(device)

    model.user_embed_init = torch.empty(model.n_users, model.emb_size).to(device)
    model.user_embed_init.data.copy_(model.user_embed_init.data)

    """define optimizer"""
    # update the attack only
    optimizer = torch.optim.Adam(attack.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False


    # # """Test before training"""
    # #
    # # """testing"""
    #
    # train_res = PrettyTable()
    # train_res.field_names = ["Start", "recall", "ndcg", "precision",
    #                          "hit_ratio"]
    #
    # model.eval()
    # test_s_t = time()
    # test_ret = test(model, user_dict, n_params, mode='test')
    # test_e_t = time()
    # train_res.add_row(
    #     ['test start', test_ret['recall'], test_ret['ndcg'],
    #      test_ret['precision'], test_ret['hit_ratio']])
    #
    # wandb.log({'epoch': -1,
    #            'loss': 0,
    #            'recall_20': test_ret['recall'][0],
    #            'recall_40': test_ret['recall'][1],
    #            'recall_60': test_ret['recall'][2],
    #            'ndcg_20': test_ret['ndcg'][0],
    #            'ndcg_40': test_ret['ndcg'][1],
    #            'ndcg_60': test_ret['ndcg'][2],
    #            'precision_20': test_ret['precision'][0],
    #            'precision_40': test_ret['precision'][1],
    #            'precision_60': test_ret['precision'][2],
    #            'hit_ratio_20': test_ret['hit_ratio'][0],
    #            'hit_ratio_40': test_ret['hit_ratio'][1],
    #            'hit_ratio_60': test_ret['hit_ratio'][2]})
    #
    # if user_dict['valid_user_set'] is None:
    #     valid_ret = test_ret
    # else:
    #     test_s_t = time()
    #     valid_ret = test(model, user_dict, n_params, mode='valid')
    #     test_e_t = time()
    #     train_res.add_row(
    #         ['valid start', valid_ret['recall'], valid_ret['ndcg'],
    #          valid_ret['precision'], valid_ret['hit_ratio']])
    # print(train_res)


    print("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        # model.train()
        model.eval()



        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            # add the attack in the model
            del model.user_embed
            # model.user_embed = attack(model.user_embed_init.detach())
            normalized_attack_e_u = attack.attack_e_u / torch.norm(attack.attack_e_u.detach(), dim=1, keepdim=True)
            user_embed = model.user_embed_init.detach()
            model.user_embed = user_embed + args.lambda_a * user_embed.norm(dim=1, keepdim=True) * normalized_attack_e_u


            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, _, _ = model(batch)

            batch_loss = -batch_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

            # del normalized_attack_e_u, user_embed

        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])

            ''' log the result to weight and bias wandb'''
            wandb.log({'epoch':epoch,
                       'loss':loss.item(),
                       'recall_20':test_ret['recall'][0],
                       'recall_40':test_ret['recall'][1],
                       'recall_60':test_ret['recall'][2],
                       'ndcg_20':test_ret['ndcg'][0],
                       'ndcg_40':test_ret['ndcg'][1],
                       'ndcg_60':test_ret['ndcg'][2],
                       'precision_20':test_ret['precision'][0],
                       'precision_40':test_ret['precision'][1],
                       'precision_60':test_ret['precision'][2],
                       'hit_ratio_20':test_ret['hit_ratio'][0],
                       'hit_ratio_40':test_ret['hit_ratio'][1],
                       'hit_ratio_60':test_ret['hit_ratio'][2]})

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)

            # # *********************************************************
            # # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            # cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0,
            #                                                             stopping_step, expected_order='acc',
            #                                                             flag_step=10)
            # if should_stop:
            #     break
            #
            # """save weight"""
            # if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
            #     os.makedirs(args.out_dir, exist_ok=True)
            #     # torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
            #     '''save attack model'''
            #     torch.save(attack.state_dict(), args.out_dir + 'attack_' + '.ckpt')

            # just save the model
            if args.save:
                os.makedirs(args.out_dir, exist_ok=True)
                '''save attack model'''
                torch.save(attack.state_dict(), args.out_dir + 'attack_' + '.ckpt')
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    run.finish()