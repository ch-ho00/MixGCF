import itertools

hparam_grid = {
    'dataset' : ['yelp2018'],
    'dim' : [64],
    'lr' : [1e-4], 
    'l2' : [1e-5],
    'batch_size' : [4096],
    'context_hops' : [3],
    'ns' : ['mixgcf'],
    'K' : [1],
    'n_negs' : [32],
    'epoch' : [100],
    'fair' : [1],
    'mixup' : [1],
    'pool' : ['concat'],
    'alpha' : [1],
    'alpha_gt' : [0, 1],
    'neg_mixup' : [0, 1],
    'pos_mixup' : [0, 1],
    'pos_neg_mixup'	: [0, 1],
    'random_sample' : [0.2],
    'lambda_mix' : [1e-1, 1e-3], 
    'lambda_fair' :  [1e-1, 1e-3],

}

keys =  list(hparam_grid.keys())
lists = [hparam_grid[k] for k in keys]
permutation = list(itertools.product(*lists))
expnames = [ " ".join([f"--{key} {value}" for key, value in zip(keys, p)]) for p in permutation]
expnames = [ "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 python3 main.py " + expname + f" --expnum {i}" for i, expname in enumerate(expnames)]

for e in expnames:
    print(e)