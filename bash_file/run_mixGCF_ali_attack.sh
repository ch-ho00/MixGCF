
out_dir='./weights/ali_mixgcf_attack'

pretrain_path='./weights/ali_mixgcf_light_gcn_model_.ckpt'

CUDA_VISIBLE_DEVICES=0 python attack.py --dataset ali --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --ns rns --K 1 --n_negs 32 --gnn lightgcn --epoch 20
