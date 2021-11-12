
out_dir='./weights/ali_mixgcf_light_gcn_'

CUDA_VISIBLE_DEVICES=4 python main.py --dataset ali --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 4 --context_hops 3 --ns mixgcf --K 1 --n_negs 32 --save True --out_dir ${out_dir} --gnn lightgcn
