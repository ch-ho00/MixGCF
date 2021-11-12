
out_dir='./weights/yelp_mixgcf_light_gcn_'

CUDA_VISIBLE_DEVICES=1 python main.py --dataset yelp2018 --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 1 --context_hops 3 --ns mixgcf --K 1 --n_negs 64 --save True --out_dir ${out_dir} --gnn lightgcn

