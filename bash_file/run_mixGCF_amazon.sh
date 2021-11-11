
out_dir='./weights/amazon_mixgcf'

CUDA_VISIBLE_DEVICES=4 python main.py --dataset amazon --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 4 --context_hops 3 --ns mixgcf --K 1 --n_negs 16 --save True --out_dir ${out_dir}

