
out_dir='./weights/amazon_attack_'

pretrain_path='./weights/amazon_mixgcf_light_gcn_model_.ckpt'

lambda_a=0.15
epoch=58

python attack.py --dataset amazon --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --ns rns --K 1 --n_negs 16 --save True --out_dir ${out_dir} --gnn lightgcn --lambda_a ${lambda_a} --epoch ${epoch} --pretrain_path ${pretrain_path}

