
out_dir='./weights/yelp_attack_'

pretrain_path='./weights/yelp_mixgcf_light_gcn_model_.ckpt'
lambda_a=0.05
epoch=56

python attack.py --dataset yelp2018 --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 2 --context_hops 3 --ns rns --K 1 --n_negs 64 --save True --out_dir ${out_dir} --gnn lightgcn --lambda_a ${lambda_a} --epoch ${epoch} --pretrain_path ${pretrain_path}

