This is the code for running Modification of BPR loss through mix-up and fair regularization.

```bash
CUDA_VISIBLE_DEVICES=6 python3 main.py \
                    --dataset yelp2018 \
                    --alpha 1 --alpha_gt 0 \
                    --neg_mixup 0 --pos_mixup 0 --pos_neg_mixup 1 \ 
                    --random_sample 0.2 \ 
                    --lambda_mix 0.001 --lambda_fair 0.1 \ 
                    --expnum 6
```
  - One can change different settings of setting the target through alpha_gt = 0,1. 
  
  - Alpha is a hyperparameter to randomly sample a mixup coefficient
     (mixup hyper-parameter α controls the strength of interpolation between feature-target)

  - pos_mixup indicates mixup between positive items / Option : [0,1]

  - neg_mixup indicates mixup between negative items / Option : [0,1]

  - pos_neg_mixup indicates mixup between positive and negative items / Option : [0,1]

  - random_sample is a value between (0,1) selecting the ratio of numbers of embeddings to mix up.

  - Alpha_gt indicates whether the ground truth of the pos-neg mixup is alpha instead / Option : any positive value

  - lambda_mix and lambda_fair represent the weights on each loss.
