# COMP 5331 Project Adversarial Learning

This is the pytorch implementation for the project on robustness learning to MixGCF.



## Pretrain

```bash
bash run_mixGCF_ali.sh 

bash run_mixGCF_amazon.sh

bash run_mixGCF_yelp.sh
```



## Adversarial Learning

After getting the pretrain model from the original model, we can start to train the adversarial model.

```bash
bash run_mixGCF_ali_attack.sh
bash run_mixGCF_amazon_attack.sh
bash run_mixGCF_yelp_attack.sh
```

You can adjust the $\lambda_a$ for attack extent. 



## Weight and biases

We use weight and biases to record the data. You can check the tutorial for weight and biases for usages: [weight and biases](https://wandb.ai/site/tutorials).

