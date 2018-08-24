# repository for Exposure Bias experiments


## What we need for AAAI -- sep 5th (decreasing order)

1) LM score vs RLM score w.r.t temperatures for MLE and GANs with multiple CV techniques :: News dataset

2) repeat 1) with LeakGAN and Leak MLE

3) repeat 1) & 2) on a new dataset (ptb or coco)

4) redo a Figure similar to 1) but in the synthetic data case (NLL_oracle vs NLL_test) 

5) repeat 1) & 2) in the conditional setting i.e. Sentence Completion task

6) repeat 1) & 2) w.r.t Adversarial Epochs

#### Bonus

Find an experiment that could shine light on why Adv training achieves a worse tradeoff than MLE
e.g. Use synthetic data were you can control the (optimal) Discriminator and the use of REINFORCE
hypothesis is that, nothing in the adv training pushes the Generator to be diverse


## Life after AAAI




