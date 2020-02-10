## Real Data Experiments
All files specific to the real data tasks are in this folder

### Structure

- `main.py` handles all the training of models. You can also load a pretrained model using the following flag 
```
python main.py --load_gen_path trained_models/news/word/best_mle
```
- `eval_bleu.py` generates BLEU5 and SBLEU5 given a trained model e.g.
```
python eval_bleu.py --load_gen_path trained_models/news/word/best_mle
```
- `score_models.py` generates LM / RLM curves given a trained model e.g.
```
python score_models.py --load_gen_path trained_models/news/word/best_mle
```

### Reproducing Figure 4
We provide pretrained models for the best MLE `(trained_models/news/word/best_mle)`  and GAN model `(trained_models/news/word/best_gan)` You can use the commands above to generate the TensorBoard logs with the desired values. If you wish to plot the actual figure, please have a look at jupyter notebooks in `notebooks/`

