## Real Data Experiments
All files specific to the real data tasks are in this folder

### Structure

- `main.py` handles all the training of models. You can also resume training of a saved run using the `load_gen_path` flag e.g.
```
python main.py --load_gen_path trained_models/news/word/best_mle
```
- `eval_bleu.py` generates BLEU5 and SBLEU5 given a trained model using the `model_path` flag e.g.
```
python eval_bleu.py --model_path trained_models/news/word/best_mle
```
- `score_models.py` generates LM / RLM curves given a trained model using the `model_path` flag e.g.
```
python score_models.py --model_path trained_models/news/word/best_mle
```

### Reproducing Figure 4
We provide pretrained models for the best MLE `(trained_models/news/word/best_mle)`  and GAN model `(trained_models/news/word/best_gan)` You can use the commands above to generate the TensorBoard logs with the desired values. If you wish to plot the actual figure, please have a look at jupyter notebooks in `notebooks/`

#### Rerunning the models 
If you look at the `args.json` file in `trained_models/news/word/best_{mle,gan}` you will find all the required hyperameter values to reproduce the pretrained model. For example, the `best_mle` would be 

```
python main.py --base_dir 'redo_mle' --hidden_dim_gen 512 --num_layers_gen 2 --var_dropout_p_gen 0.5 --mle_epochs 100 --adv_epochs 0 --gen_lr 0.001 --data_dir data/news --max_seq_len 51 --old_model 1 --transfer_weights 0
```
