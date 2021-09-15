# NA-MNMT
Source code for the ACL 2021 paper Importance-based Neuron Allocation for Multilingual Neural Machine Translation.

## Related code

Implemented based on [Fairseq-py](https://github.com/pytorch/fairseq), an open-source toolkit released by Facebook which was implemented strictly referring to [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).

## Requirements
This system has been tested in the following environment.
+ OS: Ubuntu 16.04.1 LTS 64 bits
+ Fairseq = v0.9.0 
+ Python version \>=3.6
+ Pytorch version \>=1.1

## Get started

- Preprocess the training data. Pretrain the baseline model with your data. Read [here](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) for more instructions.

- Evaluate the importance of the neurons. 

Run it for each language pair to calculate the importance.

```
bash importanceModel.sh
```

or

```
#!/bin/bash 
save_dir={checkpoints}/model.pt
langs=lang1,lang2,lang3,lang4 # All languages pairs of your model such as "en-zh,zh-en,en-ar,ar-en"
lang=lang1 # Current language pair, such as "en-zh"

python importance.py data-bin/{data} \
       --arch multilingual_transformer  --reset-optimizer \
       --encoder-langtok "tgt" \
       --task multilingual_translation --lang-pairs $langs \
       --share-encoders --share-decoders \
       --share-all-embeddings \
       --focus-lang $lang --fp16 \
       --max-tokens 2048  --save-dir $save_dir
```

- Generate mask matrix based on the importance value
  
```
python importance_mask.py
```
  
  
- Fine-tuning the model with language-specific mask matrix

```
bash run.sh
```

or

```
# !/bin/bash
model={path_to_ckpt}
#model=test

python3 train.py data-bin/{data}} \
    --arch multilingual_transformer \
    --fp16 \
    --encoder-langtok "tgt" \
    --restore-file /path_to_baseline/model.pt \
    --task multilingual_translation --lang-pairs $langs \
    --share-encoders --share-decoders \
    --share-all-embeddings --share-decoder-input-output-embed \
    --reset-lr-scheduler --reset-optimizer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0001 --min-lr 1e-09 --ddp-backend=no_c10d \
    --dropout 0.1 \
    --weight-decay 0.0 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096  --update-freq 2 \
    --no-progress-bar --log-format json --log-interval 20 \
    --save-dir checkpoints/$model |tee -a  logs/$model.log
```

- Generate the language-specific translation 

```
# !/bin/bash
t=$src
f=$tgt
cat tst.$t-$f.$t \
python interactive.py $DATABIN --path $model_path \
        --task multilingual_translation --source-lang $t --target-lang $f \
        --encoder-langtok "tgt" \
        --buffer-size 2000 --lang-pairs $langs \
        --beam 4 --batch-size 128 --lenpen 0.6 --remove-bpe \
        --log-format=none > $OUT_DIR/tst.$t-$f
```


## Citation
```
@inproceedings{XieFGY21,
  author    = {Wanying Xie and
               Yang Feng and
               Shuhao Gu and
               Dong Yu},
  title     = {Importance-based Neuron Allocation for Multilingual Neural Machine
               Translation},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
               Linguistics and the 11th International Joint Conference on Natural
               Language Processing, {ACL/IJCNLP} 2021, (Volume 1: Long Papers), Virtual
               Event, August 1-6, 2021},
  pages     = {5725--5737},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.acl-long.445},
}
```
