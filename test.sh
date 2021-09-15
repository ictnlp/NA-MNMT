export CUDA_VISIBLE_DEVICES=2,3
model=424/test.log
#model=test

python3 train.py data-bin/424/baseline \
    --arch multilingual_transformer \
    --max-epoch 80 --fp16 \
    --encoder-langtok "tgt" \
    --restore-file /data/wanying/1.research/specific/checkpoints/424/baseline/checkpoint36.pt \
    --task multilingual_translation --lang-pairs it-en,ro-en,nl-en,it-ro,en-it,en-ro,en-nl,ro-it \
    --share-encoders --share-decoders \
    --share-all-embeddings --share-decoder-input-output-embed \
    --reset-lr-scheduler --reset-optimizer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0001 --min-lr 1e-09 --ddp-backend=no_c10d \
    --dropout 0.3  \
    --weight-decay 0.0 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096  --update-freq 2 \
    --no-progress-bar --log-format json --log-interval 20 \
    --save-dir checkpoints/$model |tee -a  logs/$model.log
