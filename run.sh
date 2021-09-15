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