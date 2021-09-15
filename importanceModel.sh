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
