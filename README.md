Cityscapes

```
python3 segment.py train -d <data_folder> -c 19 -s 832 --arch dla102up \
    --scale 0 --batch-size 16 --lr 0.01 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 10 \
    --random-color --pretrained-base imagenet
```

python train.py --model=dla34 --bs=2