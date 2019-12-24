
## Segmentation and Boundary Prediction

Segmentation and boundary prediction data format is the same as
[DRN](https://github.com/fyu/drn#prepare-data).

To use `--bn-sync`, please include `lib` in `PYTHONPATH`.

Cityscapes

```
python3 segment.py train -d <data_folder> -c 19 -s 832 --arch dla102up \
    --scale 0 --batch-size 16 --lr 0.01 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 10 \
    --random-color --pretrained-base imagenet
```


## FAQ

*How many GPUs does the program require for training?*

We tested all the training on GPUs with at least 12 GB memory. We usually tried to use fewest GPUs for the batch sizes. So the actually number of required GPUs is different between models, depending on the model sizes. Some model training may require 8 GPUs, such as training `dla102up` on Cityscapes dataset.