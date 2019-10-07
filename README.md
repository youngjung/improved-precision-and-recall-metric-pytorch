# improved-precision-and-recall-metric-pytorch

Improved Precision and Recall Metric for Assessing Generative Models - Unofficial Pytorch Implementation

[Paper (arXiv)](https://arxiv.org/abs/1904.06991)

## Usage

- given two directories containing real and fake images
``` bash
python improved_precision_recall.py [path_real] [path_fake]
```

- pre-compute real manifold and save to a file
``` bash
python improved_precision_recall.py [path_real] [dummy_str] --fname_precalc [filename_dest]
```

- if the images are already on memory
```python
ipr = IPR(args.batch_size, args.k, args.num_samples)
ipr.compute_manifold_ref(args.path_real)  # args.path_real can be either directory or pre-computed manifold file
metric = ipr.precision_and_recall(images)
print('precision =', metric.precision)
print('recall =', metric.recall)
```

- realism score
```python
realism_score = ipr.realism(image_in_tensor)
```

## Discussions

- Corner case
  - For A = {999 samples from uniform(0,1)} + {2} and B = {999 samples from uniform(2,3)} + {1},<br/>
precision = 1 and recall = 1.<br/>
  - Outliers can be handled by estimating the quality of individual samples and pruning out.

- Number of samples
  - For A = 1000 real images from celeba_hq and B = 4 images among A,<br/>
precision = 1 and recall = 0.638.<br/>
Wow, 4 images cover 64% of 1000 images!<br/>
  - Manifold estimate becomes inaccurate when number of samples is small.

- Not getting close to 1 given two sets of real images
  - For A = 1000 real images from celeba_hq and B = another 1000 real images from the same dataset, <br/>
precision = 0.639 and recall = 0.661.<br/>
They are not close to 1 even though both set are sampled from the same distribution (=dataset).<br/>
  - It happens because image data in general is extremely sparse.

> We thank Tuomas for enjoyable discussion.

## Link to official repo

https://github.com/kynkaat/improved-precision-and-recall-metric
