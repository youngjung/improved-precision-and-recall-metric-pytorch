# improved-precision-and-recall-metric-pytorch
Improved Precision and Recall Metric for Assessing Generative Models - Unofficial Pytorch Implementation

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
