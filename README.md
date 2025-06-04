
# Dynamic Uncertainty Learning with Noisy Correspondence for Text-based Person Retrieval 

Official PyTorch implementation of the paper  Dynamic Uncertainty Learning with Noisy Correspondence for Text-based Person Retrieval . 
## Updates

- (6/4/2025) Code (This is not the final version.)released!




## Usage
### Requirements
we use single RTX3090 24G GPU for training and evaluation. 
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```


## Training

```python
sh 
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

## IRRA on Text-to-Image Person Retrieval Results
#### CUHK-PEDES dataset

|     Method      |     Backbone     |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |   mINP    |
| :-------------: | :--------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|     CMPM/C      |    RN50/LSTM     |   49.37   |     -     |   79.27   |     -     |     -     |
|      DSSL       |    RN50/BERT     |   59.98   |   80.41   |   87.56   |     -     |     -     |
|      SSAN       |    RN50/LSTM     |   61.37   |   80.15   |   86.73   |     -     |     -     |
|   Han et al.    |  RN101/Xformer   |   64.08   |   81.73   |   88.19   |   60.08   |     -     |
|      LGUR       | DeiT-Small/BERT  |   65.25   |   83.12   |   89.00   |     -     |     -     |
|       IVT       |  ViT-B-16/BERT   |   65.59   |   83.11   |   89.21   |     -     |     -     |
|      CFine      |  ViT-B-16/BERT   |   69.57   |   85.93   |   91.15   |     -     |     -     |
|    **CLIP**     | ViT-B-16/Xformer |   68.19   |   86.47   |   91.47   |   61.12   |   44.86   |
|      **IRRA ** | ViT-B-16/Xformer | **73.38** | **89.93** | **93.71** | **66.13** | **50.24** |
|      **DUNC （ours）** | ViT-B-16/Xformer | |

[Model & log for CUHK-PEDES](https://drive.google.com/file/d/1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW/view?usp=share_link)

#### ICFG-PEDES dataset

|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |   mINP   |
| :-------------: | :-------: | :-------: | :-------: | :-------: | :------: |
|     CMPM/C      |   43.51   |   65.44   |   74.26   |     -     |    -     |
|      SSAN       |   54.23   |   72.63   |   79.53   |     -     |    -     |
|       IVT       |   56.04   |   73.60   |   80.22   |     -     |    -     |
|      CFine      |   60.83   |   76.55   |   82.42   |     -     |    -     |
|    **CLIP**     |   56.74   |   75.72   |   82.26   |   31.84   |   5.03   |
|     **IRRA ** | **63.46** | **80.24** | **85.82** | **38.05** | **7.92** |
|      **DUNC（ours） ** | ViT-B-16/Xformer | |

[Model & log for ICFG-PEDES](https://drive.google.com/file/d/1Y3D7zZsKPpuEHWJ9nVecUW-HaKdjDI9g/view?usp=share_link)

#### RSTPReid dataset

|     Method      |  Rank-1   |  Rank-5   |  Rank-10  |    mAP    |   mINP    |
| :-------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|      DSSL       |   39.05   |   62.60   |   73.95   |     -     |     -     |
|      SSAN       |   43.50   |   67.80   |   77.15   |     -     |     -     |
|       IVT       |   46.70   |   70.00   |   78.80   |     -     |     -     |
|      CFine      |   50.55   |   72.50   |   81.60   |     -     |     -     |
|    **CLIP**     |   54.05   |   80.70   |   88.00   |   43.41   |   22.31   |
| **IRRA ** | **60.20** | **81.30** | **88.20** | **47.17** | **25.28** |
|      **DUNC （ours）** | ViT-B-16/Xformer | |



## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [TextReID](https://github.com/BrandonHanx/TextReID) and [TransReID](https://github.com/damo-cv/TransReID). We sincerely appreciate for their contributions.


## Citation
If you find this code useful for your research, please cite our paper.

```tex

```

## Contact
If you have any question, please feel free to contact us. E-mail: [zqxie@zju.edu.cn].
