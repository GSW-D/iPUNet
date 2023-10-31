# iPUNet

This is the official repository for the publication [iPUNet: Iterative Cross Field Guided Point Cloud Upsampling](https://arxiv.org/pdf/2310.09092.pdf).

If you should encounter any problems with the code, don't hesitate to contact me at guangshunwei@gmail.com.

# Installation

    git clone https://github.com/GSW-D/iPUNet.git
    cd iPUNet-main
    conda env create -f iPUNet.yml
    conda activate iPUNet

## Dependencies

- [chamfer_distance]
- [KNN]
- [PointNet2] custom ops must be built by running python setup.py install under pointnet2_utils/custom_ops folder.



## Download
Dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1L3qa9wGGTWX3ZLG2-31StWZPru4-iwLd).

### Training
```
python train.py
```

### Testing
```
python test.py
```

## Citation
```
@misc{wei2023ipunetiterative,
      title={iPUNet:Iterative Cross Field Guided Point Cloud Upsampling}, 
      author={Guangshun Wei and Hao Pan and Shaojie Zhuang and Yuanfeng Zhou and Changjian Li},
      year={2023},
      eprint={2310.09092},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

