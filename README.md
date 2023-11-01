# iPUNet

This is the official repository for the publication [iPUNet: Iterative Cross Field Guided Point Cloud Upsampling](https://enigma-li.github.io/projects/iPUNet/iPUNet.html).

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
@article{wei2023ipunet,
  title={iPUNet: Iterative Cross Field Guided Point Cloud Upsampling},
  author={Wei, Guangshun and Pan, Hao and Zhuang, Shaojie and Zhou, Yuanfeng and Li, Changjian},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
```

