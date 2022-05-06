# Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving. (CVPR 2022)
<font size=4> Authors: Yi-Nan Chen, [Hang Dai](https://scholar.google.com/citations?hl=en&user=6yvjpQQAAAAJ&view_op=list_works) and Yong Ding \
<font size=3>[\[Paper\]](https://arxiv.org/abs/2203.02112) [\[Supplementary file\]](pdf/supplementary_file.pdf)</font>

![avatar](img/overview.png)

`The code is tested on a ubuntu server with NVIDIA 3090`.

We now release the code for feature-level generation and faeture-clone generation. We apply our methods on the follows stereo-based detectors:
- LIGA-Stereo

    `Step I`: Follow the instruction of [LIGA-Stereo](https://github.com/xy-guo/LIGA-Stereo) to install the dependencies.

    `Step II`: Replace the some files in `LIGA-Stereo` use the files that we provide in [here](stereo_models/LIGA).

    `Step III`: Prepare the data. Please fisrt follow instruction in [LIGA-Stereo](https://github.com/xy-guo/LIGA-Stereo) to prepeare the data. Then download the estimated depth maps by DORN from `here`([training](https://drive.google.com/open?id=1lSJpQ8GUCxRNtWxo0lduYAbWkkXQa2cb), [testing](https://drive.google.com/file/d/1JuDhHGH8DXzNkZSmaVrWyEhI3YuE2GqT/view)). 

    `Step IV`: Training, use the command as follows to train the model. Note that we can only set bacth size to 1 on each GPU in our practice.
    - feature-level generation
    - feature-clone
- YOLOStereo

    `Step I`: Follow the instruction of [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D) to install the dependencies.

    `Step II`: Replace the some files in `visualDet3D` use the files that we provide in [here](stereo_models/YOLOStereo3D).

    `Step III`: Prepare the data. Please fisrt follow instruction in [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D) to prepeare the data. Then download the estimated depth maps by DORN from `here`([training](https://drive.google.com/open?id=1lSJpQ8GUCxRNtWxo0lduYAbWkkXQa2cb), [testing](https://drive.google.com/file/d/1JuDhHGH8DXzNkZSmaVrWyEhI3YuE2GqT/view)).

    - feature-level generation

For image-level generation, we will release the synthesised virtual right iamges.
# Citation
```
@InProceedings{Chen_2022_CVPR,
    title={Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving},
    author={Yi-Nan Chen and Hang Dai and Yong Ding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
}
```
# Acknowledgements
 We would like to thank the repositories as follow:

 [stereo-from-mono](https://github.com/nianticlabs/stereo-from-mono)

 [LIGA-Stereo](https://github.com/xy-guo/LIGA-Stereo)

 [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D)

 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
 
 [D4LCN](https://github.com/dingmyu/D4LCN)


