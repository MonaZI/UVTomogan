# UVTomogan

Implementation for the paper: 

UVTomo-gan: An adversarial learning based approach for unknown view X-ray tomographic reconstruction, accepted in ISBI 2021

by Mona Zehni, Zhizhen Zhao

Link to paper: https://ieeexplore.ieee.org/document/9433970
Link to paper (arxiv): https://arxiv.org/abs/2102.04590

## Prerequisites
- Astra toolbox (https://www.astra-toolbox.com/)
- Pytorch, Numpy, Matplotlib, pyyaml
- Optional: GlobalBioIm (https://biomedical-imaging-group.github.io/GlobalBioIm/), used for the baselines.

## Run
The config files for different experiments are located in ```./configs/```.

Pass the general and experiment's config file as:

```r
python run_2d.py -config_gen ./configs/config_gen.yaml -config_exp ./configs/config_phantom_known_clean.yaml
``` 

## More Information
If you find this repositry helpful in your publications, please consider citing our paper.
```r
@INPROCEEDINGS{uvtomogan,  
               author={Zehni, Mona and Zhao, Zhizhen},  
               booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},   
               title={UVTOMO-GAN: An Adversarial Learning Based Approach For Unknown View X-Ray Tomographic Reconstruction},   
               year={2021},  
               volume={},  
               number={},  
               pages={1812-1816},  
               doi={10.1109/ISBI48211.2021.9433970}}
```
If you have any questions, please contact Mona Zehni (mzehni2@illinois.edu).
