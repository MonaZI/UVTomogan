# UVTomogan

Implementation for the paper: 

UVTomo-gan: An adversarial learning based approach for unknown view X-ray tomographic reconstruction, accepted in ISBI 2021

by Mona Zehni, Zhizhen Zhao

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
@ARTICLE{UVTomogan,
       author = {{Zehni}, Mona and {Zhao}, Zhizhen},
        title = "{UVTomo-GAN: An adversarial learning based approach for unknown view X-ray tomographic reconstruction}",
      journal = {arXiv e-prints},
         year = 2021,
        month = feb,
        pages = {arXiv:2102.04590},
       eprint = {2102.04590}}
```
If you have any questions, please contact Mona Zehni (mzehni2@illinois.edu).
