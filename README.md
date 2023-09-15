# MindTrojan
A Collection of Backdoor Learning Resources and Examples with MindSpore

## Quick Start

* Installation 

    ```
    pip install -r requirements.txt
    ```

* Quick demo:

    ```
    bash quick_demo.sh
    ```

* Supported Attacks: BadNets, Blended
* Supported Defense: Fine-tune

* Note: By Default, the models are imported from MindCV (https://github.com/mindspore-lab/mindcv). In case some methods need modify models, a local copy of models from MindCV is also includes in this repo. Change the import part in code to switch between local models folder and models in MindCV.

## Citation

The default settings are in line with BackdoorBench (https://github.com/SCLBD/BackdoorBench) and we refer users to BackdoorBench for more details about the settings.

If interested, you can read our recent works about backdoor learning, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).

```
@inproceedings{backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}

@article{wu2023adversarial,
  title={Adversarial Machine Learning: A Systematic Survey of Backdoor Attack, Weight Attack and Adversarial Example},
  author={Wu, Baoyuan and Liu, Li and Zhu, Zihao and Liu, Qingshan and He, Zhaofeng and Lyu, Siwei},
  journal={arXiv preprint arXiv:2302.09457},
  year={2023}
}

@article{cheng2023tat,
  title={TAT: Targeted backdoor attacks against visual object tracking},
  author={Cheng, Ziyi and Wu, Baoyuan and Zhang, Zhenya and Zhao, Jianjun},
  journal={Pattern Recognition},
  volume={142},
  pages={109629},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{sensitivity-backdoor-defense-nips2022,
 title = {Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples},
 author = {Chen, Weixin and Wu, Baoyuan and Wang, Haoqian},
 booktitle = {Advances in Neural Information Processing Systems},
 volume = {35},
 pages = {9727--9737},
 year = {2022}
}

@inproceedings{dbd-backdoor-defense-iclr2022,
    title={Backdoor Defense via Decoupling the Training Process},
    author={Huang, Kunzhe and Li, Yiming and Wu, Baoyuan and Qin, Zhan and Ren, Kui},
    booktitle={International Conference on Learning Representations},
    year={2022}
}

@inproceedings{ssba-backdoor-attack-iccv2021,
    title={Invisible backdoor attack with sample-specific triggers},
    author={Li, Yuezun and Li, Yiming and Wu, Baoyuan and Li, Longkang and He, Ran and Lyu, Siwei},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={16463--16472},
    year={2021}
}
```