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
* Supported Defense: Fine-tune, SAU

* Note: By Default, the models are imported from MindCV (https://github.com/mindspore-lab/mindcv). In case some methods need to modify models, a local copy of models from MindCV is also included in this repo. Change the import part in the code to switch between the local models' folder and models in MindCV.

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

@article{wei2024shared,
  title={Shared adversarial unlearning: Backdoor mitigation by unlearning shared adversarial examples},
  author={Wei, Shaokui and Zhang, Mingda and Zha, Hongyuan and Wu, Baoyuan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
