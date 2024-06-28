# DQN-based Ethical Decision-Making for Self-Driving Cars

This repository contains the code for the paper **"DQN-based ethical decision-making for self-driving cars in unavoidable crashes: An applied ethical knob."**


## Description

This project focuses on developing a Deep Q-Network (DQN) based approach for ethical decision-making in self-driving cars during unavoidable crashes. It implements the concept of an **ethical knob with 9 levels, ranging from altruism to egoism**, allowing the adjustment of ethical considerations in real-time decision-making processes.


## How to run

This code is optimized for versions 0.9.12 to 0.9.15 of the CARLA simulator. Download the CARLA simulator from the [CARLA releases download page](https://github.com/carla-simulator/carla/blob/master/Docs/download.md) based on your operating system (Linux and Windows supported).

**Caution:** Since the vehicle physics model is crucial for the proper functioning of the agent, it is highly recommended to use a different vehicle dynamics model instead of the default CARLA PhysX engine. [CarSim](https://carla.readthedocs.io/en/latest/tuto_G_carsim_integration/) or [ProjectChrono](https://carla.readthedocs.io/en/latest/tuto_G_chrono/) integration is recommended. The Chrono integration is implemented in this project. To activate it, a built-from-source CARLA is needed, as the packaged version does not support this feature. Using the packaged version with the default PhysX engine may result in unusual acceleration parameters.

- Create an Anaconda environment via: `conda create -n edm-dqn python=3.10.11`
- Activate the Anaconda environment via: `conda activate edm-dqn`
- Install the required packages via: `pip install -r requirements.txt`
- Install the CARLA Python package based on your downloaded version of CARLA via: `pip install carla==0.9.your_version`
- Set the desired (hyper)parameters, including k-value, via: `config.py`
- Train the agent for the set k-value via: `train.py`
- Test the trained agent for different k-values via: `test.py`


## Examples

The training process is done in random scenarios to ensure good generalization for the agent. Some training episodes are shown below:

<p align="center"><img src="assets/training.gif" width="800" title="Some training episode samples (Increased playback speed)" alt=""></p>

The evaluation is done in 2 scenarios that remain unseen during the training phase. The video below shows how the trained agent acts in the first and second scenarios for k=0.1, 0.5, and 0.9.

<p align="center"><img src="assets/testing.gif" width="1274" title="Testing for different k values" alt=""></p>


## Hardware requirements

It is highly recommended to use a GPU with at least 8 GB of dedicated memory (e.g., RTX 30 or 40 series) for [running CARLA](https://carla.readthedocs.io/en/latest/start_quickstart/#before-you-begin) and training using TensorFlow. Also, at least 60 GB of hard disk space is required for saving model checkpoints and buffers at each training run.


## Citations

If you find this repository helpful and decided to utilize it in your research project, consider citing us as below:
```bibtex
@article{VAKILI2024124569,
title = {DQN-based ethical decision-making for self-driving cars in unavoidable crashes: An applied ethical knob},
journal = {Expert Systems with Applications},
volume = {255},
pages = {124569},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.124569},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424014362},
author = {Ehsan Vakili and Abdollah Amirkhani and Behrooz Mashadi},
keywords = {Ethical decision-making, Unavoidable crash, Injury, Self-driving car, Reinforcement learning},
abstract = {Ethical decision-making in complex urban driving scenarios remains a challenge, particularly when human lives are at risk. This paper presents an ethical decision-making model for self-driving cars in critical urban traffic situations, utilizing deep reinforcement learning (DQN algorithm). The primary objective is to minimize injuries resulting from self-driving car crashes. Crash injury severity prediction models are incorporated to design a proper reward function as well as accounting for the age groups of individuals involved in the crash. It explores seven levels from egoism to altruism, investigating the adjustability and sensitivity of the “ethical knob” concept, as well as the responsibilities of manufacturers, potential risks to passengers and other individuals at the crash scene. The model also incorporates constraints on vehicle longitudinal and lateral accelerations to ensure appropriate vehicle movement toward the collision target. Results indicate that the agent attempts to avoid hazardous situations whenever possible, prioritizing minor injuries when escape is impossible. In the most egoistic mode, the agent prioritizes passenger safety, while the highest level of altruism allows for potential harm to passengers to save those outside the vehicle. The results of the uncertainty analysis indicate that the model presented in this study is robust to noise in the environmental input and dropout in the convolutional neural network; although it should be adjusted for more sensitive conditions. The proposed model offers a practical framework for ethical decision-making in self-driving cars. The code is available at: https://github.com/ehsan-vakili/DQN-ethical-decision-making-self-driving-cars.}
}
```


## License

This code is released under the MIT license, see [LICENSE.md](LICENSE.md) for details.
