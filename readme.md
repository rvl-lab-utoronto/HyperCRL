# Extension of HyperCRL

In this fork of the [HyperCRL](https://github.com/rvl-lab-utoronto/HyperCRL) we will extend it with a vision-based state
representation learning (SRL) system.

The aim of our research is to explore how we can use SRL on image data to train model-based RL agents.

Based on the results we will implement an automated task detection approach to infer task boundaries.

***

Following is the slightly altered [ReadMe](https://github.com/rvl-lab-utoronto/HyperCRL/blob/master/readme.md) of the
original repository.

***

# HyperCRL

This is the official implementation
of [Continual Model-Based Reinforcement Learning with Hypernetworks](https://arxiv.org/abs/2009.11997)

## Code Structure

We provide a breakdown of our code structure:

```
  HyperCRL
    hypercrl/
        | -> control/
            | -> agent.py
            | -> cem.py (CEM implementation)
            | -> reward.py (reward function in PyTorch)
        | -> dataset/
            | -> datautil.py
        | -> envs/
            | -> assets/
                | -> door (Modified from DoorGym)
                | -> box.xml
                | -> cartpole.xml
            | -> mujoco/ (Modified from Gym-Extension)
            | -> rs/
                | -> door.py
                | -> generated_objects.py
                | -> push.py
    
        | -> model/
            | -> mbrl.py
            | -> regularizer.py (Modified from Three Scenarios for CL repo)
                | -> EWC/SI implementation
            | -> tools.py

        | -> tools/
            | -> default_arg.py (Hyperparameters and Settings)
            | -> hnet_monitor.py
            | -> tools.py

        | -> hypercl/ (Not our contribution)
            | -> Modified from the original HyperCL repo

        | -> hnet_exp.py (Hnet Main Code)

        | -> lqr_exp.py (Baselines Main Code)

        | -> table_result.py (for calculating forgetting/forward transfer)
    
    robosuite/ (Originally from RoboSuite v1.0 branch)

    scripts/
        | -> Training and ploting scripts
        | -> plot_door_rew.py
        | -> plot_pusher_rew.py
        | -> run.py
        | -> ...

    main.py (main script)

    requirements.txt

    readme.md (this file)
  ```

## Installation

We recommend using a virtualenv to install all the dependencies.

From an Ubuntu 20.04 machine, run the following:

### Python Installation

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y && sudo apt upgrade -y
sudo apt install python3.6 python3.6-dev -y
```

### Python Virtualenv

```
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
python -m pip install -I -r requirements.txt
cd robosuite && pip install -e . --no-use-pep517
cd ..
```

In addition, this repository also requires the MuJoCo physics engine to be installed. Please refer to
the [mujoco-py](https://github.com/openai/mujoco-py) for instructions on how to obtain license and other system
dependencies.

## Run

### Individual Experiment

Use the main python file to start training the RL experiment in any environment

```
python main.py [method_name] [env_name] [robot_name] (optional seed) (optional save_dir) (optional --render)
```

```[method_name]``` can be one of ```hnet```, ```coreset```, ```ewc```, ```si```, ```multitask```, ```single```
or ```finetune```

```[env_name]``` can be one of ```pusher```, ```door_pose```, or ```half_cheetah_body```

```[robot_name]``` can be one of ```Panda``` or ```IIWA```

### Playback Trained Model

Use the main python file to reload a checkpoint and replay a trained agent in a GUI window

```
python main.py [method_name] [env_name] (optional seed) (optional save_dir) --play
```

### Reproduce All Results

To reproduce the result in the paper, the following python scripts (commands) include sub-commands on how to run
experiment using different CL methods (HyperCRL, coreset, SI, ...)

Pusher

```
python scripts/run_pusher.py
```

Door

```
python scripts/run_door.py
```

Half_Cheetah

```
python scripts/run_cheetah.py
```

## Citation

If you find this work or code helpful in your research, please cite:

```
@misc{huang2020continual,
      title={Continual Model-Based Reinforcement Learning with Hypernetworks}, 
      author={Yizhou Huang and Kevin Xie and Homanga Bharadhwaj and Florian Shkurti},
      year={2020},
      eprint={2009.11997},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgments

Please refer to the following github repo for a more detailed description about the original code which are not part of
the contribution of the author's submission.

* [Surreal Robotics Suite](https://github.com/StanfordVL/robosuite)

    * [Paper](http://proceedings.mlr.press/v87/fan18a.html)

* [HyperCL](https://github.com/chrhenning/hypercl)
    * [Paper](https://arxiv.org/abs/1906.00695)

* [Gym Extension](https://github.com/Breakend/gym-extensions)
    * [Paper](https://arxiv.org/abs/1708.04352)

* [DoorGym](https://github.com/PSVL/DoorGym)
    * [Paper](https://arxiv.org/abs/1908.01887)

* [Three Scenarios for Contiual Learning](https://github.com/GMvandeVen/continual-learning)
    * [Paper](https://arxiv.org/abs/1904.07734)
