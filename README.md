# ActiveCar
#### An Individual Project Completed by Rhys Stewart

---

## Introduction

Welcome to ActiveCar. This repository serves the basis for research and investigation for the fifth year individual project at the University of Glasgow under the Computing Science Degree. This project is worth 60 credits. For more information, visit the [course catalogue](https://www.gla.ac.uk/coursecatalogue/course/?code=COMPSCI5073P). 

---

*Side Note:*
> 

---

Before diving into the content, hypothesis, or the report (dissertation), it is important to understand what is being researched here. This will be discussed in the section below, but first some pre-requisites. 


### What is Active Vision?

>Active vision is a concept inspired by human biology; the idea that an observer can solve visual problems much more efficiently through dynamic movement of their sensory apparatus [1, 11] (in simpler terms, e.g., turning your head to see something more clearly) mirroring the biological "look to act" principles of human visual behaviour [4]

### What is Gymnasium?

>Gymnasium provides a standardised, interoperable API for reinforcement learning environments [8]. It ensures that the underlying RL algorithms can interact with the custom DOOM scenarios in a reproducible manner through defining strict observation and action spaces.


### What is Stable-Baselines3?

>SB3 is an open-source library build on top of the RL framework PyTorch, that provides highly reliable, benchmarked implementations of modern Deep Reinforcement Learning (DRL) algorithms [9]. 

---

## Purpose and Aims

The purpose of this project is to answer the following research question: "*Does an active vision approach to autonomous vehicle perception result in significant improvement in task performance in contrast to the current industry standards (that which are used currently to provide autonomous vehicles perception)?*"


The aims can then be derived as follows:

- **Environment Development**: 

- **Agent Training**: Implement and train a baseline model and an active vision model. 

- **Comparative Evaluation**: Evaluate and compare the efficacy of the active visual sensor against the baseline model across varied levels of environmental complexity, visual occlusion, and restricted fields of view.

- **Generalisation Analysis**: Assess the generalisability of the learned active vision policies in unseen, dynamic environments, addressing a core challenge in current DRL literature. 


### How Will this be Achieved?

The project is going to be developed in the following ways:

- 

---

## File Structure and What to Expect

This project has the following directory structures:

```
ActiveCar/
├── data/
│   ├── Graphs
│   │   ├── /                               # 
│   │   └── /                               # 
│   │ 
│   ├── README.md                           # Description of data sources
│   ├── /                                   # 
│   ├── notebooks/                          # Jupyter Notebooks and Python Scripts used to visualise data from evaluations
│   │
│   ├── visualisations/                     # 
│   │   ├── /                               # 
│   │   ├── /                               # 
│   │   ├── /                               # 
│   │   └── /                               # 
│   │ 
│   └── /                                   # 
│
├── meetings/
│   └── README.md                           # Weekly supervisor meeting minutes
│
├── presentation/
│   ├── slides/                             # PowerPoint and PDF presentations
│   ├── figures/                            # Images and diagrams
│   └── references.md                       # Slide bibliography
│
├── src/
│   ├── /                                   # 
│   │   └──                                 # 
│   │
│   ├── /
│   │   └── /
│   │       ├── /                           # 
│   │       │   ├── /
│   │       │   ├── /
│   │       │   └── /
│   │       │
│   │       │
│   │       │
│   │       │
│   │       ├── README.md
│   │       └── user_manual.md
│   │ 
│   ├── tests/                               # 
│   │   ├── /                                # 
│   │   │   ├──                              # 
│   │   │   └──                              #
│   │
│   └── README.md                            # Source-level overview
│
├── .coveragerc                              # Exceptions for the PyTest coverage module (print statements and if main statements)
├── .gitignore                               # Files that the VC system should ignore
├── plan.md                                  # Project plan / milestones
├── pytest.ini                               # Sets up the PyTest module, pointing to directories and runtime instructions
├── README.md                                # Project overview
├── requirements.txt                         # Dependencies list (pip)
├── setup_project.sh                         # Helper script; creates virtual environments and runs project tests
└── timelog.md                               # A timelog of project completion                                   

```

This provides absolutely every directory within the project. For more information, most directories contain a further `README.md` to explain in-depth what can be found within. 

---

## Build Instructions



---

### Requirements

This project has been developed using a Linux environment Therefore, the recommendation is to use a Linux operating system or through WSL [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install). 

Additionally, models have been trained using a GPU however, these [notes](https://github.com/BattmannWann/The-VizDoom-Experience/wiki/Saving-Models-in-PyTorch#using-a-gpu) provide a detailed explanation of how to transfer models saved and trained on a GPU to the CPU and vice versa if required. 

Specific requirements for packages and versions used can be found [here](requirements.txt). However, the following provides a quick summary:

- 

Most of the other requirements are installed by these dependencies.

---

### Build Steps

The following assumes a **UNIX SYSTEM** (e.g. a Linux distribution or WSL). It can be done on **other systems** but a warning is issued as this has **NOT** been tested:

```bash

#Create a venv: Recommendation is to create this OUTSIDE of the git repository directory (i.e. one parent directory level up)
$ python3 -m venv [enter_venv_name_here]

#Activate the venv
$ source venv_name_here/bin/activate

#Clone the repository
$ git clone https://github.com/BattmannWann/The-VizDoom-Experience.git

#move into the git directory
$ cd The-VizDoom-Experience/

#Install the dependencies
$ pip install -r requirements.txt

```

And that is all that is needed to build and activate the project dependencies. 

See the following if more information is needed:

- [Installing WSL](https://learn.microsoft.com/en-us/windows/wsl/install#:~:text=Install%20WSL%20command,-You%20can%20now&text=Open%20PowerShell%20in%20administrator%20mode,command%2C%20then%20restart%20your%20machine.&text=This%20command%20will%20enable%20the,the%20Ubuntu%20distribution%20of%20Linux.)



<br>

#### Alternative Method

The `setup_project.sh` script, found in project root, has a function that will handle the creation of a virtual environment for you, installing dependencies and activating it.

---

### Test Steps

Assumptions:

- You are in the project root to start with
- Your virtual environment has been activated and dependencies installed correctly
- You have permission to change file permissions (if not, replace `./` with `source`)


Instructions to run these tests are as follows:

---

```bash

```

---

All of these tests can also be run by the `setup_project.sh` script, on following the instructions presented on screen when running.

---

## Usage

Please consult the [`User Manual`]() for step-by-step instructions on how to use the files provided in this repository.

---

## Support

Please feel free to contact me regarding any queries, issues, or concerns using the [issues page](https://github.com/BattmannWann/The-VizDoom-Experience/issues) or through email: 2682261s@gmail.com | 2682261s@student.gla.ac.uk

I am more than happy to discuss the project at any stage and provide demonstrations.

---

## Project Status

- Initiation stage has begun [week of 21/09/2026]


---

## Hardware Specifications

The following lists the hardware used by my machine but, this does not mean that these are the only hardware that can run the project:

- **CPU**: AMD Ryzen 5 5600X (12 Core)
- **GPU**: NVIDIA GeForce RTX 3060
- **Installed RAM**: 16.0 GB
- **OS**: Windows 11 Home / WSL Ubuntu

---

## Bibliography

[1] Aloimonos, J., Weiss, I. and Bandyopadhyay, A. (1988), ‘Active vision’, International Journal ofComputer Vision 1(4), 333–356.URL: https://doi.org/10.1007/BF00133571

[2] Chuang, I., Lee, A., Gao, D., Naddaf-Sh, M.-M. and Soltani, I. (2025), ‘Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation’. arXiv:2409.17435[cs].URL: http://arxiv.org/abs/2409.17435

[4] Kerr, J., Hari, K., Weber, E., Kim, C. M., Yi, B., Bonnen, T., Goldberg, K. and Kanazawa,A. (2025), ‘Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop’.arXiv:2506.10968 [cs].URL: http://arxiv.org/abs/2506.10968

[5] Li, Z. (2024), ‘Advances in Deep Reinforcement Learning for Computer Vision Applications’,Journal of Industrial Engineering and Applied Science 2(6), 16–26.URL: https://www.suaspress.org/ojs/index.php/JIEAS/article/view/v2n6a03

[6] Rabault, J. & Ren, Feng & Zhang, Wei & Tang, Hui & Xu, Hui. (2020). DEEP REINFORCEMENT LEARNING IN FLUID MECHANICS: A PROMISING METHOD FOR BOTH ACTIVE FLOW CONTROL AND SHAPE OPTIMIZATION. 10.13140/RG.2.2.29692.28800.

[7] Rangel, G., Cuevas-Tello, J. C., Nunez-Varela, J., Puente, C. and Silva-Trujillo, A. G.(2024), ‘A Survey on Convolutional Neural Networks and Their Performance Lim-itations in Image Recognition Tasks’, Journal of Sensors 2024(1), 2797320. _eprint:https://onlinelibrary.wiley.com/doi/pdf/10.1155/2024/2797320.URL: https://onlinelibrary.wiley.com/doi/abs/10.1155/2024/2797320

[8] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. and Dormann, N. (2021), ‘Stable-Baselines3: Reliable Reinforcement Learning Implementations’, 22, 1–8.URL: https://jmlr.org/papers/volume22/20-1364/20-1364.pdf

[9] Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., Cola, G. D., Deleu, T., Goulão, M., Kallinteris,A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Tan, H.and Younis, O. G. (2025), ‘Gymnasium: A Standard Interface for Reinforcement LearningEnvironments’. arXiv:2407.17032 [cs].URL: http://arxiv.org/abs/2407.17032

[10] Trehan, Shubham & Aakur, Sathyanarayanan. (2021). Towards Active Vision for Action Localization with Reactive Control and Predictive Learning. 10.48550/arXiv.2111.05448. 

[11] Wechsler, H. (1990), Active Perception, in ‘Computational Vision’, Elsevier, pp. 248–300.URL: https://linkinghub.elsevier.com/retrieve/pii/B9780127412450500129

[12] Yosha, B.. 5 Top Self-Driving Car Manufacturers [Internet]. Yosha Law Firm - Personal Injury Lawyers. 2023 Feb [cited 2026 Mar 23]. Available from: https://yoshalawfirm.com/blog/5-top-self-driving-car-manufacturers/

[13] Hartwig, B.. 7 key robot applications in automotive manufacturing [Internet]. The Robot Report. 2019 Nov [cited 2026 Mar 23]. Available from: https://www.therobotreport.com/7-key-robot-applications-in-automotive-manufacturing/


