# TD-C Learning: Time Dependent Contrastive Learning
###  [Paper](https://www.nature.com/articles/s41928-022-00888-7) | [Video(keyboard)](https://static-content.springer.com/esm/art%3A10.1038%2Fs41928-022-00888-7/MediaObjects/41928_2022_888_MOESM6_ESM.mp4) | [Video(Object Recognition)](https://static-content.springer.com/esm/art%3A10.1038%2Fs41928-022-00888-7/MediaObjects/41928_2022_888_MOESM7_ESM.mp4)

## [A substrate-less nanomesh receptor with meta-learning for rapid hand task recognition](https://www.nature.com/articles/s41928-022-00888-7)  
 [Kyun Kyu Kim](http://kyunkyukim.com)\#<sup>1</sup>,
 [Min Kim](https://minkim.io/))\#<sup>2</sup>,
 [Sungho Jo](http://nmail.kaist.ac.kr/wordpress/index.php/professor-jo-sungho/)\*<sup>2</sup>,
 [Seung Hwan Ko](link)\*<sup>3</sup>,
 [Zhenan Bao](http://baogroup.stanford.edu)\*<sup>1</sup> <br>
 <sup>1</sup>Stanford, CA, USA, <sup>2</sup>Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Korea, <sup>3</sup>Seoul National University, Seoul, Korea 
 \#denotes equal contribution   
in Nature Electronics

# Dependecy

This repo is written in Python 3.9. Any Python version > 3.7 will be compatible with our code. 

This repo is tested on Windows OS with CUDA 11. For the same environment, you can install Pytorch with the below command line, otherwise, please install Pytorch by following the instructions on the official Pytorch website: https://pytorch.org/get-started/locally/


```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# quick setup

Python 3 dependencies:
* Pytorch 1.12
* attrs
* numpy
* PyQt5
* scikit-learn

We provide a conda environment setup file having all the dependencies required for running our code. You can create a conda environment tdc by running below command line:


```bash
conda env create -f environment.yml
```

# Runnig code
Our training steps are divided into two seperate parts: 1. TD-C Learning, 2. Rapid Adaptation
We provide codes and experiment environments for adopting our learning method, including data parsing, training code, and basic UI for collecting few-shot demonstration and making real-time inference. 

## TD-C Learning
TD-C learning is an unsupervised learning method that utilizes jittering signal augmentation and time-dependent contrastive learning to learn sensor representations with unlabeled random motion data. Here we show data format used to run our code and how to run our code with sample unlabeled data. 

### 1. Prepare unlabeled dataset
To run the code, first prepare byte-encoded pickle files containing sensor signals in a dictionary data structure with key 'sensor' and value sequential sensor signals: {'sensor': array(s1, s2, ....)}
Our code will read and parse all pickle files in ./data/train_data with above dictionary format. 

### 2. Change hyperparameters
We found out that the best-performing window size and data embedding size are dependent on the total amount of data, data collection frequency, and so on. You can change different hyperparameter settings by simply modifying values in params.py file. 

### 3. Running tdc train
Run
```
python tdc_train.py 
```

## Rapid few-shot adaptation and real-time inference
To allow pretrained model to be adapted to perform different tasks, we applied a few-shot transfer learning and metric-based inference mechanism for real-time inference. Here we provide basic UI system implemented with PyQT5 which allows users to collect few-shot demo and make real-time inference. 

### 1. Basic UI
We provide basic UI code in ui directory

The UI contains two buttons: 1. Collect Start, 2. Start Prediction and two Widgets: 1. status widget showing current prediction, 2. sensor widget showing current sensor values. 

The system starts to record few-shot labeled data from demonstration when user press "Collect Start" button. After providing all required demonstration, make sure to press "Start Prediction" button, so that the system starts to transfer learn the model. 

### 2. Few-shot rapid adaptation, Data embedding and metric-based inference system
In transfer_learning_base.py file, we provide transfer learning, data embedding, and metric-based inference functions

In our system, the system does transfer learning with provided few-shot demonstrations. The number of transfer epochs can be modified by changing transfer_epoch variable in params.py.

After running a few transfer epoch, the system encode few-shot label data with transferred model to generate demo_embeddings. These embeddings are then used as references for Maximum Inner Product Search(MIPS). Given a window of sensor values, the model generates its embedding and phase variable. If the phase variable exceeds predefined threshold, the system perform MIPS and corresponding prediction is appeared on the status widget. Otherwise, the system regards the current state as a resting state. 


