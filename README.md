# Model Compression Analysis

The goal of the project is to analyse the effects of compression techniques - namely pruning and quantization - on neural networks. Metrics like size, accuracy (top1 and top5), and inference speed of the compressed and original models have been captured and analysed. 
Additionally, applying these techniques to neural networks of different sizes also helped highlight their efficiency and trade-offs when scaling up.
Finally, the compressed models were also deployed to an edge-device, a mobile phone in this case, and tested via an android application.

The models used for this analysis are:
- **LeNet5**:
  
  Trained on the MNIST dataset. Chosen as the smaller convolution neural network.
- **ResNet50**:
  
  Trained on ImageNet200, which is a subset of the ImageNet1000 dataset. Chosen as the larger convolution neural network.

### Compression Techniques
- **Pruning**: Structured, Unstructured, and Structured + Unstructured
-  **Quantization**: Dynamic and Static (both per-tensor symmetric style)

A more detailed explanation of the project is available in the **project_report.pdf** file present in the root directory of the project.

### Results
(Explanation of all the plots is available in the project report)
- **ResNet50**
  
  ![image](https://github.com/user-attachments/assets/5b57c404-1d5d-4ac4-b46f-624d34886e5f)
  

- **LeNet5**
  
  <img width="868" alt="image" src="https://github.com/user-attachments/assets/4fe3aacf-4164-4f09-af30-b2e8c40cc0f1" />

  <img width="868" alt="image" src="https://github.com/user-attachments/assets/892bc8e8-cbfd-4f26-be19-e2088fbf86ca" />

  <img width="868" alt="image" src="https://github.com/user-attachments/assets/f60ebe44-0f9b-4993-9b8e-1a50170af287" />


### Setting up the project

### Prerequisites

- **Python 3.x**: Install from [python.org](https://www.python.org/downloads/).
- **Git**: Install from [git-scm.com](https://git-scm.com/).

### Step 1: Clone the repository

Clone the repo to your local machine:

```bash
git clone https://github.com/lakshyasuri/ModelCompressionAnalysis.git
cd ModelCompressionAnalysis
```

### Step 2: Set up a virtual environment, Jupyter notebook kernel, and install dependencies

```bash
source setup.sh
source .venv/bin/activate
```




