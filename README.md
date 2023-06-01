# MNIST Dataset
A simple model using pytorch I made for learning purposes that runs and trains on the **MNIST** Dataset. 

## *Decription*

 Layer (type) | Output Shape | Param 
--- | --- | ---
Conv2d-1 | [-1, 32, 26, 26] | 320
Conv2d-2 | [-1, 64, 24, 24] | 18,496
Conv2d-3 | [-1, 128, 10, 10] | 73,856
Conv2d-4 | [-1, 256, 8, 8] | 295,168
Linear-5 | [-1, 50] | 204,850
Linear-6 | [-1, 10] | 510

Total params: 593,200  
Trainable params: 593,200  
Non-trainable params: 0

Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94



## How to Install and Run the Project

To install the project simply download the code as a zip. To run the project just run the ```S5.ipynb``` file. The dependencies required are pytorch, tqdm, matplotlib, torchsummary etc. If not able to run just use an python environment file to run it.

