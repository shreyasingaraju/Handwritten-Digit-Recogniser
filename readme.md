# Handwritten Digit Recogniser

This program allows users to hand draw digits from 0 to 9 on the screen and recognise the digits using AI machine learning in Python. The program uses the MNIST dataset and PyTorch, an open source machine learning library, to train the model and recognise the digits.

## Installation and set-up

First install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) and download Python 3.8 environment. 
Install the following packages in Conda:
- PyQt5
- torch
- torchvision
- matplotlib
- numpy

```bash 
>> conda create –n py38 python=3.8
>> conda activate py38
>> pip install PyQt5 torch torchvision
>> pip install matplotlib
>> pip install numpy
```
Next set up the python environment in [Visual Studio Code](https://code.visualstudio.com/download). 
Open the project folder and select the following interpreter:

```bash
Python 3.8.8 64-bit ('py38':conda)
```

## Running the program

1. Select the main.py file and run the program. A window 'Digit recogniser' will pop up. 
2. Click on 'File' >> 'Train Model'.
3. A second window 'Download and Train' will pop up. Click on 'Download MNIST' 
4. Once the dataset is loaded, click on 'Train'. This will take a few minutes. 
5. Once the progress bar shows 100%, training is complete. You can close the 'Download and Train' window.
6. To view the MNIST dataset click on 'View' >> 'view Testing Images' or 'view Training Images'.
7. Now you can draw any digit from 0 to 9 on the 'Drawing box' or generate a random digit by clicking 'Random'
8. Choose between the 'default' model and 'with_dropout' model by selecting from the dropdown on the right-hand side.
9. Use the trained model to predict the digit by clicking 'Recognise' 
10. A plot showing the probabilities and predicted digit will pop-up on your screen. 
11. The 'Clear' button will clear the main window. 
12. You can repeat steps 7 to 9 and try different digits.
13. To quit the program you can close the window at the top right corner or click on 'View' >> 'Exit' 

## Contributors 

Henry Mitchell-Hibbert & Shreya Singaraju
