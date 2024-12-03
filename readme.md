# Irrelevant Face Recognition

This is the official pytorch implementation of the paper [Recognizing irrelevant faces in short-form videos based on feature fusion and active learning](https://doi.org/10.1016/j.neucom.2022.06.064)

The dataset used in this project can be accessed [here](https://github.com/JasonZuu/IF-Dataset)

## Getting Started
### A Quick Run
1. Clone this repository
2. Install the dependencies
3. Run the demo of mmfnet
```bash
python demo_mmfnet.py
```

### Train and Test
1. Change the parameters in `configs` to your own settings
2. Run the training script
```bash
python train.py
```
3. Run the testing script
```bash
python test.py
```

### Active Learning
Not yet implemented, will be added soon. 

The current _run_fn.active_fn_ is a old version and has not been verified.

## Project Structure
+ configs: configurations of the training process and the dataset
+ dataset: dataset classes
+ mmfnet: the implementation of the MMFNet
+ run_fn: the training and testing scripts, including the active learning algorithm
+ utils: some utility functions

## Contact
If you have any questions, please contact me at _mingchengzhu250@gmail.com_

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
