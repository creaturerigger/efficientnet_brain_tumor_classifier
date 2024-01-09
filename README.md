# Reproducing AlexNet Paper

This repository is the official implementation of Brain Tumor MRI Images Classification with EfficientNet - B7 as the final project of CMP5550 Computer Vision course. In this study the [Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) dataset has been used.

## Environment

In this project Python virtual environment has been utilized. To create a virtual environment for the project run the following code:

```virtualenv
python -m venv <path-to-virtualenv>
```

After virtual environment is created to activate the virtual environment run the following code:

```windows
call <path-to-virtualenv>/Scripts/activate.bat
```

```linux&macos
source <path-to-virtualenv>/bin/activate
```

After activating the virtual environment your terminal should look like below:

```terminal-windows
(venv) C:\<path-to-project>
```

```terminal-macos
(venv) machine-name:path-to-project username$
```

```terminal-linux
(venv) username@machine-name:path-to-project$
```

## Requirements

After activating the virtual environment to install the requirements run the following code:

```setup
pip install -r requirements.txt
```

## Training

To train the model described in the paper, run the following command:

```train
python train.py -d <device-type> -mv <model-version> -e <number-of-epochs> -lr <learning-rate> -op <optimizer> -tb <train-batch-size> -vb <validation-batch-size> -dp <dataset-path> -ci <checkpoint-save-interval> -r <resolution> -nc <number-of-out-classes>
```

To get more help and see the options run the following command:

```train-help
python train.py --help
```

Train creates a directory named as "artifacts" and saves the following artifacts in this directory:

* Train history plot
* Model checkpoints
* Optimizer checkpoints
* Learning rate scheduler checkpoints
* Scaler checkpoints

## Evaluating

To evaluate the model run the following command:

```eval
python eval.py -dp <dataset-path> -tbs <test-batch-size> -mp <model-checkpoint-path> -mv <model-version> -nc <number-of-classes>
```

To get more help and see the options run the following command:

```eval-help
python eval.py --help
```

## Results

Our model achieves the following performance:

| AlexNet            | Top-1 Accuracy  | Top-5 Accuracy |
| ------------------ |---------------- | -------------- |
| AlexNet            |     3.36%       |      14.34%    |


## Contributing Guidelines

We welcome contributions to this project! To ensure a smooth contribution process, please follow these guidelines:

**Areas of Contribution:**

* Bug fixes and improvements to the codebase
* New features related to the paper or model
* Documentation improvements and tutorials
* Unit tests and code coverage enhancements
* Sharing experimental results and analyses

**Contribution Workflow:**

1. Fork this repository and create a new branch for your changes.
2. Implement your changes and update the relevant documentation.
3. Run unit tests and ensure your code adheres to the project's style guide.
4. Create a pull request and clearly describe your changes.
5. Be prepared to address any feedback or suggestions from the maintainers.

**Additional Notes:**

* Please adhere to the PEP 8 coding style guide.
* Include unit tests for any new code you add.
* Use descriptive commit messages and pull request titles.
* We appreciate contributions in any form, even if they are small bug fixes or suggestions.

Thank you for your interest in contributing!

For citation you can use the following BibTeX

```BibTeX
@misc{alexnet_reproduction,
  author = "Anwar Abuelrub, and Volkan Bakir",
  title = "Reproducing AlexNet Paper: Final Project for ARI5004 Deep Learning Course",
  year = "2024",
  howpublished = "\url{https://github.com/creaturerigger/reproducing_alexnet_paper}",
  note = "Implements the AlexNet model from the {ImageNet Classification with Deep Convolutional Neural Networks} paper on the ImageNet100 dataset. Includes training, evaluation, and performance analysis scripts."
}
```