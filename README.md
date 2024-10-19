# TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures


![Framework_model.png](docs_images%2FFramework_model.png)

## Link to paper
[Paper](https://arxiv.org/abs/2306.17248)

## Description of files
- train.py contains the training module that is initialized in master before the  training sessions begins
- master.py loads the trainer and data to begin training
- gan.py holds the Discriminator and Generator Architectures for the base models that are trained
- config contains hyper-params, path to training files and other notes/conditions to be imposed by training session

## How to run
Sessions can be run locally, on virtual servers, or on google Colab. Colab notebook is included in the main branch of the repository
- Use requirements.txt to create a conda repository.
- Simply set the config.json files to point to data files and results directory. Set hyperparameters and set up training session.
- Go to master.py and run the file.
- Repository includes pre-trained models that can be used to generate samples. Refer to notebook for steps to do this.
- Repository does not include data because data files are large. Find training data [here](https://data.mendeley.com/datasets/9k892pzkfx/1).
