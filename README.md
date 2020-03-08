# VAE Spatial Sample Generator
This repository explores the possibility of using Variational Auto-Encoders (VAE) to learn the statistical characteristics of a certain specified dataset and generate samples having similar characteristics. This work presents an interesting direction of research for applications involving Approximate Query Processing (AQP) .

## How to use this repository

### Installing the Dependecies
This project runs on **Python 3.6**. The models are built using **Keras** and all the dependencies are written inside the `requirement.txt` file. A virtual environment is recommended. You can install one easily (Just Google).

Follow instructions below to set up environment. Open up your terminal and type the following instructions.

```python
python3 -m venv vae_env # to create a virtual environment

source vae_env/bin/activate # this activates the virtual environment

pip3 install -r requirements.txt # installing dependencies stated in the requirements file. 
```
**You are good to try out the models once the above is done. Just go into each of the models, open up the notebooks and run anything.**

### File Structure

This repository contains notebooks for training the proposed model, generating samples and performing testing. It is not yet intended for practical use. 

The different models developed are stored inside `models` directory. Each model contains 3 kinds of scripts. 

1. **Model Scripts (Model Tranining)** - Contains Model Architecture and facilitates model training.
2. **Sample Generating Scripts (Sample Generation)** - Generate samples using the model trained from **Model Scripts**
3. **Experiment Scripts (Experiments)** - Scripts for conducting experiments to evaluate model performance. Uses samples generated from **Sample Generating Scripts**.




