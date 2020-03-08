# VAE Spatial Sample Generator
This repository explores the possibility of using Variational Auto-Encoders (VAE) to learn the statistical characteristics of a certain specified dataset and generate samples having similar characteristics. This work presents an interesting direction of research for applications involving Approximate Query Processing (AQP) . 

This is a project is done under the guidance and supervision of Professor Cong Gao and his student Li Xiucheng. Feel free to Email me at `liangweitan300895@gmail.com` for any questions. 

## How to use this repository

### Installing the Dependecies
This project runs on **Python 3.6**. The models are built using **Keras** and all the dependencies are written inside the `requirement.txt` file. Also, the experiemnts are conducted inside **Jupyter Notebook**. A virtual environment is recommended. 


Follow instructions below to set up environment. Open up your terminal and type the following instructions.

```python
python3 -m venv vae_env # to create a virtual environment

source vae_env/bin/activate # this activates the virtual environment

pip3 install -r requirements.txt # installing dependencies stated in the requirements file. 
```

For **Jupyter Notebook**:

[https://jupyter.org/install](https://jupyter.org/install)

```
pip install notebook # to install

jupyter notebook # to run it
```


**You are good to try out the models once the above is done. Just go into each of the models, open up the notebooks and run anything.**

### File Structure

This repository contains notebooks for training the proposed model, generating samples and performing testing. It is not yet intended for practical use. 

The different models developed are stored inside `models` directory. Each model contains 3 kinds of scripts. 

1. **Model Scripts (Model Tranining)** - Contains Model Architecture and facilitates model training.
2. **Sample Generating Scripts (Sample Generation)** - Generate samples using the model trained from **Model Scripts**
3. **Experiment Scripts (Experiments)** - Scripts for conducting experiments to evaluate model performance. Uses samples generated from **Sample Generating Scripts**.

**You can start ignore models 1 and 2 and start looking from the third model. It has the best performance thus far.**




