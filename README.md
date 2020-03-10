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

## Model Descriptions

### Model 1
A vanilla model. 50 and 50 for both intermediate layers.

- stars - Multinomial Distribution - softmax activation

- review_count - Gaussian Distribution - No activation

- latitude and longitude - Gaussian Distribution - No activation

- is_open - Bernoulli Distribution - Sigmoid Activation

*Very Poor Results*

### Model 2

- To keep longitude and latitude within [-180, 180], by projecting the entire range within 0 and 1. Basically, (coordinate+180)/360.

- To ensure that review_count stays positive, we used Relu as the activation function for the mean. 

*Coordinates making more sense now but still doing poorly for review_count*

### Model 2_1

- Same as above but adjusted model to output log mu for review_count. Adjusted lost function and generation accordingly

*Actually Results are much better. Achieved less than 10% error for review_count. Comparable to using poisson distribution.*

### Model 3
- Same as above but with 12 and 8 for intermediate layers. 

*Worse as compared to Model 2*

### Model 4 (Key Changes)
- Used the Poisson Distribution for review_count
-  12 - 8 for intermediate layers

*Much better results for review_count aggregates*

### Model 5
- Same as above but with 50-50 intermediate layer weights. 

*Better results as compared to model 4. Best Results thus far.*



## TO-DO
1. Implementing R-Tree for faster evaluation of spatial queries. (Done)
2. Try MDN (Done)
3. Try converting to mercator coordinate
3. Complete documentation for 6_2 and 8
4. Implmenting normalizing-flows
5. Trying model out on a bigger dataset
6. Try what Professor Cong proposed







