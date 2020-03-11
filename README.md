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

## Overview of Results
Model 8 is the best performer so far. Here are the results.

- Poisson Distribution for predicting review_count
- We normalized coordinates to mean 0 variance 1 before training the model
- Two intermediate layers of 50-50 weights

```
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for stars and ==
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  0.0  |    12.77     |   4.5   |            |  0.0  |     9.25     |   0.9   |
| median |  0.0  |     2.4      |    na   |            |  0.0  |     1.45     |    na   |
|  sum   |  4.42 |    12.76     |   8.18  |            |  2.52 |     9.91     |   2.24  |
| count  |  4.42 |     4.42     |   4.42  |            |  2.52 |     2.52     |   2.52  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for stars and >=
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  0.27 |     4.01     |   0.96  |            |  0.08 |     4.33     |   0.33  |
| median |  0.0  |     0.0      |    na   |            |  0.0  |     3.11     |    na   |
|  sum   |  2.23 |     6.0      |   2.61  |            |  0.77 |     4.52     |   0.86  |
| count  |  1.9  |     1.9      |   1.9   |            |  0.8  |     0.8      |   0.8   |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for stars and <=
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  0.26 |     7.53     |   3.84  |            |  0.14 |     5.93     |   0.36  |
| median |  0.0  |     1.52     |    na   |            |  0.0  |     4.95     |    na   |
|  sum   |  2.76 |     7.5      |   5.96  |            |  1.23 |     5.95     |   0.97  |
| count  |  2.73 |     2.73     |   2.73  |            |  1.03 |     1.03     |   1.03  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for review_count and ==
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  2.36 |     0.0      |   2.06  |            |  2.0  |     0.0      |   0.99  |
| median |  0.0  |     0.0      |    na   |            |  5.56 |     0.0      |    na   |
|  sum   |  8.86 |     8.47     |  12.66  |            |  3.74 |     3.25     |   3.44  |
| count  |  8.47 |     8.47     |   8.47  |            |  3.24 |     3.24     |   3.24  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for review_count and >=
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  0.49 |     0.84     |   0.42  |            |  0.09 |     3.77     |   0.02  |
| median |  0.46 |     0.54     |    na   |            |  0.83 |    10.01     |    na   |
|  sum   |  0.76 |     0.85     |   1.0   |            |  0.43 |     3.57     |   0.37  |
| count  |  0.15 |     0.15     |   0.15  |            |  0.32 |     0.32     |   0.32  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for review_count and <=
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  0.57 |     7.23     |   0.19  |            |  0.19 |     0.32     |   0.15  |
| median |  0.0  |     2.5      |    na   |            |  0.0  |     0.0      |    na   |
|  sum   |  1.89 |     9.78     |   1.12  |            |  1.19 |     0.72     |   1.2   |
| count  |  1.46 |     1.46     |   1.46  |            |  0.94 |     0.94     |   0.94  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for is_open and ==
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  3.94 |    11.32     |   0.0   |            |  0.56 |     2.43     |   0.0   |
| median |  5.56 |    10.56     |    na   |            |  0.0  |     0.0      |    na   |
|  sum   |  2.96 |    14.44     |   0.36  |            |  0.71 |     2.52     |   0.0   |
| count  |  2.27 |     2.27     |   2.27  |            |  0.01 |     0.01     |   0.01  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for is_open and >=
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |   na  |      na      |    na   |            |   na  |      na      |    na   |
| median |   na  |      na      |    na   |            |   na  |      na      |    na   |
|  sum   |   na  |      na      |    na   |            |   na  |      na      |    na   |
| count  |   na  |      na      |    na   |            |   na  |      na      |    na   |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for is_open and <=
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |   na  |      na      |    na   |            |   na  |      na      |    na   |
| median |   na  |      na      |    na   |            |   na  |      na      |    na   |
|  sum   |   na  |      na      |    na   |            |   na  |      na      |    na   |
| count  |   na  |      na      |    na   |            |   na  |      na      |    na   |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
(174566, 5) (10000, 5) (10000, 5)
Evaluating Error for coordinates and range
+--------+-------+--------------+---------+------------+-------+--------------+---------+
| errors | stars | review_count | is_open | next_model | stars | review_count | is_open |
+--------+-------+--------------+---------+------------+-------+--------------+---------+
|  mean  |  6.7  |    11.66     |   4.94  |            |  0.85 |     5.22     |   0.37  |
| median |  6.22 |     9.53     |    na   |            |  0.0  |     4.93     |    na   |
|  sum   | 26.29 |     32.1     |  26.82  |            |  2.08 |     4.81     |   1.62  |
| count  | 26.27 |    26.27     |  26.27  |            |  1.71 |     1.71     |   1.71  |
+--------+-------+--------------+---------+------------+-------+--------------+---------+

```


## TO-DO
1. Implementing R-Tree for faster evaluation of spatial queries. (Done)
2. Try MDN (Done)
3. Try converting to mercator coordinate
3. Complete documentation for 6_2 and 8
4. Implmenting normalizing-flows
5. Trying model out on a bigger dataset
6. Try what Professor Cong proposed

## Links
1. http://bjlkeng.github.io/posts/variational-autoencoders-with-inverse-autoregressive-flows/
2. https://srbittner.github.io/2019/06/26/normalizing_flows/
3. https://github.com/LukasRinder/normalizing-flows/blob/master/example_training.ipynb
4. https://github.com/LukasRinder/normalizing-flows/blob/master/utils/train_utils.py
5. https://github.com/braintimeException/vi-with-normalizing-flows/blob/master/NF/autoencoders/VariationalAutoencoder.py







