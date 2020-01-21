# Owkin application - Thomas AUZARD

# Description
Deposit for the Owkin data challenge as part of my application for the "Machine Learning Scientist - Medical imaging - Internship" offer.
It contains the (short) code, the data used (.csv) and some explanations.
### Requirements
```bash
pip install -r requirements.txt
```

# My approach for this problem
## What's in and behind the code
After reading the description of the challenge and understanding it, I decided to use the ```lifeline``` package for the Cox model. 
I quickly tried to obtain a first prediction using all radiomics + all clinical data, which gave me a score just a little below the benchmark. After playing around with the features and the lifeline package which can be very visual, I did a feature selection based on Pearson correlation coefficient. From that I extracted "useful" features, and did a second prediction, which gave me a score a little above the benchmark (0.7198). I finally used ```lifeline``` to have a quick overview of the model, and realised that a lot of features were "useless" for the regression (0 coefficients in the \beta vector). I removed them and did a last prediction for the test dataset, which gave me a score of 0.728 (please note that the submission from the account *orsonlelyonnais* and *thomas_auzard* are both mine).

I first coded some functions : it was much easier to play with the data then. I searched a little on the internet about Cox model and survival prediction, and realised that in the amount time I wanted to spend on this project, I would probably not be able to use something different from the "standard" Cox model. 

I found some deposits on GitHub and other packages with "optimized" Cox model (Gradient-boosted, different base-learner), but I decided to stick with standard one and rather play with the parameters.

## What was test and is not in the code
I wanted to give neural net (Keras) a try even though it I thought it would probably be overkill and unefficient.
At first, I used 3d convolutionnal layer with raw scan as inputs, which turned to be way too computationnally expensive for my computer (I wanted to use the scans or the masks, because radiomics were described as biased and suboptimal). It was still too slow with the binary masks, so I decided to try with the radiomics and clinical data as inputs. I tried a "regression" neural net (MSE on the survival time as loss, single output), which gave terrible results. 
I then tried to use a "discrete time" model as classification to use a *standard* neural network. My idea was to set a max survival time, and divide it in a given number of intervals. The goal was to predict in which interval the lifetime would be. Unfortunately, I spent some time on it without having any results so I stopped trying neural nets.

## What could not be tested
- Extracting features directly from the scans instead of using the "standard" radiomics features. I thought that after making a very deep work on the radiomics features (features selection based on different criterion, dimensionality reduction (PCA or ICA)), we could use a neural net using the scan or the binary masks as inputs, and train the network to predict the new features, using image processing methods, features which would then be used in the Cox model, or any other model.

- The differents optimized Cox fitter I read or found about online, and more effective features selections. I used Pearson correlation coefficients but could have probably used other criterion.

- Using other survival time prediction model : from the few papers and articles I read about survival time prediction, Cox model appeared to be the best "estimator" when it comes to censored data, yet I could have tested other model (there actually were some in the ```lifeline``` package).


