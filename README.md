# Owkin-data-challenge

# Description
Deposit for the Owkin data challenge.
It contains the (short) code and the data used (.csv).
### Requirements
```bash
pip install -r requirements.txt
```

# My approach for this problem
## What's in and behind the code
After reading the description of the challenge and understanding it, I decided to use the ```lifeline``` package for the Cox model. 
I quickly tried to obtain a first prediction using all radiomics + all clinical data, which gave me a score just a little below the benchmark. After playing around with the features and the lifeline package which can be very visual, I did a feature selection based on Pearson correlation coefficient. From that I extracted "usefull" features, and did a second prediction, which gave me a score a little above the benchmark (0.7198). I finally used ```lifeline``` to have a quick overview of the model, and realised that a lot of features were "useless" for the regression (0 coefficients in the \beta vector). I removed them and did a last prediction for the test dataset, which gave me a score of 0.728 (please note that the account 

#### My way of coding here

## What's not in the code

## What I wanted to be in this code
