# Owkin-data-challenge

# Description
Deposit for the Owkin data challenge.
It contains the (short) code and the data used (.csv).
### Requirements
```bash
pip install -r requirements.txt
```

# My approach for this problem
### What's in and behind the code
After reading the description of the challenge and understanding it, I decided to use the lifeline package for the Cox model. 
I quickly tried to obtain a first prediction using all radiomics + all clinical data, which gave me a score just a little below the benchmark. After playing around with the features and the lifeline package which can be very visual, I did a feature selection based on Pearson correlation coefficient. From that I extracted "usefull" features, and did a second prediction, which gave me a score a little above the benchmark (0.7198). I finally used the ```lifeline```
