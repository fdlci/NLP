# NLP Exercise 2: Aspect-Based Sentiment Analysis

## TEAM NLP:
- QABEL Mohamed Aymen
- CHENENE Mohamed
- FLOREZ DE LA COLINA Inès
- AOUAD Elias


## Project structure:
The main files and folders are briefly described below. 

```
.
├── data                         - folder containing the csv data files (details below)
├── src                          - python package
│   ├─Data_prepa                 - Preparation of the dataset loading 
│   ├─sentbert                   - The model used in our project
│   ├─classifier                 - The script used to train our model
│   └── tester                   - script used to test the models
└──
```


In the script Data_prepa, we preprocess our data, by encoding the categories and the opinions into integer labels and
we tokenize our sentences so that they can be treated by BERT Model. Also, we extract the position of the target words
in the tokenized sentences.

In the sentbert script, we present our model. The idea is instead of making our prediction from the embedding of CLS after the 
BERT Model, we concatenate this CLS embedding with the embedding of the target words. And then we make our prediction on the 
categories space using  a Linear layer then choosing the values corresponding to the category to which the sentence belongs.

Finally, we train our model in the classifier script using as a validation set the devset file provided. 

### The final accuracy on the validation set was around 88%.


