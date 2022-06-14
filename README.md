# Report on The Prediction Model

### General  Information about the Dataset

This model contains 301 records concerning heart disease diagnosis. All attributes (8) are numeric-valued. The data was collected from the Cleveland Clinic Foundation. "Num" is the target value of my model. Thus, if num = 1, the patient is more than 50% likely to have any heart disease, but if num = 0, the probability of having the disease is less than 50%. For the model, I used only 8 attributes of the dataset, which originally consisted of 14 attributes. The reason for this was that the number of records in the 6 attributes of the original dataset that I did not take into account for calculation, was very small.

![image](https://user-images.githubusercontent.com/58222828/173665916-6dd17abd-b251-4d58-908f-3e8565dfc7df.png)
                                     (? sign represents the missing value for corresponding attribute)

I deleted only one row in dataset, because that record has a missing value for “thal” attribute:

```python
df_cleveland.drop(df_cleveland[df_cleveland['thal'] == '?'].index, inplace = True)
df_cleveland['thal'] = df_cleveland['thal'].astype(float)
```

70% of the data was used for training and the remaining 30% was used for testing. However, due to the large number of attributes but only a small number of training records in the *LogisticRegression* module, the error is varying noticeably in each iteration of the lbfgs algorithm process, so I gave the number of iterations manually:

```python
from sklearn.linear_model import LogisticRegression
my_model = LogisticRegression(solver='lbfgs', max_iter=100000)
```

As can be seen from the name of the [processed.cleveland.data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) file I selected, the irrelevant attributes in the dataset have been deleted, so there is no need to further look for irrelevant attributes among the 8 attributes to increase accuracy.

### Evaluation of The Model

![image](https://user-images.githubusercontent.com/58222828/173665976-3dd508e1-6828-40b7-905c-86e9f397718a.png)
Based on the content of dataset, we can say that due to the importance of the diagnosis of heart disease, the number of *False Negative* cases in the confusion matrix should be as small as possible. From the results of the *Confusion Matrix* and *Recall* score (~ 80%) we can say that our model shows good results for the above criteria. In general, based on our *AUC* value (~ 90%), the model's ability to distinguish between two classes is acceptable.
