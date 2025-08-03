# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model uses a random forest classifier to classify whether a person makes over 50K a year. This model is trained on the UCI Census Income Dataset.
## Intended Use
This is not a production model, and this dataset has traditionally only beed used for research purposes.

## Training Data
The training data accounts for 80% of the whole dataset. The categorical features are workclass, education, marital-status, occupation, relationship, race, sex, native-country. The continous features are age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week. The label is salary.

## Evaluation Data
Evaluation data accounts for 20% of the whole dataset. The categorical features used one hot encoder.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The metrics used are precision, recall, and F-beta score. The model has precision 0.81, recall: 0.57, and F-beta score: 0.67

## Ethical Considerations
Risk: We risk expressing the viewpoint that the attributes in this dataset are the only ones that are predictive of someone's income, even though we this is not the case.
Mitagation strategy: As mentioned, some interventions may need to be performed to address the class imbalances in the dataset.

## Caveats and Recommendations
This dataset is class-imblanced across a variety of sensitive classes. For example the ratio of male-female is about 2:1 and in race there are way more cases of White then other race. To avoid this, we can try various remediation strategies in future iterations, but we may not be able to fix all of the fairness issues. 