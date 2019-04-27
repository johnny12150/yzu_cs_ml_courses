# Hw01 PLA titanic 1041621

### Model 準確度比較

|          | Feature                                                              | Scaler    | Accuracy (NB) | Bias   |
|:--------:|:--------------------------------------------------------------------:|:---------:|:-------------:| ------ |
| baseline | Age, Sex                                                             | x         | 0.736         |        |
|          | Age, Sex                                                             | ✔︎        | 0.45          |        |
|          | Age, Sex, Embarked(ohe)                                              | x         | 0.65          |        |
|          | Age, Sex, Embarked                                                   | x         | 0.765         |        |
|          | Age, Sex, Embarked, TitleGroup(le)                                   | x         | 0.77          | 0.3875 |
| Noise出現  | Age, Sex, Embarked, **Pclass**, TitleGroup(le)                       | x         | 0.734         |        |
|          | Age, Sex, Embarked, **Pclass(ohe)**, TitleGroup(le)                  | x         | 0.7511        |        |
|          | Age, Sex, Embarked, Pclass(ohe), **TitleGroup(ohe)**                 | x         | 0.72          |        |
|          | Age, Sex, Embarked, TitleGroup(le), SibSp                            | x         | 0.6387        |        |
| 要一起      | Age, Sex, Embarked, TitleGroup(le), **SibSp, Parch**                 | x         | 0.763         |        |
|          | Age, Sex, Embarked, **Pclass(ohe)**, TitleGroup(le), SibSp, Parch    | x         | 0.755         |        |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), SibSp, Parch             | x         | **0.7727**    | 0.69   |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), SibSp, Parch             | ✔︎ (Age)  | 0.727         | 0.672  |
|          | Age, Sex, Embarked, TitleGroup(le), **Fare**                         | x         | 0.65          |        |
|          | Age, Sex, Embarked, TitleGroup(le), **Fare**                         | ✔︎        | 0.7368        |        |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), SibSp, Parch, **Fare**   | ✔︎ (Fare) | 0.72          |        |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), FamilySize               | x         | 0.7296        |        |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), FamilySize(ohe)          | x         | 0.717         |        |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), FamilySize, SibSp, Parch | x         | 0.7           |        |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), SibSp, Parch, Fare_5     | x         | 0.727         | 0.741  |
| **Best** | Age, Sex, Embarked, Pclass, TitleGroup(le), SibSp, Parch, Fare_5     | ✔︎ (Age)  | 0.758         | 0.7846 |
|          | Age, Sex, Embarked, Pclass, TitleGroup(le), FamilySize, Fare_5       | x         | 0.677         | 0.38   |

### 觀察與心得

* bias對提升準確度的影響不到 (SGD對pla成效低)

* sclaer無法提升準確度

* 多數feature做one hot的成效不高

* Kaggle跑出的score約是ipynb裡面的少約2%

* 加上bias的效果不是一定都會更好
