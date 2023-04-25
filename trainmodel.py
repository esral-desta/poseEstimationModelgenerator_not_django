# reading file
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('newmodel.csv')
# df.head()
# df.tail()
# df[df["class"]=='up']
x= df.drop('class',axis=1)
y=df["class"]

X_train,X_test,Y_train,Y_test  = train_test_split(x,y,test_size=0.3,random_state=1234)
# Y_test


# generating model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

pipelines = {
    "lr":make_pipeline(StandardScaler(),LogisticRegression()),
    "rc":make_pipeline(StandardScaler(),RidgeClassifier()),
    "rf":make_pipeline(StandardScaler(),RandomForestClassifier()),
    "gb":make_pipeline(StandardScaler(),GradientBoostingClassifier()),
}

fit_models = {}

for algo,pipeline in pipelines.items():
    model = pipeline.fit(X_train,Y_train)
    fit_models[algo]=model


## fitmodels
## fitmodels['rc'].predict(X_test)

# evaluating asn serializing model

from sklearn.metrics import accuracy_score,precision_score,recall_score
import pickle

for algo,model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo,accuracy_score(Y_test.values,yhat),

        precision_score(Y_test.values,yhat,average="binary",pos_label="up"),
        recall_score(Y_test.values,yhat,average="binary",pos_label="up")

    )


with open("mymodel.pkl","wb") as f:
    pickle.dump(fit_models["rf"],f)



