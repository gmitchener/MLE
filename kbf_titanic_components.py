from ast import In
from re import A
import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (component,Input,Output,Dataset,Metrics,pipeline,Artifact)


@component(packages_to_install=['pandas'])
def column_cleaning(df_artifact: Input[Artifact], cols_to_drop_string: str, df_cleaned_artifact: Output[Artifact])->None:
    import pandas as pd 
    import json 

    df = pd.read_csv(df_artifact.path)
    cols_to_drop = json.loads(cols_to_drop_string)

    df = df.drop(cols_to_drop, axis=1)

    df.to_csv(df_cleaned_artifact.path)


@component(packages_to_install=['pandas', 'sklearn'])
def onehot_encoding(df_artifact: Input[Artifact], col_to_encode_string: str, df_encoded_artifact: Output[Artifact])->None:
    import pandas as pd 
    import json
    import sklearn
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()

    df = pd.read_csv(df_artifact.path)
    col_to_encode = json.loads(col_to_encode_string)

    results = ohe.fit_transform(df[col_to_encode].values.reshape(-1,1)).toarray()

    feature_names = [string.split('_')[1] for string in ohe.get_feature_names()]
    df[feature_names] = pd.DataFrame(result, index=df.index)

    least_common = df[col_to_encode].value_counts().index[-1]
    df.drop([col_to_encode, least_common], axis=1)

    df.to_csv(df_encoded_artifact.path)


@component(packages_to_install=['pandas'])
def numerical_imputation_by_group(df_artifact: Input[Artifact], col_to_imput_string: str, cols_to_group_by_string: str, df_imputed_artifact: Output[Artifact])->None:
    import pandas as pd
    import json 

    df = pd.read_csv(df_artifact.path)
    col_to_imput = json.loads(col_to_imput_string)
    cols_to_group_by = json.loads(cols_to_group_by_string)

    df[col_to_imput] = df.groupby(cols_to_group_by)[col_to_imput].apply(lambda x: fillna(x.meadian()))

    df.to_csv(df_imputed_artifact.path)


@component(packages_to_install=['pandas', 'sklearn'])
def categorical_imputation_most_common(df_artifact: Input[Artifact], col_to_imput_string: str, df_imputed_artifact: Output[Artifact])->None:
    import pandas as pd 
    import json 

    df = pd.read_csv(df_artifact.path)
    col_to_imput = json.loads(col_to_imput_string)

    most_common = df[col_to_imput].mode()[0]
    df[col_to_imput].fillna(most_common)

    df.to_csv(df_imputed_artifact.path)


@component(packages_to_install=['pandas', 'sklearn'])
def robust_scaling(df_artifact: Input[Artifact], col_to_scale_string: str, df_scaled_artifact: Output[Artifact])->None:
    import pandas as pd 
    import json 
    import sklearn
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    df = pd.read_csv(df_artifact.path)
    col_to_scale = json.loads(col_to_scale_string)

    df[col_to_scale] = scaler.fit_transform(train_df[col_to_scale].values.reshape(-1,1))

    df.to_csv(df_scaled_artifact.path)


@component(packages_to_install=['pandas'])
def binary_binning(df_artifact: Input[Artifact], col_to_bin_string: str, df_binned_artifact: Output[Artifact])->None:
    import pandas as pd 
    import json

    df = pd.read_csv(df_artifact.path)
    col_to_bin = json.loads(col_to_bin_string)

    correction = lambda x: 1 if x != 0 else 0
    df[col_to_bin] = df[col_to_bin].apply(correction)

    df.to_csv(df_binned_artifact.path)


@component(packages_to_install=['pandas', 'sklearn'])
def train_model(df_processed_artifact: Input[Artifact], model_artifact: Output[Artifact])->None:
    import pandas as pd 
    import pickle
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()

    df = pd.read_csv(df_processed_artifact.path)
    y = df.Survived
    x = df.drop('Survived', axis=1)
    

    logreg.fit(x,y)

    pickle.dump(logreg, open(model_artifact.path, 'wb'))


@component(packages_to_install=['pandas', 'sklearn'])
def predict(model_artifact: Input[Artifact], x_test_artifact: Input[Artifact], y_pred_artifact: Output[Artifact])->None:
    import pandas as pd 
    import pickle 

    model = pickle.load(open(model_artifact.path, 'rb'))

    x_test = pd.read_csv(x_test_artifact.path)
    y_pred = model.predict(x_test)
    y_pred.to_csv(y_pred_artifact.path)


@component(packages_to_install=['pandas', 'sklearn'])
def evaluate(y_test_artifact: Input[Artifact], y_pred_artifact: Input[Artifact])->float:
    import pandas as pd
    import sklearn 
    from sklearn.metrics import accuracy_score

    y_test = pd.read_csv(y_test_artifact.path)
    y_pred = pd.read_csv(y_pred_artifact.path)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

@pipeline(name='titanic_pipleine', description='dummy pipeline on titanic dataset')
def build_pipeline():
    preprocessing_op = preprocessing(df)
    train_model_op = train_model(preprocessing_op.output)
    predict_op = predict(train_model_op.output, x_test)
    evaluate_op = evaluate(y_test, predict_op.output)