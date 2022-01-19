import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (component,Input,Output,Dataset,Metrics,pipeline,Artifact)

@component(packages_to_install=['pandas', 'sklearn'])
def preprocessing(df_artifact: Input[Artifact], df_processed_artifact: Output[Artifact])->None:
    import pandas as pd 
    import sklearn
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
    le = LabelEncoder()
    ohe = OneHotEncoder()
    scaler = RobustScaler()

    df = pd.read_csv(df_artifact.path)

    def drop_cols(df, cols_to_drop):
        df = df.drop(cols_to_drop, axis=1)
        return df

    def encode_sex(df):
        df['Male'] = le.fit_transform(df.Sex)
        df = df.drop(['Sex'], axis=1)
        return df

    def imputation_by_group(df, column_to_impute, columns_to_group_by):
        """Impute the column_to_impute by taking the median group on columns_to_group_by"""
        df[column_to_impute] = df.groupby(columns_to_group_by)[column_to_impute].apply(lambda x: x.fillna(x.median()))
        return df 

    def fill_categorical_most_common(df, column_to_fill):
        """Fill nulls in categorical column_to_fill with the most common variable"""
        most_common = df[column_to_fill].mode()[0]
        df[column_to_fill].fillna(most_common)
        return df

    def one_hot_encoding(df, col_to_encode):
        """One hot encodes categorical variable col_to_encode and drops the least common column"""
        result = ohe.fit_transform(df[col_to_encode].values.reshape(-1,1)).toarray()

        feature_names = ohe.get_feature_names().tolist()
        df[feature_names] = pd.DataFrame(result, index=df.index)

        least_common = df[col_to_encode].value_counts().index[-1]
        col_to_drop = 'x0_'+least_common
        df = df.drop([col_to_encode, col_to_drop], axis=1)
        return df 

    def robust_scaling(df, col_to_scale):
        df[col_to_scale] = scaler.fit_transform(train_df[col_to_scale].values.reshape(-1,1))
        return df

    def binary_binning(df, col_to_bin):
        """Takes col_to_bin and creates binary count for 0 or 1+"""
        correction = lambda x: 1 if x != 0 else 0
        df[col_to_bin] = df[col_to_bin].apply(correction)
        return df

    non_informative_cols = ['PassengerId', 'Name', 'Ticket']
    df = drop_cols(df, non_informative_cols)

    mostly_null_cols = ['Cabin']
    df = drop_cols(df, mostly_null_cols)

    df = encode_sex(df)

    df = imputation_by_group(df, 'Age', ['Pclass', 'Male'])
    df = imputation_by_group(df, 'Fare', 'Pclass')

    df = fill_categorical_most_common(df, 'Embarked')

    df = one_hot_encoding(df, 'Embarked')

    df = robust_scaling(df, 'Age')
    df = robust_scaling(df, 'Fare')

    df = binary_binning(df, 'SibSp')
    df = binary_binning(df, 'Parch')

    df.to_csv(df_processed_artifact.path) 


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