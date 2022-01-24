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


@component(packages_to_install=['google-cloud-storage'])
def download_gcs_file(project: str, bucket: str, filepath: str, output_file: Output[Artifact]) -> None:
    '''Downloads a file from GCS to a KFP Artifact component'''
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket)
    blob = bucket.blob(filepath)

    blob.download_to_filename(output_file.path)


@component(packages_to_install=['pandas', 'sklearn'])
def train_model(df_artifact: Input[Artifact], target_string: str, model_artifact: Output[Artifact])->None:
    import pandas as pd 
    import pickle
    import json 
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    df = pd.read_csv(df_artifact.path)
    target = json.load(target_string)

    y = df[target]
    x = df.drop(target, axis=1)
    
    model.fit(x,y)
    pickle.dump(model, open(model_artifact.path, 'wb'))


@component(packages_to_install=['pandas', 'sklearn'])
def predict(model_artifact: Input[Artifact], df_artifact: Input[Artifact], target_string: str, y_pred_artifact: Output[Artifact])->None:
    import pandas as pd 
    import pickle 
    import json

    df = pd.read_csv(df_artifact.path)
    target = json.load(target_string)

    model = pickle.load(open(model_artifact.path, 'rb'))

    x = df.drop(target, axis=1)
    y_pred = model.predict(x)
    y_pred.to_csv(y_pred_artifact.path)


@component(packages_to_install=['pandas', 'sklearn'])
def evaluate(df_artifact: Input[Artifact], target_string: str, y_pred_artifact: Input[Artifact], accuracy_artifact: Output[Artifact])->None:
    import pandas as pd
    import json
    import sklearn 
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(df_artifact.path)
    target = json.load(target_string)

    y = df[target]
    y_pred = pd.read_csv(y_pred_artifact.path)

    accuracy = accuracy_score(y, y_pred)
    df_accuracy = pd.DataFrame([accuracy], columns=['Accuracy']) 
    df_accuracy.to_csv(accuracy_artifact.path)


@pipeline(name='titanic_pipleine', description='dummy pipeline on titanic dataset')
def build_pipeline(
    data_project_string: str, 
    data_bucket_string: str, 
    data_filepath_string: str,
    cols_to_drop_string: str, 
    sex_string: str, 
    age_string: str,
    age_group_by_cols_string: str,
    fare_string: str,
    fare_group_by_cols_string: str,
    embarked_string: str,
    sibsp_string: str,
    parch_string: str,
    model_project_string: str, 
    model_bucket_string: str,
    model_filepath_str: str,
    target_string: str
    ): 


    download_gcs_file_data_op = download_gcs_file(data_project_string, data_bucket_string, data_filepath_string)

    column_cleaning_op = column_cleaning(download_gcs_file_data_op.output, cols_to_drop_string)

    onehot_encoding_sex_op = onehot_encoding(column_cleaning_op.output, sex_string)
    
    numerical_imputation_by_group_age_op = numerical_imputation_by_group(onehot_encoding_sex_op.output, age_string, age_group_by_cols_string)
    numerical_imputation_by_group_fare_op = numerical_imputation_by_group(numerical_imputation_by_group_age_op.output, fare_string, fare_group_by_cols_string)

    onehot_encoding_embarked_op = onehot_encoding(numerical_imputation_by_group_fare_op.output, embarked_string)

    robust_scaling_age_op = robust_scaling(onehot_encoding_embarked_op.output, age_string)
    robust_scaling_fare_op = robust_scaling(robust_scaling_age_op.output, fare_string)

    binary_binning_sibsp_op = binary_binning(robust_scaling_fare_op, sibsp_string)
    binary_binning_parch_op = binary_binning(binary_binning_sibsp_op, parch_string)

    download_gcs_file_model_op = download_gcs_file(model_project_string, model_bucket_string, model_filepath_str)
    predict_op = predict(download_gcs_file_model_op.output, binary_binning_parch_op.output, target_string)

    evaluate_op = evaluate(binary_binning_parch_op.output, target_string, predict_op.output) 