"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    def normalise(df_clean,column_name):
    # your code here
        max_threshold = df_clean[column_name].quantile(0.95)
        min_threshold = df_clean[column_name].quantile(0.05)
    
        df_clean[column_name] = [value if value < max_threshold  else df_clean[column_name].mean() for value in df_clean[column_name]  ]
        df_clean[column_name] = [value if value > min_threshold  else df_clean[column_name].mean() for value in df_clean[column_name]  ]
    
        return df_clean

    df_train_clean = df_train_clean.drop(['Valencia_wind_deg', 'Seville_pressure'], axis=1)

    df_test_clean = df_test_clean.drop(['Valencia_wind_deg', 'Seville_pressure'], axis=1)

    def avg_wind_speed_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df["spain_wind_speed"] =  df.loc[:, similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_wind_speed_features(df_train_clean, keyword= 'wind_speed')
    df_test_clean = avg_wind_speed_features(df_test_clean, keyword= 'wind_speed')

    def avg_rain_1h_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_rain_1h"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_rain_1h_features(df_train_clean, keyword= 'rain_1h')
    df_test_clean = avg_rain_1h_features(df_test_clean, keyword= 'rain_1h')


    def avg_humidity_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_humidity"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_humidity_features(df_train_clean, keyword= 'humidity')
    df_test_clean = avg_humidity_features(df_test_clean, keyword= 'humidity')


    def avg_clouds_all_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_clouds_all"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_clouds_all_features(df_train_clean, keyword= 'clouds_all')
    df_test_clean = avg_clouds_all_features(df_test_clean, keyword= 'clouds_all')

    def avg_wind_deg_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_wind_deg"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_wind_deg_features(df_train_clean, keyword= 'wind_deg')
    df_test_clean = avg_wind_deg_features(df_test_clean, keyword= 'wind_deg')

    def avg_pressure_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_pressure"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_pressure_features(df_train_clean, keyword= 'pressure')
    df_test_clean = avg_pressure_features(df_test_clean, keyword= 'pressure')

    def avg_snow_3h_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_snow_3h"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_snow_3h_features(df_train_clean, keyword= 'snow_3h')
    df_test_clean = avg_snow_3h_features(df_test_clean, keyword= 'snow_3h')

    def avg_weather_id_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_weather_id"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_weather_id_features(df_train_clean, keyword= 'weather_id')
    df_test_clean = avg_weather_id_features(df_test_clean, keyword= 'weather_id')

    def avg_temp_max_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_temp_max"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_temp_max_features(df_train_clean, keyword= 'temp_max')
    df_test_clean = avg_temp_max_features(df_test_clean, keyword= 'temp_max')

    def avg_temp_min_features(df, keyword):
        similar_columns = [col for col in df_train_clean.columns if keyword in col]
        df.loc[:,"spain_temp_min"] =  df_train_clean.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_temp_min_features(df_train_clean, keyword= 'temp_min')
    df_test_clean = avg_temp_min_features(df_test_clean, keyword= 'temp_min')

    def avg_temp_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_temp"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_temp_features(df_train_clean, keyword= 'temp')
    df_test_clean = avg_temp_features(df_test_clean, keyword= 'temp')

    def avg_temp_min_features(df, keyword):
        similar_columns = [col for col in df.columns if keyword in col]
        df.loc[:,"spain_rain_3h"] =  df.loc[:,similar_columns].mean(axis=1)
        
        return df
    df_train_clean = avg_temp_min_features(df_train_clean, keyword= 'rain_3h')
    df_test_clean = avg_temp_min_features(df_test_clean, keyword= 'rain_3h')

    from datetime import datetime
    if 'Year' not in df_train_clean:
        df_train_clean.insert(loc=1, column='Year', value= pd.to_datetime(df_train_clean['time']).dt.year)
    if 'Month' not in df_train_clean:   
        df_train_clean.insert(loc=2, column='Month', value= pd.to_datetime(df_train_clean['time']).dt.month)
    if 'Day' not in df_train_clean:
        df_train_clean.insert(loc=3, column='Day', value= pd.to_datetime(df_train_clean['time']).dt.day)
    if 'Day_date' not in df_train_clean:
        df_train_clean.insert(loc=4, column='Day_date', value= pd.to_datetime(df_train_clean['time']).dt.hour)
        df_train_clean.head(1)

    df_train_clean =  df_train_clean[[#'time', 
            #'Year', 
            'Month', 'Day', 'Day_date', 
            #'Madrid_wind_speed',
        #'Bilbao_rain_1h', 'Valencia_wind_speed', 'Seville_humidity',
        #'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed',
        #'Seville_clouds_all', 'Bilbao_wind_deg', 'Barcelona_wind_speed',
        #'Barcelona_wind_deg', 'Madrid_clouds_all', 'Seville_wind_speed',
        #'Barcelona_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h',
        #'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
        #'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
        #'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
        #'Valencia_pressure', 'Seville_temp_max', 'Madrid_pressure',
        #'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
        #'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
        #'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
        #'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
        #'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min',
        'spain_wind_speed', #'spain_rain_1h',
        'spain_humidity', 
            'spain_clouds_all', 'spain_wind_deg',
        'spain_pressure', 'spain_snow_3h', 'spain_weather_id', #'spain_temp_max',
        #'spain_temp_min', 
            'spain_temp', 'spain_rain_3h', 'load_shortfall_3h',]]

    # ------------------------------------------------------------------------

    return df_train_clean

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
