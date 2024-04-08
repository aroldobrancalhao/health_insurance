import pandas as pd
import numpy as np
import pickle
import os


class HealthInsurance:
    def __init__( self ):
        # carregar todos os arquivos pickle de transformação de dados
        self.home_path =  ''
        self.age_scaler = pickle.load( open( self.home_path + 'features/age_reescaling.pkl', 'rb') )
        self.region_encoder = pickle.load( open( self.home_path + 'features/encoder_region_coder.pkl', 'rb') )
        self.sales_channel_encoder = pickle.load( open( self.home_path + 'features/encoder_sales_channel.pkl', 'rb') )
        self.annual_premium_scaler = pickle.load( open( self.home_path + 'features/scaler_annual_premium.pkl', 'rb') )
        self.vintage_reescaling = pickle.load( open( self.home_path + 'features/vintage_reescaling.pkl', 'rb') )
    
    def feature_engineering(self, df2):
        # vehicle_damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        return df2
    
    def data_preparation( self, df4 ):
        # anual premium - StandarScaler
        df4['annual_premium'] = self.annual_premium_scaler.transform( df4[['annual_premium']].values )

        # Age - MinMaxScaler
        df4['age'] = self.age_scaler.transform( df4[['age']].values )

        # Vintage - MinMaxScaler
        df4['vintage'] = self.vintage_reescaling.transform( df4[['vintage']].values )

        # gender - One Hot Encoding / Target Encoding
        list_gender = {'Male':1,'Female':0}
        df4.loc[:, 'gender'] = df4['gender'].map( list_gender )

        # region_code - Target Encoding / Frequency Encoding
        df4.loc[:, 'region_code'] = df4['region_code'].map( self.region_encoder )

        # vehicle_age - One Hot Encoding / Frequency Encoding
        df4 = pd.get_dummies( df4, prefix='vehicle_age', columns=['vehicle_age'] )

        # policy_sales_channel - Target Encoding / Frequency Encoding
        df4.loc[:, 'policy_sales_channel'] = df4['policy_sales_channel'].map( self.sales_channel_encoder )
        
        # Feature Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured',
                         'policy_sales_channel']
        
        return df4[ cols_selected ]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
    
    