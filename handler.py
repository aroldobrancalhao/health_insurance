import pickle
import pandas as pd
import numpy as np
import os
import sys
from flask             import Flask, request, Response
# Adding the 'healthinsurance' directory to the path
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'healthinsurance'))
from healthinsurance.HealthInsurance import HealthInsurance

#path = ''
model = pickle.load(open('models/model_health_insurance.pkl', 'rb'))

# initialize API
app = Flask( __name__ )

@app.route( '/predict', methods=['POST'] )

def PredictCross():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

        #test_raw1 = df_api_test.drop('id',axis=1)

        pipeline = HealthInsurance()

        dfa = pipeline.feature_engineering(test_raw)

        dfb = pipeline.data_preparation(dfa)

        df_response = pipeline.get_prediction(model, test_raw, dfb)


        return df_response
    else:
        return Reponse( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( '0.0.0.0' )