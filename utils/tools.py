from jaxtyping import Integer, Array
import pandas as pd

def prediction_to_csv(prediction: Integer[Array, "M"]) -> None:
    Yte = {'Prediction' :prediction } 
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv('Yte_pred.csv',index_label='Id')

