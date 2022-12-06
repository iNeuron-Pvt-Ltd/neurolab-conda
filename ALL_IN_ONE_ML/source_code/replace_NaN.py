from sklearn.impute import KNNImputer
import pandas as pd

class replace_nan:
  def __init__(self):
    pass
  def replace_nan_knnimpute(self,data:pd.DataFrame)->pd.DataFrame:
    imputer = KNNImputer(n_neighbors=3)
    columns=data.columns
    After_imputation_data = imputer.fit_transform(data)
    After_imputation_data=pd.DataFrame(data=After_imputation_data,columns=columns)
    return After_imputation_data
  def mean_median_mode(self,data:pd.DataFrame)->pd.DataFrame:
    for col in data.columns:
      if data[col].isna().sum()>0:
        if data[col].nunique()/len(data)>=0.1:
          
          data[col].fillna(data[col].mode(),inplace=True)
          
        else:
          print(data[col].mean())
          data[col].fillna(data[col].mean(),inplace=True)
    return data
