from imblearn.combine import SMOTEENN,SMOTETomek
import pandas as pd
class handle_imbalanced_data:
  
  def __init__(self):
    pass
  def using_smoteen(self,feature:pd.DataFrame,label:pd.Series)->pd.DataFrame:
    smk = SMOTEENN()
    X_res,y_res=smk.fit_resample(feature,label)
    return X_res,y_res
  def using_smotetomek(self,feature:pd.DataFrame,label:pd.Series)->pd.DataFrame:
    smk = SMOTETomek()
    X_res,y_res=smk.fit_resample(feature,label)
    return X_res,y_res