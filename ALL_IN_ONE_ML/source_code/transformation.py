from sklearn.preprocessing import StandardScaler
import pandas as pd
class transformation:
  def __init__(self):
    pass
  def log_dist(self,data:pd.DataFrame)->pd.DataFrame:
    return data/2.7183
  def std_scaler_dist(self,data:pd.DataFrame)->pd.DataFrame:
    columns=data.columns
    scaler=StandardScaler()
    np_transform_data=scaler.fit_transform(data)
    return pd.DataFrame(data=np_transform_data,columns=columns)