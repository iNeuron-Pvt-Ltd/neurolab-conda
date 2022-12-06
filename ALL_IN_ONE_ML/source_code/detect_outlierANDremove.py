from path_name_provoiders.all_names import outlier_list as user_outlier_li
from path_name_provoiders.all_names import outlier_index_list as ind_li
from path_name_provoiders.all_names import outlier_index_dict as outlier_ind_dic
from path_name_provoiders.all_names import outlier_column_percentage_dic as col_per_dic
import pandas as pd
import numpy as np
class detect_remove_outliers:
  def __init__(self):pass
  def _detect_outlier(self,data:pd.DataFrame)->list:
    try:
      df=data
    
      for col in df.columns:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        IQR=q3-q1
        ucl=q3+1.5*IQR
        lcl=q1-1.5*IQR
        
        ind=np.where((df[col]<lcl) | (df[col]>ucl))[0]
        
        per=(len(ind)/len(df))*100
        if len(ind)!=0:
          [ind_li.append(i) for i in ind]
          outlier_ind_dic.update({col:ind})
        col_per_dic.update({col:f'percentage {per}'})
      
      user_outlier_li.append(outlier_ind_dic)
      user_outlier_li.append(col_per_dic)  
      return list(set(ind_li)),user_outlier_li
    except Exception as e:
      raise e
  def remove_outlier(self,data:pd.DataFrame)->pd.DataFrame:
    try:
      ind_li,_=self._detect_outlier(data)
      new_dataframe=data.drop(ind_li)
      return new_dataframe
    except Exception as e:
      raise e

