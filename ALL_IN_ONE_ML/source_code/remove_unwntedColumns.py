import pandas as pd
class remove_col:
  def __init__(self):
    pass
  def _remove_col_zero_std(self,data:pd.DataFrame)->list:
    remove_col=[col for col in data.columns if data[col].std()==0.0]
    return data.drop(remove_col,axis=1)
  def _remove_col_maxNan_val(self,data:pd.DataFrame)->list:
    remove_col=[col for col in data.columns if data[col].isna().sum()>len(data)/2]
    return data.drop(remove_col,axis=1)
  def _continuous_data_remove(self,data:pd.DataFrame):
    for col in data.columns:
      if set(data.index+1)==set(data[col]):
        final_data=data.drop(col,axis=1,inplace=True)
        return final_data
      return data
  def all_columns_remove(self,data:pd.DataFrame)->pd.DataFrame:
    remove_col_zero_std_data=self._remove_col_zero_std(data)
    remove_col_maxNan_val_data=self._remove_col_maxNan_val(remove_col_zero_std_data)
    final_dataframe=continuous_data_remove_data=self._continuous_data_remove(remove_col_maxNan_val_data)
    return final_dataframe
