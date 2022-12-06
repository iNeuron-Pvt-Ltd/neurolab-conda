import pandas as pd
from path_name_provoiders.all_names import corr_remove_columns_list as remove_col_li
from path_name_provoiders.all_names import corrrelation_dict as corr_dic
from path_name_provoiders.all_names import most_corrrelation_dict as most_corr_dic

class find_correlation:
  def __init__(self):
    pass
  def _retun_corr(self,feature:pd.DataFrame):
    try:
      all_col=list(feature.columns)
      
      for i in feature.columns:
        for j in feature.columns:
          if i!=j:
            # print(f'{i} vs {j}')
            temp_data=pd.concat((feature[i],feature[j]),axis=1)
            corr_score=round(temp_data.corr()[j][0],5)
            # print()
            # print('_____________________________________________')
            if (corr_score not in corr_dic.values()) or (corr_score not in most_corr_dic.values()):
              corr_dic.update({f'{i} vs {j}':corr_score})
              if corr_score > 0.8:
                if (i in all_col) and (j in all_col):
                  remove_col_li.append(i)
                  all_col.remove(i)
      

                most_corr_dic.update({f'{i} vs {j}':corr_score})
        
      print(remove_col_li)
      return corr_dic,most_corr_dic,remove_col_li
    except Exception as e:
      raise e
  def remove_corr_col(self,data)->pd.DataFrame:
    try:
      _,_,corr_list=self._retun_corr(data)
      print(corr_list)
      return data.drop(columns=corr_list)
    except Exception as e:
      raise e