import pandas as pd
import os
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.cluster import KMeans
from source_code.hyper_parameter import hyper_parameter_classifier
from model_folder.classification import svc,logisticRegression,randomForestClassifier,xgbClassifier,knnClassifier,decisionTreeClassifier,naive_bayes_Gaus,naive_bayes_Mul
from model_folder.regression import linearRegression,randomForestRegressor,svr,kneighborsRegressor,randomForestRegressor,decisiontreeregressor

class non_hyper_parameter_classifier_model(hyper_parameter_classifier):
  def __init__(self):
    self.hyper_parameter_classifier_obj=hyper_parameter_classifier()
    self._final_all_model_dic=dict()
    self.model_dict=dict()
  def _model_created(self,isClassification=True,hyper_parameter=dict()):
    model_list=[]
    try:
      if isClassification:
        log=logisticRegression
        random_forest=randomForestClassifier
        desicion_treee=decisionTreeClassifier
        knn=knnClassifier

        model_list.extend((log,random_forest,desicion_treee,knn))
    
      elif isClassification==False:
        linearregressor=linearRegression
        random_forest=randomForestRegressor
        desicion_treee=decisiontreeregressor
        knn=kneighborsRegressor
        svc=svr
        model_list.extend((linearregressor,random_forest,desicion_treee,knn,svc))
      # xgb_clf=xgbClassifier
      return model_list

    except Exception as e:
      raise e


  def _default_model_para_training(self,feature:pd.DataFrame,label=None,hyper_parameter=dict()):
    model_score=dict()
    try:
      if len(hyper_parameter)!=0:
        if (label.nunique()/len(label)>0.1):
          print('regressor')
          model_li=self._model_created(isClassification=False)
          
        else:
          
          model_li=self._model_created()
        #model_name=[str(model).replace('()','').lower() for model in model_li]
        model_name=[]
        for modelname in model_li:
          MN_with_para=str(modelname).replace('()','').lower()
          MN_without_para=MN_with_para.split('(')[0]
          
          model_name.append(MN_without_para)

        for model_name in model_name:
          for model in model_li:
            if model_name==str(model).replace('()','').lower():
            
              model_hyperParaMeter=hyper_parameter[model_name]

              model.set_params(**model_hyperParaMeter)
        print(label.unique())
        fit_model=[i.fit(feature,label) for i in model_li]
        [model_score.update({i:i.score(feature,label)}) for i in model_li]
        find_max_accuracy_model=max(model_score,key=model_score.get)

        return find_max_accuracy_model

      else:
        model_li=self._model_created(isClassification=True)
        fit_model=[i.fit(feature,label) for i in model_li]
        [model_score.update({i:i.score(feature,label)}) for i in model_li]
        find_max_accuracy_model=max(model_score,key=model_score.get)
        return find_max_accuracy_model
    except Exception as e:
      raise e

  def _cluster_data(self,data:pd.DataFrame,n_groups):
    try:
      kmeans=KMeans(n_clusters=n_groups)
      #data=data.drop(['outcome','kmeans_label'],axis=1)
      kmeans_label=kmeans.fit_predict(data)
      
      path=os.path.join('KMeans_model_dir')
      os.makedirs(path,exist_ok=True)
      file_name=f'kMeans.pkl'
      print(f'path of file {path}/{file_name}')
      joblib.dump(kmeans,path+'/'+file_name)
      self._final_all_model_dic.update({'kmeans_model':kmeans})
      return kmeans_label
    except Exception as e:
      raise e

  def _divide_groups(self,df:pd.DataFrame,col_name:str) -> list:
    try:
      final_li=[]
      catg=df[col_name].value_counts().index
      for i in catg:
        new_data=df[df[col_name]==i]
        final_li.append(new_data.drop(col_name,axis=1))
      return final_li
    except Exception as e:
      raise e
  


  def _helper_model_predicted(self,kmeans_path:str,model_path:str,feature):

    final_output_list,final_df_list=[],[]
    feature_copy=feature.copy()
    isExist_Kmeans= os.path.exists(kmeans_path)
    isExist_Model= os.path.exists(model_path)
    try:
      if isExist_Kmeans and isExist_Model:
          kmeans_model_list=list(os.listdir(kmeans_path))
          print(kmeans_model_list)
          trained_model_list=os.listdir(model_path)
          if len(kmeans_model_list)!=0 and len(trained_model_list)!=0:
          
            for kmeans_model in kmeans_model_list:
              print(kmeans_model)
              if '.pkl' in kmeans_model:

                path=os.path.join(kmeans_path,kmeans_model)
                loaded_kmean_model=joblib.load(path)

                kmean_label=loaded_kmean_model.predict(feature_copy)
                print(kmean_label)
                feature_copy['kmean_label']=kmean_label
                unique_val=feature_copy['kmean_label'].unique()
                for unique in unique_val:
                  kmeans_label_feature=feature_copy[feature_copy['kmean_label']==unique]

                  for model in list(trained_model_list):
                    if f'kmeans_model_{unique}_' in model:
                      model=joblib.load(os.path.join(model_path,model))
                      kmean_feature=kmeans_label_feature.drop(columns=['kmean_label'])

                      predicted_val= model.predict(kmean_feature)
                      print(predicted_val)
                      final_output_list.append(predicted_val)
                      final_df_list.append(kmean_feature)
    

      return final_output_list,final_df_list
    except Exception as e:
      raise e

  def model_predicted(self,feature:pd.DataFrame):
    feature_copy=feature.copy()

    inp=input('if you provide specific path of kmeans model and trained model yes[Y] or no[N] --->   ')
    try:
      if inp.lower()=='y':
        kmeans_path=input('enter the KMeans dir path')
        model_path=input('enter the trained model dir path')
        final_out,final_df=self._helper_model_predicted(kmeans_path=kmeans_path,model_path=model_path,feature=feature_copy)
        
        return final_out,final_df

      elif inp.lower()=='n':
        
        kmeans_path=os.path.join(os.getcwd(),'KMeans_model_dir')
        model_path=os.path.join(os.getcwd(),'model_dir')
        print(kmeans_path,model_path)
        final_out,final_df=self._helper_model_predicted(kmeans_path=kmeans_path,model_path=model_path,feature=feature_copy)
        return final_out,final_df

    except Exception as e:
      raise e

  def split_data_training(self,feature:pd.DataFrame,label=None,predict=False,hyper_parameter=False):

    try:
      if hyper_parameter:
        all_best_parameter_dict=dict()
        kmeans_label=self._cluster_data(data=feature,n_groups=2)
        feature['outcome']=label
        feature['kmeans_label']=kmeans_label
        if (label.nunique()/len(label)>0.1).values[0]:
          all_model_name=['linearregression','randomforestregressor','svr','decisiontreeregressor','kneighborsregressor']

        else:
          all_model_name=['logisticregression','decisiontreeclassifier','randomforestclassifier','svc','xgbclassifier','kneighborsclassifier']

        for model_name in all_model_name:

          best_para_of_model=self.hyper_parameter_classifier_obj.hyper_parameter_tuneing_classifier(model_name,x=feature,y=label)
          all_best_parameter_dict.update({model_name:best_para_of_model})
        
        split_data=self._divide_groups(feature,'kmeans_label')
        all_data_li=[]
        for data,val in zip(split_data,feature['kmeans_label'].value_counts().index):
          split_feature=data.drop(columns='outcome')
          split_label=data['outcome']
          model_obj=self._default_model_para_training(split_feature,split_label,hyper_parameter=all_best_parameter_dict)
          self.model_dict.update({f'kmeans_model_{val}':model_obj})
        ct = datetime.datetime.now()
        time_stamp=str(ct).replace(' ','_').replace('-','_').replace(':','_').replace('.','_')
        for model_name,model in self.model_dict.items():
          path=os.path.join('model_dir')
          os.makedirs(path,exist_ok=True)
          file_name=f'{time_stamp}_{model_name}_{str(model).replace("()","")}.pkl'
          print(f'path of file {path}/{file_name}')
          joblib.dump(model,path+'/'+file_name)
      return self.model_dict

    except Exception as e:
      raise e
    
  def model_score(self,feature:pd.DataFrame,label:pd.Series):
    pass
  