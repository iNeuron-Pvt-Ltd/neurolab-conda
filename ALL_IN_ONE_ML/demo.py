from source_code.hyper_parameter import hyper_parameter_classifier
from source_code.detect_outlierANDremove import detect_remove_outliers
from source_code.handle_imbalanced_dataset import handle_imbalanced_data
from source_code.diamensionalityReduction import diamensionality_reduction
from source_code.remove_unwntedColumns import remove_col
from source_code.find_Corr_remove import find_correlation
from source_code.transformation import transformation
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
from source_code.replace_NaN import replace_nan
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import joblib
import sys,os
# data=load_iris(as_frame=True)
# feature1=data.data
# label1=data.target
# # print(feature.head(1))
# train=non_hyper_parameter_classifier_model()
# #a,b=train.model_predicted(feature1)
# train.split_data_training(feature1,label1,hyper_parameter=True)

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
feature = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]))
label = pd.DataFrame(raw_df.values[1::2, 2])
# load_model=joblib.load('/config/workspace/ALL_IN_ONE_ML/model_dir/2022_12_06_13_17_47_891414_kmeans_model_0_DecisionTreeRegressor(ccp_alpha=0.001, min_samples_leaf=3, min_samples_split=3).pkl')
# print(load_model.predict(feature))
train=non_hyper_parameter_classifier_model()
# a,b=train.model_predicted(feature)
# print(len(a))
# print(len(b))
train.split_data_training(feature,label,hyper_parameter=True)

print('===================================================')

# hyp=hyper_parameter_classifier()
# best=hyp.hyper_parameter_tuneing_classifier('xgbclassifier',feature,label)
# print(best)


#dummy(10, 0)
# dr=diamensionality_reduction()
# df=dr.pca_pipe(feature)
# print(df.shape)
# print(feature.shape)
# imba=handle_imbalanced_data()
# x,y=imba.using_smotetomek(feature, label)
# print(feature.shape)
# print(label.shape)
# print('_________________')
# print(x.shape)
# print(y.shape)
# out=detect_remove_outliers()
# df=out.remove_outlier(feature)
# print(feature.shape)
# print(df.shape)
# print(df.head())
# print('_________________')
# print(feature.head())
# hyper=hyper_parameter_classifier()
# hyper.hyper_parameter_tuneing_classifier('xgbclassifier',feature,label)

