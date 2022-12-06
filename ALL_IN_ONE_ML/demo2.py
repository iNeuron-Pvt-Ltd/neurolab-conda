import pandas as pd
import numpy as np
from source_code.classifierTrainer import non_hyper_parameter_classifier_model
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
feature = pd.DataFrame(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]))
label = pd.DataFrame(raw_df.values[1::2, 2])
train=non_hyper_parameter_classifier_model()
# a,b=train.model_predicted(feature)
# print(len(a))
# print(len(b))
train.split_data_training(feature,label,hyper_parameter=True)
