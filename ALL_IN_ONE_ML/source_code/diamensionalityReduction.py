from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import pandas as pd

class diamensionality_reduction:
  def __init__(self):
    pass
  def pca_pipe(self,feature:pd.DataFrame)->pd.DataFrame:
    scaler=StandardScaler()
    pipe=make_pipeline(scaler,PCA(n_components = 0.5))
    x_pca=pipe.fit_transform(feature)
    return x_pca