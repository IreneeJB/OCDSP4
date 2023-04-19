import pandas as pd
import numpy as np
import missingno as msno

#Sci-Kit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

from category_encoders import TargetEncoder

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer
from sklearn.compose import TransformedTargetRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

##############################################################################
#  initializePandas() :
#         Aucun paramètres
#
#         Initialise les options pandas
# 
#         Return : None
##############################################################################

def initializePandas() :
    pd.set_option('display.max_columns', 10)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000
    pd.set_option('display.max_colwidth', 30)  # or 199
    return None
    
    
##############################################################################
#  compareColumns(df, L) :
#         df : pd.dataframe
#         L : liste de string de noms de colomnes de data
#
#         Affiche le nombre de valeurs présente dans une colonnes et absentes dans l'autre
#          
#
#         Return : None
##############################################################################

def compareColumns(df, L) :
    for e1 in L :
        for e2 in L:
            if e1 != e2 :
                try :
                    mask = df[e1].notna()
                    print(f'il y a {df[mask][e2].isna().sum()} valeurs dans {e1} qui sont manquantes dans {e2}.')
                except KeyError :
                    print(f"Erreur de clé, couple {e1} - {e2} non traité.")
            else :
                pass
    return None

##############################################################################
#  missingValuesInfos(df) :
#         df : pd.dataframe
#
#         Affiche le nombre de valeurs manquantes, totales, le taux de remplissage et la shape du dataframe
#         Affiche la msno.matrix du dataframe          
#
#         Return : None
##############################################################################

def missingValuesInfos(df) :
    nbRows, nbCols = df.shape
    print(f"Il y a {df.isna().sum().sum()} valeurs manquantes sur {nbRows * nbCols} valeurs totales.")
    print(f"Le taux de remplissage est de : {int(((nbRows*nbCols - df.isna().sum().sum())/(nbRows*nbCols))*10000)/100} %")
    print("Dimension du dataframe :",df.shape)
    msno.matrix(df)
    return None

##############################################################################
# gridExtract(df) :
#         df : pd.dataframe
#
#         
#
#         Return : None
##############################################################################

def gridExtract(df):
    df.dropna(inplace = True)
    print("Temps d'entrainement minimum :",df["mean_fit_time"].min())
    print("Score pr ce temps :\n",df.loc[df["mean_fit_time"] == df["mean_fit_time"].min(),['mean_test_neg_root_mean_squared_error','mean_test_r2']])
    
    print("\nMeilleur r2 score : ",df['mean_test_r2'].max())
    print(df.loc[df['rank_test_r2'] == 1,['mean_test_r2','mean_test_neg_root_mean_squared_error']])
    
    print("\nMeilleur RMSE score : ",df['mean_test_neg_root_mean_squared_error'].abs().min())
    print(df.loc[df['rank_test_neg_root_mean_squared_error'] == 1,['mean_test_r2','mean_test_neg_root_mean_squared_error']])

def saveBestModel(name, gridResults, resultsDF):
    L = ['params','mean_train_neg_root_mean_squared_error','mean_train_r2','mean_test_neg_root_mean_squared_error','mean_test_r2','mean_fit_time']
    resultsDF.loc[name,:] = gridResults.loc[gridResults['rank_test_neg_root_mean_squared_error'] == 1,L].reset_index(drop = True).loc[0,:].to_numpy()
    return resultsDF
                                         
def log1(x) :
    return np.log(1+x)

def exp1(x) :
    return np.exp(x)-1


##############################################################################
#  getColumnNames(model, choixEncoder, X_train, poly = False) :
#         model: pipeline  -  model utilisé
#         choixEncoder : str  -  "TargetEncoder"  ou "OneHotEncoder" 
#         X_train : pd.dataframe  -  data d'entrée de model
#         poly : bool  -  utilisation de polynomial features?
#
#         Renvoie le nom de colonnes a mettre sur les shap_values
#                 
#
#         Return : list or index or iterable
##############################################################################

def getColumnNames(model, choixEncoder, X_train, poly = False):
    if poly :
        if choixEncoder == "TargetEncoder" :
            return model['polynomialfeatures'].get_feature_names_out(X_train.columns)
        else :
            return model['columntransformer'].get_feature_names_out()
    else :
        if choixEncoder == "TargetEncoder" :
            return X_train.columns
        else :
            return model['columntransformer'].get_feature_names_out()