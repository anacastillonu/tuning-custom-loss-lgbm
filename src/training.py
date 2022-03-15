import pandas as pd
import numpy as np
import dill
import types
from typing import Union
import random
import spikeval
import shap
import yaml
import sys
import warnings

from dataclasses import dataclass

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression

from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import IsotonicRegression
from sklearn.calibration import calibration_curve

from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

import optuna
import optuna.integration.lightgbm as olgb
import lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier
from scipy.misc import derivative

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


@dataclass
class ClassifierModel:
    '''
    Stores the demand model including: 
        model_name: str
            unique identifier of the model. The final pkl file will be named with this identifier
        input_data: DataFrame
            DF with 100% of the data used to train/test the model
        test_data: DataFrame
            subset of data used to test (this data will also be used in test optimizations)
        model_settings: dict
            dictionary of model settings defined in the training notebook
        hyperparams: dict
            dictionary of model's hyperparams
        preprocessor: 
            sklearn's Column Tranformer for categorical and numerical columns in the model
        lgbm_model:
            lgbm model (without calibration)
        calibrator:
            IsosnicRegression to calibrate probabilities
        test_predictions: DataFrame
            DF with predictions for the different model stages (i.e. lgbm, calibration, interpolation)
        metrics: DataFrame
            DF with the average_precision_score, auc and logloss metrics. 
    '''
    
    # MODEL SETUP
    model_name: str = None
    model_settings: dict = None
    input_data: pd.DataFrame = None
    #train_data: pd.DataFrame = None
    test_data: pd.DataFrame = None
    hyperparams: dict = None
    
    # MODEL ARTIFACTS
    preprocessor: sklearn.compose.ColumnTransformer = None
    lgbm_model: lightgbm.sklearn.LGBMClassifier = None
    calibrator: sklearn.calibration.IsotonicRegression = None
        
    #RESULTS 
    test_predictions: pd.DataFrame = None 
    metrics: pd.DataFrame = None

def load_and_split_data(model: ClassifierModel, 
                        testing_fraction = 0.3, 
                        valid_fraction = 0.10):
    """_summary_

    Args:
        model (ClassifierModel): _description_
        testing_fraction (float, optional): _description_. Defaults to 0.9.
        valid_fraction (float, optional): _description_. Defaults to 0.2.

    Returns:
        train(pd.Dataframe): _description_
        validate(pd.Dataframe): _description_
        test(pd.Dataframe): _description_
    """

    test_ids =  model.input_data.loc[:, model.model_settings['id_columns']].sample(frac=testing_fraction, random_state=42)
    train_ids = model.input_data.loc[:, model.model_settings['id_columns']][~model.input_data.loc[:,model.model_settings['id_columns']].isin(test_ids.to_dict('list')).all(axis=1)]
    
    test = model.input_data.merge(test_ids.set_index(model.model_settings['id_columns']), on = model.model_settings['id_columns'], how = 'inner')
    train = model.input_data.merge(train_ids.set_index(model.model_settings['id_columns']), on = model.model_settings['id_columns'], how = 'inner')
    
    
    # Select ID's that will be included in the calibration dataset
    validation_ids = train.loc[:, model.model_settings['id_columns']].sample(frac=valid_fraction, random_state=42)
    validate = train.merge(validation_ids.set_index(model.model_settings['id_columns']), on = model.model_settings['id_columns'], how = 'inner')
    
    print('train dataset size: {}'.format(train.shape[0]))
    print('validate dataset size: {}'.format(validate.shape[0]))
    print('test dataset size: {}'.format(test.shape[0]))
    
    # Delete variables that are no longer necessary
    del validation_ids, train_ids, test_ids
    return train, validate, test

def create_preprocessor(
    train: pd.DataFrame,
    model_settings: dict) -> ColumnTransformer:
    """_summary_

    Args:
        train (pd.DataFrame): _description_
        model_settings (dict): _description_

    Returns:
        ColumnTransformer: _description_
    """
    
    #For categorical variables, we first need to impute NaN values, and then encode.
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    
    #For numerical variables, we only use an imputer (no need to encode or scale)
    numeric_transformer = SimpleImputer(strategy="most_frequent")
    
    #Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, model_settings["categorical_columns"]),
            ("num", numeric_transformer, model_settings["numerical_columns"]),
        ]
    )
    
    #Fit preprocessor to existing data
    preprocessor.fit(train.loc[:, model_settings["categorical_columns"]+model_settings["numerical_columns"]])

    return preprocessor

def set_lgbm_train_valid_dsets(model, train, validate):
   
    # Transform x with preprocessor
    X_train_transformed = pd.DataFrame(
        model.preprocessor.transform(
            train.loc[:,model.model_settings["categorical_columns"]
                        +model.model_settings["numerical_columns"]]),
        columns=model.model_settings["categorical_columns"]+model.model_settings["numerical_columns"])

    X_valid_transformed = pd.DataFrame(
        model.preprocessor.transform(
            validate.loc[:,model.model_settings["categorical_columns"]
                        +model.model_settings["numerical_columns"]]),
        columns=model.model_settings["categorical_columns"]+model.model_settings["numerical_columns"])

    # Create LGBM datasets
    train_dset = lgb.Dataset(
        pd.DataFrame(X_train_transformed,
                    columns=model.model_settings["categorical_columns"]+model.model_settings["numerical_columns"]),
        categorical_feature=model.model_settings["categorical_columns"],
        label=train.loc[:,model.model_settings["label_column"]],
        free_raw_data=False)

    valid_dset = lgb.Dataset(
        pd.DataFrame(X_valid_transformed, 
                    columns=model.model_settings["categorical_columns"]+model.model_settings["numerical_columns"]),
        categorical_feature=model.model_settings["categorical_columns"],
        label= validate.loc[:,model.model_settings["label_column"]],
        free_raw_data=False)
    
    del X_train_transformed, X_valid_transformed

    return train_dset, valid_dset

def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    """
    Adapation of the Focal Loss for lightgbm to be used as loss function.
    Parameters:
    -----------
        y_pred: numpy.ndarray
            array with the predictions
        dtrain: lightgbm.Dataset
        alpha, gamma: float
            See original paper https://arxiv.org/pdf/1708.02002.pdf
    
    Returns:
    ---------
        grad: float, 
        hess: float
        
    """
    
    a,g = alpha, gamma
    y_true = dtrain.label
    
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    
    return grad, hess

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    """
    Adapation of the Focal Loss for lightgbm to be used as evaluation loss.
    Parameters:
    -----------
        y_pred: numpy.ndarray
            array with the predictions
        dtrain: lightgbm.Dataset
        alpha, gamma: float
            See original paper https://arxiv.org/pdf/1708.02002.pdf
    
    Returns:
        name: str, 
        eval_result: float, 
        is_higher_better: bool  
        
    """
    a,g = alpha, gamma
    y_true = dtrain.label

    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    
    # (eval_name, eval_result, is_higher_better)
    return 'focal_loss', np.mean(loss), False

def predict_proba(self, dataframe):
    '''
    custom to predict probability from lgbm raw results
    (lgbm returns raw results when using a custom loss function)
    '''
    raw = self.predict(dataframe)
    return 1/(1+np.exp(-raw))

def dual_loss_and_hyperparams_objective(trial, 
                                        train_dset,
                                        valid_dset,
                                        model_params,
                                        predict_proba_func = predict_proba):

    params_for_tuning =  {"lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                           "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                           "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                           "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                           "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                           "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                           "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
                           "alpha":trial.suggest_float('alpha',0.25, 0.75),
                           "gamma":trial.suggest_float('gamma',0, 10)
                          }
    
    model_params.update(params_for_tuning)

    focal_loss = lambda x,y: focal_loss_lgb(x, y, params_for_tuning['alpha'], params_for_tuning['gamma'])
    focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, params_for_tuning['alpha'], params_for_tuning['gamma'])
    
    lgbm_model = lgb.train(params=model_params,
                           
                           train_set=train_dset,
                           valid_sets=[valid_dset,],
                           
                           fobj = focal_loss,
                           feval = focal_loss_eval,
                           
                           early_stopping_rounds=25,
                           #verbose_eval = False
                          )

    lgbm_model.predict_proba = types.MethodType(predict_proba_func, lgbm_model)
    
    fpr, tpr, thresholds = roc_curve(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))
    metric = auc(fpr, tpr)

    return metric


def loss_function_objective(trial,
                            train_dset,
                            valid_dset,
                            model_params,
                            predict_proba_func = predict_proba):
    """_summary_

    Args:
        trial (_type_): _description_
        model_settings (_type_): _description_
        train_dset (_type_): _description_
        valid_dset (_type_): _description_
        model_params (_type_): _description_
        predict_proba_func (_type_, optional): _description_. Defaults to predict_proba.

    Returns:
        _type_: _description_
    """
    
    # TUNNING
    # -------------------
    alpha = trial.suggest_float('alpha',0.25, 0.75)
    gamma = trial.suggest_float('gamma',0, 10)

    focal_loss = lambda x,y: focal_loss_lgb(x, y, alpha, gamma)
    focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, alpha, gamma)
    
    lgbm_model = lgb.train(params=model_params,
                           
                           train_set=train_dset,
                           valid_sets=[valid_dset,],
                           
                           fobj = focal_loss,
                           feval = focal_loss_eval,
                           
                           early_stopping_rounds=25,
                           #verbose_eval = False
                          )

    lgbm_model.predict_proba = types.MethodType(predict_proba_func, lgbm_model)
    

    # CALIBRATE
    # -------------------
    #isotonic = IsotonicRegression(out_of_bounds='clip',
                                  #y_min=0,
                                  #y_max=1)
    
    #isotonic.fit(sigmoid(lgbm_model.predict(calib_dset.data)), 
                 #calib_dset.label.iloc[:, 0])
    
    # EVAL
    # -------------------
    #metric = log_loss(valid_dset.label, isotonic.predict(sigmoid(lgbm_model.predict(valid_dset.data))))

    
    fpr, tpr, thresholds = roc_curve(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))
    metric = auc(fpr, tpr)
   
    #metric = log_loss(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))
    
    #metric = average_precision_score(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))
        
    return metric

def hyperparams_objective(trial,
                          train_dset,
                          valid_dset,
                          model_params,
                          predict_proba_func = predict_proba):
    '''
    objective when tunning the model's hyperparameters
    it suggests a set of parameters (l1, l2, num_leaves, feature_fraction, bagging_fraction, bagging_freq, min_child_samples)
    it trains a new model, then it returns a metric evaluated over the validation set
    
    parameters
    ----------
        trial: optuna trial
            trial in optuna's study for the loss function
        model_settings: dict
            dictionary with model's settings, 
            including information about the metric to be used for evaluation
            and whether the model should be calibrated during tuning
        train_dset: lightgbm.dataset
            ligthGBM dataset with training data
        valid_dset:lightgbm.dataset
            ligthGBM dataset with validation data
        model_params:dict
            model parameters to be used for training on this stage 
            thus, it includes the predifined alpha and gamma
        
    returns
    --------
        metric: float
            evaluation metric optained over the validation dataset
    
    '''
    # TUNNING
    # -------------------
    focal_loss = lambda x,y: focal_loss_lgb(x, y, model_params['alpha'], model_params['gamma'])
    focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, model_params['alpha'], model_params['gamma'])
    
    params_for_tunning =  {"lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                           "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                           "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                           "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                           "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                           "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                           "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
                          }
    
    model_params.update(params_for_tunning)
    
    lgbm_model = lgb.train(params=model_params,
                           
                           train_set=train_dset,
                           valid_sets=[valid_dset,],
                           
                           fobj = focal_loss,
                           feval = focal_loss_eval,
                           
                           early_stopping_rounds=25,
                           #verbose_eval = False
                           )
    
    lgbm_model.predict_proba = types.MethodType(predict_proba_func, lgbm_model)
    
    # CALIBRATE
    # -------------------
    #isotonic = IsotonicRegression(out_of_bounds='clip',
                                  #y_min=0,
                                  #y_max=1)
    
    #isotonic.fit(sigmoid(lgbm_model.predict(calib_dset.data)), 
                 #.label.iloc[:, 0])
    
    # EVAL
    # -------------------
    #metric = log_loss(valid_dset.label, isotonic.predict(sigmoid(lgbm_model.predict(valid_dset.data))))
    
    fpr, tpr, thresholds = roc_curve(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))
    metric = auc(fpr, tpr)
   
    #metric = log_loss(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))
    
    #metric = average_precision_score(valid_dset.label, lgbm_model.predict_proba(valid_dset.data))

    return metric

def iterate_for_optimal_boosting_rounds(params,
                                        model_settings,
                                        train_dset,
                                        valid_dset,
                                        predict_proba_func = predict_proba
                                       ):
    '''
    run N interations with early stopping rounds 
    and store the best boosting round for each iteration
    
    parameters
    ----------
        
        model_settings: dict
            dictionary with model's settings, 
            including information number of interations to run
        train_dset: lightgbm.dataset
            ligthGBM dataset with training data
        valid_dset:lightgbm.dataset
            ligthGBM dataset with validation data
        model_params:dict
            model parameters to be used for training on this stage 
            thus, it includes the predifined alpha and gamma
        
    returns
    --------
        mean(best_iterations): int
            average of best iterations
    
    '''
    #Create list to store best iterations from all runs
    best_iterations = list()
    focal_loss = lambda x,y: focal_loss_lgb(x, y, alpha=params['alpha'], gamma=params['gamma'])
    focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, alpha=params['alpha'], gamma=params['gamma'])
    
    params['learning_rate']=0.01
    #Run several to obtain best iteration from each round
    for seed in np.arange(model_settings["n_iters_boost_rounds"]):
        
        params['seed']=seed


        classifier = lgb.train(params=params,
                               
                               train_set=train_dset,
                               valid_sets=[valid_dset,],
                               
                               fobj = focal_loss,
                               feval = focal_loss_eval,
                               
                               early_stopping_rounds=25,
                               num_boost_round=10000,
                               
                               #verbose_eval = False
                               )
        print(f"Finished trial. Best iteration: {classifier.best_iteration}")
        best_iterations.append(classifier.best_iteration)
        
    del classifier
    
    return int(np.median(best_iterations))

def get_study_results(study:optuna.study)->dict:
    """_summary_

    Args:
        study (optuna.study): _description_

    Returns:
        dict: _description_
    """

    trial = study.best_trial
    study_params = trial.params

    fig = optuna.visualization.plot_slice(study, params=study_params)
    fig.show()
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    
    print(f"-------------------------------------------------------- \n Number of finished trials: {len(study.trials)}")
    
    print("Best trial:")

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in study_params.items():
        print("    {}: {}".format(key, value))

    print("--------------------------------------------------------")

    return study_params


def model_tunning_focal_loss_2_steps(model, train, validate):
    """
    Process hyperparameter tunning for Lgbm
    Parameters:
    -----------
        model_name: str
            model name as deffined in model settings
        train: DataFrame,
            DF with training data
        validate: DataFrame,
            DF with data for tunning purposes.
            
    Returns:
    -----------
        params: dict,
            Tunned Hyperparams
        
    """
    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------
    
    
    model_params = {"boosting": "gbdt",
                    "deterministic":True,
                    "num_iterations": 10000, #Will have early stopping for tunning
                    'learning_rate':0.1,
                    'verbosity':-1}
    
    print(model_params)
    
    train_dset, valid_dset = set_lgbm_train_valid_dsets(model, train, validate)
    
    # -----------------------------------------------------
    # TUNE LOSS FUNCTION
    # -----------------------------------------------------
    
    loss_function_study = optuna.create_study(direction='maximize')
    loss_function_study.optimize(lambda trial: 
                   loss_function_objective(trial,
                                           train_dset,
                                           valid_dset,
                                           model_params),
                                 n_trials=model.model_settings["n_iters_loss"])

    best_params = get_study_results(loss_function_study)
    
    del loss_function_study
    model_params.update(best_params)
    print(f"{model_params} \n --------------------------------")
 
    
    # -----------------------------------------------------
    # TUNE HYPERPARAMS
    # -----------------------------------------------------
    
    hyperparams_study = optuna.create_study(direction='maximize')
    hyperparams_study.optimize(lambda trial: 
                   hyperparams_objective(trial,
                                         train_dset,
                                         valid_dset,
                                         model_params),
                               n_trials=model.model_settings["n_iters_hyperparams"])
    
    best_params = get_study_results(hyperparams_study)
    
    del hyperparams_study
    model_params.update(best_params)
    print(f"{model_params} \n --------------------------------")
 

    # -----------------------------------------------------
    # SELECT NUMBER OF ITERATIONS
    # -----------------------------------------------------
  
    model_params['num_iterations'] = iterate_for_optimal_boosting_rounds(model_params,
                                                                         model.model_settings,
                                                                         train_dset,
                                                                         valid_dset)

    
    print("--------------------------------------------------------")
    print(f"Final Model Parameters: {model_params}")
    print("--------------------------------------------------------")
    
    
    return model_params

def model_tunning_focal_loss_1_step(model, train, validate):

        # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------
    
    
    model_params = {"boosting": "gbdt",
                    "deterministic":True,
                    "num_iterations": 10000, #Will have early stopping for tunning
                    'learning_rate':0.1,
                    #'verbosity':-1
                    }
    
    print(model_params)
    
    train_dset, valid_dset = set_lgbm_train_valid_dsets(model, train, validate)
    
    # -----------------------------------------------------
    # TUNE LOSS FUNCTION
    # -----------------------------------------------------
    
    dual_loss_hyperparams_study = optuna.create_study(direction='maximize')
    dual_loss_hyperparams_study.optimize(lambda trial: dual_loss_and_hyperparams_objective( trial,
                                                                                            train_dset,
                                                                                            valid_dset,
                                                                                            model_params),
                                                                                            n_trials=model.model_settings["n_iters_hyperparams"])

    best_params = get_study_results(loss_function_study)
    
    del loss_function_study
    model_params.update(best_params)
    print(f"{model_params} \n --------------------------------")
 


    return model_params

def model_tuning_logloss(model, train, validate):

    model_params = {'objective': 'binary',
                    'metric': model.model_settings['optuna_eval_metric'],
                    "boosting": "gbdt",
                    "deterministic":True,
                    'learning_rate':0.1,
                    #'verbosity':-1
                    }
    
    print(model_params)
    
    train_dset, valid_dset = set_lgbm_train_valid_dsets(model, train, validate)


    modelo = olgb.train(train_set=train_dset,
                       valid_sets=[valid_dset,],
                       num_boost_round = 5000,
                       early_stopping_rounds = 25,
                       #verbose_eval = False,
                       params=model_params)

    return modelo.params
