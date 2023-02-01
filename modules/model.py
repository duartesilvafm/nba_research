from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Model():


    seed: int
    X: np.array([])
    y: np.array([])
    base_model: object
    param_grid: dict


    def train_test_split(self, test_size = 0.2):
        """
        method to split the data between train and test
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state = 0)


    def train_model(self):
        """
        method to fit the model with training data
        """

        self.base_model.fit(self.X_train, self.y_train)
        self.train_score = self.base_model.score(self.X_train, self.y_train)
        print(f"Model train score: {self.train_score}")


    def test_model(self):
        """
        method to test the model
        """

        self.test_score = self.base_model.score(self.X_test, self.y_test)
        print(f"Model test score: {self.test_score}")

    
    def cross_val_score_model(self, n_splits = 4):
        """
        method to run cross validation on the model
        """
        
        kfold = KFold(n_splits=n_splits, random_state = self.seed, shuffle = True)
        self.results_cross_val = cross_val_score(self.base_model, self.X, self.y, cv=kfold)

        print(f'Model - Accuracy {self.results_cross_val.mean()*100:.3f}% std {self.results_cross_val.std()*100:3f}')


    def get_cf_matrix(self):
        """
        Generate confucius matrix for 
        """

        # predict y for the confucius matrix
        self.y_matrix = self.base_model.predict(self.X_test)
        self.cf_matrix = confusion_matrix(self.y_matrix, self.y_test)

        # code to visualize the matrix
        plt.figure(figsize = (15,8))
        ax = sns.heatmap(self.cf_matrix, annot=True, cmap='Blues', fmt = "g")

        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
        plt.show()


@dataclass
class ML_Model(Model):


    def get_best_params(self, metrics = ["accuracy"]):
        """
        Create a function to return the best results on a grid search for a specific model
        Inputs:
        model: object instantiated with to grid search on (should be an empty object with no parameters defined)
        param_grid: grid to check which parameters to go through
        scoring: list indicating the metrics to go through
        X: array with all values to predict the y
        y: array with y values
        """

        scoring_dfs = {}
        best_params = {}
        kfold = KFold(n_splits=4, random_state=self.seed, shuffle = True)

        for scoring in metrics:

            # create the grid_search objects
            grid = GridSearchCV(estimator=self.base_model, param_grid=self.param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
            grid_result = grid.fit(self.X,self.y)

            # get the best score
            best_score = grid_result.best_score_

            best_parameters = {}
            for key in self.param_grid:
                var = grid_result.best_params_[key]
                best_parameters[f"{key}"] = var
            
            # save results into different series
            means = grid_result.cv_results_["mean_test_score"]
            stds = grid_result.cv_results_["std_test_score"]      
            params = grid_result.cv_results_["params"] 

            # save the results into a dataframe
            columns_scoring_df = [key for key in self.param_grid]
            scoring_df = pd.DataFrame(columns = columns_scoring_df + ["Metric", "Score"])

            # go through the parameters and append into a dataframe
            for mean, std, param in zip(means, stds, params):
                scoring_df = pd.concat([scoring_df, pd.Series({
                    'Metric': scoring, 
                    'Score': mean,
                    **param
                })])

            scoring_dfs[scoring] = scoring_df
            best_params[scoring] = best_parameters

        self.scoring_dfs = scoring_dfs
        self.best_params = best_params

        
    def get_best_score_gs(self, metric_to_optimize = "accuracy"):
        """
        Get the best score from the grid search
        """

        # get the df for the score
        scoring_df = self.scoring_dfs[metric_to_optimize]
        params_of_scoring = self.best_params[metric_to_optimize]
            
        # save the best score
        self.best_score = scoring_df[scoring_df["Metric"] == metric_to_optimize]["Score"].max()
        print(f"Best score: {self.best_score}, Best params: {params_of_scoring}")


    def set_params(self, metric_to_optimize = "accuracy"):
        """
        method to set parameters for the model based on the grid search
        """

        params_to_set = self.best_params[metric_to_optimize]
        self.base_model.set_params(**params_to_set)