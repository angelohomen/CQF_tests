import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

class MachineLearningTools:
    def __init__():
        print('Initializing ML tools.')

    def __del__():
        print('Deinitializing ML tools.')

    def past_ret(k, dataframe):
        vet = []

        for row in range(len(dataframe.index)):
            if row >= k:
                vet.append(dataframe[row] - dataframe[row - k])
            else:
                vet.append(None)

        return vet

    def momentum(k, dataframe):
        vet = []

        for row in range(len(dataframe.index)):
            if row >= k:
                vet.append(dataframe[row] - dataframe[row - k])
            else:
                vet.append(None)

        return vet

    def sign(dataframe):
        vet = []

        for row in range(len(dataframe.index)):
            if dataframe[row] > 0:
                vet.append('1')
            else:
                vet.append('-1')

        return vet

    def vif(dataframe):
        scaler = StandardScaler()
        xs = scaler.fit_transform(dataframe)

        vif = pd.DataFrame()
        vif["Features"] = dataframe.columns
        vif["VIF Factor"] = [variance_inflation_factor(xs, i) for i in range(xs.shape[1])]
        
        return vif

    def alpha_value(X_alp, Y_alp, initial, final, by, number_var):
        vet = []
        vet_features = []

        for value in range(initial, final, by):
            value = value / 100
            alpha = value
            laso = Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=alpha))])    
            X_train, X_test, y_train, y_test = train_test_split(X_alp, Y_alp, test_size=0.2, random_state=0)
            laso.fit(X_train, y_train)
            vet_features.append(laso['regressor'].coef_)
            vet.append([alpha, laso.score(X_train, y_train), laso.score(X_test, y_test)])

        vet_df = pd.DataFrame(vet)
        vet_df.columns = ['alpha', 'R² Train', 'R² Test']
        vet_feat_df = pd.DataFrame(vet_features)
        vet_feat_df.columns = X_alp.columns
        vet = vet_df.join(vet_feat_df)

        vet_alpha = vet.drop(['R² Train', 'R² Test'], 1)
        columns_number = len(vet_alpha.columns)
        element = 0
        vet_aux = []
        alpha_value = 0

        for row in range(0, len(vet_alpha), 1):
            for col in vet_alpha.columns:
                if element >= columns_number:
                    if len(vet_aux) == number_var + 1:
                        alpha_value = vet_aux[0]
                        break
                    element = 0
                    vet_aux = []

                if vet_alpha[col][row] > 0 or vet_alpha[col][row] < 0:
                    vet_aux.append(vet_alpha[col][row])
                element += 1
        
        return vet_alpha, alpha_value

    def sigmoid(theta_0, theta_n, initial_v, final_v, by_v):
        vet=[]
        
        for z in range(initial_v, final_v, by_v):
            vet.append(1 / (1 + np.exp(- (theta_0 + theta_n * z))))

        return vet

    def plot_sigmoidal_features(z, phi_z, feature_name):
        plt.plot(z, phi_z)
        plt.axvline(0.0, color='k')
        plt.xlabel('z')
        plt.ylabel('$\phi(z)$')
        plt.yticks([0.0, 0.5, 1.0])
        ax = plt.gca()
        ax.yaxis.grid(True)
        ax.set_title(f'{feature_name} Logistic Sigmoidal plot')
        plt.tight_layout()
        plt.show()

    def plot_coeff(alpha_range, coef, modelname):
        fig = plt.figure(figsize=(20,8))
        ax = plt.axes()
        
        ax.plot(alpha_range, coef)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_title(f'{modelname} coefficients as a function of the regularization')
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('$\mathbf{w}$')