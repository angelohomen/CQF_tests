import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy import *
from statsmodels.tsa.api import adfuller

class PairsTrading():
    def __init__():
        print('Initializing pairs trading.')
    
    def __del__():
        print('Deinitializing pairs trading.')

    def data_download_and_adjust(stock1, stock2, init, final, train_perc):
        
        '''Download data and adjust dataframes by its datetimes of existance, if one of the input 
        stocks were recently released to stock exchange. It returns a dataframe with both adjusted 
        close prices of stocks, train and test dataframes already splitted.

        Parameters:
        ----------

        stock1: string.
            Stock 1 ticker.

        stock2: string.
            Stock 2 ticker.

        init: datetime.
            Beginning of data download and adjustments.

        final: datetime.
            Ending of data download and adjustments. It must be a higher datetime than 'init' 
            parameter.

        train_perc : float between 0 and 1.
            Percentage of data to be aplitted into train and test. The input percentage represents 
            how much data will be used as train data.'''

        df1 = pd.DataFrame(yf.download(tickers=stock1, start=init, end=final)['Adj Close'])
        data_stock_1 = pd.DataFrame(df1)
        data_stock_1.columns = {stock1}

        df2 = pd.DataFrame(yf.download(tickers=stock2, start=init, end=final)['Adj Close'])
        data_stock_2 = pd.DataFrame(df2)
        data_stock_2.columns = {stock2}
  
        if len(df1.columns) < len(df2.columns):
            merged_df = pd.DataFrame(df1.merge(df2, on='Date', how='inner'))
            merged_df = merged_df.dropna()
        else:
            merged_df = pd.DataFrame(df2.merge(df1, on='Date', how='inner'))
            merged_df = merged_df.dropna()

        train_df, test_df = np.split(merged_df, [int(train_perc * len(merged_df))])
        
        return merged_df, train_df, test_df

    def linear_regression(x, y):

        '''Applying a linear regression estimation. It uses 'x' to estimate 'y'. It's inputs are 
        all dataframes of stocks prices, and the function returns are: a dataframe to print a 
        table of parameters (print_df), a residuals series (residuals), a residuals dataframe
        (residuals_df) and the linear regression coefficient (b) and constant (a).

        Parameters:
        ----------

        x : dataframe or series.
            A nobs x k array where `nobs` is the number of observations and `k`
            is the number of regressors. An intercept is not included by default
            and should be added by the user. See `statsmodels.tools.add_constant`.

        y : dataframe or series.
            Response variable. The dependent variable.'''
        
        # Name of the x stock and it's dataframe
        x_name = x.name
        x_df = x
        
        # Name of the y stock and it's dataframe
        y_name = y.name
        y_df = y
        
        # This part guarantees the right way of variables to linear regression
        x = np.array(x)
        y = np.array(y)
        x = [[a] for a in x]
        y = [[b] for b in y]
        
        # Adding the constant term
        x = sm.add_constant(x)
        
        # Fit data
        lin_reg = sm.OLS(y, x).fit()
        
        # Residuals series and dataframe
        residuals = lin_reg.resid
        residuals_df = pd.DataFrame(residuals, index=x_df.index, columns=[[f'Residuals {x_name} (x) vs. {y_name} (y)']])
        
        # Estimation coefficients
        a, b = lin_reg.params
        
        # Coefficients standard deviations
        a_sd, b_sd = lin_reg.bse
        
        # Coefficients t-statistics
        a_t_test, b_t_test = lin_reg.tvalues
        
        # Print results
        print(f'Pairs trading parameters:')
        print(f'μ_e = {a:.4f}')
        print(f'β_coint = {b:.4f}')
        print()
        
        # Print dataframe
        print_df = pd.DataFrame()
        
        # Print dataframe first column - variables
        print_df.index = ['Constant', y_name]
        
        # Print dataframe second column - constant and coefficient
        print_df[f'Estimate {x_name}'] = [a, b]
        
        # Print dataframe third column - standard deviation of each constant and coefficient
        print_df[f'SD of Estimate {x_name}'] = [a_sd, b_sd]
        
        # Print dataframe fourth column - t-statistic of each constant and coefficient
        print_df[f't-statistic {x_name}'] = [a_t_test, b_t_test]
        
        return print_df, residuals, residuals_df, a, b

    def aug_dickey_fuller(series_to_adf, y_hat_name):   
        
        '''This function applies Augmented Duckey-Fuller test to a numpy series. There's an autoral 
        numerical method to estimate the ADF and the ADF given by statsmodels, which gives same 
        results.

                delta_e_t = phi.e_t-1 + phi_aug1.delta_e_t_minus_1 + const + resid

        It returns a dataframe to print a table with all parameters, a bool variable that sinalizes 
        p_value less than 0.05 and ADF Statistics to reject the null hypothesis with a significance 
        level of less than 5%.

        Parameters:
        ----------

        series_to_adf : dataframe or series.
            Residuals series after passing both dependent and independent variables into linear
            regression and estimating its residuals.

        y_hat_name : string.
            Name of estimated stock.'''
        
        # Creating a dataframe of the main series
        series_to_adf = pd.DataFrame(series_to_adf, columns=[['e_t']]).dropna()
        
        # Bool variable
        adf_stat_bool = False
        
        # delta_e_t
        delta_e_t = series_to_adf.diff().dropna()
        delta_e_t.columns = [['Δe_t']]
        
        # e_t-1
        e_t_minus_1 = series_to_adf.shift(1).dropna()
        e_t_minus_1.columns = [['e_t_minus_1']]
        
        # delta_e_t_minus_1
        delta_e_t_minus_1 = delta_e_t.shift(1).dropna()
        delta_e_t_minus_1.columns = [['Δe_t_minus_1']]
        
        # Preparing data - guarantees the right way of variables to linear regression
        dataframe_with_all = e_t_minus_1.join(delta_e_t_minus_1).dropna()
        dataframe_with_all = dataframe_with_all.join(delta_e_t).dropna()
        
        # Matrix 'x'
        x = np.array(dataframe_with_all[['e_t_minus_1', 'Δe_t_minus_1']])
        
        # Estimation 'y'
        y = np.array(dataframe_with_all['Δe_t'])
        
        # Adding constant term
        x = sm.add_constant(x)
        
        # Fit data
        result = sm.OLS(y, x).fit()
        
        # Residuals series
        residuals = result.resid
        
        # Estimation coefficients, where:
        # - a: const
        # - b: phi
        # - c: phi_aug1
        a, b, c = result.params
        
        # Coefficients standard deviations
        a_sd, b_sd, c_sd = result.bse
        
        # Coefficients t-statistics
        a_t_test, b_t_test, c_t_test = result.tvalues
        
        # Print dataframe 
        print_df = pd.DataFrame()
        
        # Print dataframe first column - variables
        print_df.index = ['Constant', f'(Lag 1, Res({y_hat_name}))', f'(Lag 1, ΔRes({y_hat_name}))']
        
        # Print dataframe second column - constant and coefficients
        print_df[f'Estimate Δ{y_hat_name}'] = [a, b, c]
        
        # Print dataframe third column - standard deviation of each coefficients and constant
        print_df[f'SD of Estimate Δ{y_hat_name}'] = [a_sd, b_sd, c_sd]
        
        # Print dataframe fourth column - t-statistic of each coefficients and constant
        print_df[f't-statistic Δ{y_hat_name}'] = [a_t_test, b_t_test, c_t_test]
        
        # Lag 1 ADF by statsmodel
        adf_result = adfuller(series_to_adf, regression='c', maxlag=1, autolag=None)
        
        # Lag 1 ADF by statsmodel results
        adf_pvalue = adf_result[1] # p_value
        perc_1, perc_5, perc_10 = list(adf_result[4].values()) # Critical values
        
        # ADF Statistics
        adf_stat = b_t_test
        
        # Printing results
        print('Augmented Dickey-Fuller Test:')
        print(f'ADF Statistics = {adf_stat:.4f}')
        print(f'ADF p_value = {adf_pvalue:.4f}')
        print(f'1% = {perc_1:.4f}')
        print(f'5% = {perc_5:.4f}')
        print(f'10% = {perc_10:.4f}')
        
        print()
        
        # True value to ADF
        if adf_pvalue <= 0.05 and adf_stat < perc_5:
            adf_stat_bool = True
        
        # Printing more details about the result:
        
        # p_value less than 0.05
        if adf_pvalue <= 0.05:
            print(f'p_value ({adf_pvalue:.4f}) is less than 0.05.')
            
            #ADF statistics less than 1%
            if adf_stat < perc_1:
                print(f'We can reject the null hypothesis with a significance level of less than 1%.')
                print(f'The process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.')

            #ADF statistics less than 5%
            if adf_stat < perc_5 and adf_stat > perc_1:
                print(f'We can reject the null hypothesis with a significance level of less than 5%.')
                print(f'The process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.')

            #ADF statistics less than 10%
            if adf_stat < perc_10 and adf_stat > perc_5:
                print(f'We can reject the null hypothesis with a significance level of less than 10%.')
                
            #ADF statistics bigger than 10%
            if adf_stat > perc_10:
                print(f'We cannot reject the null hypothesis because the adf_stat is bigger than 10% significance level.')
        
        # p_value bigger than 0.05
        else:
            print(f'p_value ({adf_pvalue:.4f}) is bigger than 0.05.')
            
            #ADF statistics less than 1%
            if adf_stat < perc_1:
                print(f'We can reject the null hypothesis with a significance level of less than 1%.')
                print(f'The process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.')

            #ADF statistics less than 5%
            if adf_stat < perc_5 and adf_stat > perc_1:
                print(f'We can reject the null hypothesis with a significance level of less than 5%.')
                print(f'The process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.')

            #ADF statistics less than 10%
            if adf_stat < perc_10 and adf_stat > perc_5:
                print(f'We can reject the null hypothesis with a significance level of less than 10%.')
                
            #ADF statistics bigger than 10%
            if adf_stat > perc_10:
                print(f'We cannot reject the null hypothesis because the adf_stat is bigger than 10% significance level.')
                
        return print_df, adf_stat_bool, adf_stat

    def second_step_engle_granger(price_stock_a, price_stock_b, e):
        
        '''Estimating the Equilibrium Correction Model, or error correction equations:

                    delta_price_a_t = phi.delta_price_b_t - (1 - alpha).e_hat_t-1

        Using as inputs the adjusted closes of stocks 'a' and 'b' and their estimation residuals.
        It returns a dataframe to print a table with all parameters and a bool variable that receives 
        'True' when all coefficients and constant have p_value less than 0.05.

        Parameters:
        ----------

        price_stock_a : dataframe or series.
            Dataframe with adjusted close prices of stock 1.

        price_stock_b : dataframe or series.
            Dataframe with adjusted close prices of stock 2.
            
        e : dataframe or series.
            Dataframe with residuals between estimation of stock 2 using stock 1 as dependent variable.'''
        
        # stocks names
        stock_1 = price_stock_a.name
        stock_2 = price_stock_b.name
        
        # p_value < 0.05 bool variable
        p_value_less = False
        
        # Residuals to dataframe
        e_t_hat = pd.DataFrame(e, columns=[['e_t_hat']]).dropna()
        e_t_hat = e_t_hat.reset_index()['e_t_hat']
        
        # delta_price_a_t
        price_stock_a = pd.DataFrame(price_stock_a).diff().dropna()
        price_stock_a.columns = [[f'Δprice_stock_{stock_1}_t']]
        price_stock_a = price_stock_a.reset_index()[f'Δprice_stock_{stock_1}_t']
        
        # delta_price_b_t
        price_stock_b = pd.DataFrame(price_stock_b).diff().dropna()
        price_stock_b.columns = [[f'Δprice_stock_{stock_2}_t']]
        price_stock_b = price_stock_b.reset_index()[f'Δprice_stock_{stock_2}_t']
        
        # Preparing data - guarantees the right way of variables to linear regression
        dataframe_with_all = price_stock_a.join(price_stock_b).dropna()
        dataframe_with_all = dataframe_with_all.join(e_t_hat).dropna()
        
        # Matrix 'x'
        x = np.array(dataframe_with_all[[f'Δprice_stock_{stock_2}_t', 'e_t_hat']])
        
        # Estimation 'y'
        y = np.array(dataframe_with_all[[f'Δprice_stock_{stock_1}_t']])
        
        # Fit data
        result = sm.OLS(y, x).fit()
        
        # Residuals series
        residuals = result.resid
        
        # Estimation coefficients, where:
        # - a: phi
        # - b: (1 - alpha)
        a, b = result.params
        
        # Coefficients standard deviations
        a_sd, b_sd = result.bse
        
        # Coefficients t-statistics
        a_t_test, b_t_test = result.tvalues
        
        # Coefficients p_values
        stock_a_p_value, stock_b_p_value = result.pvalues
        
        # Bool variable check
        if stock_a_p_value < 0.05 and stock_b_p_value < 0.05:
            p_value_less = True
        
        # Print dataframe 
        print_df = pd.DataFrame()
        
        # Print dataframe first column - variables
        print_df.index = [f'Δ{stock_2}', f'(Lag 1, Res({stock_1}))']
        
        # Print dataframe second column - coefficients
        print_df[f'Estimate Δ{stock_1}'] = [a, b]
        
        # Print dataframe third column - standard deviation of each coefficient
        print_df[f'SD of Estimate Δ{stock_1}'] = [a_sd, b_sd]
        
        # Print dataframe foyrth column - t-statistic of each coefficient
        print_df[f't-statistic Δ{stock_1}'] = [a_t_test, b_t_test]
        
        # Print dataframe fifth column - p_values of each coefficient
        print_df[f'p_value'] = [stock_a_p_value, stock_b_p_value]
        
        # Print dataframe sixth column - is p_value less than 0.05?
        print_df[f'p_value < 0.05?'] = [stock_a_p_value < 0.05, stock_b_p_value < 0.05]
        
        return print_df, p_value_less

    def third_step_engle_granger(e_t_hat, y_hat_name):    
        
        '''Applying AR(1) to residuals estimation as a initial step to estimate the Ornstein-Uhlenbeck
        process:

                        e_hat_t = C + B.e_hat_t-1 + residuals

        It returns a dataframe to print a table with all parameters, AR(1) constant (C), AR(1) 
        coefficient (B), AR(1) residuals (residuals) and AR(1) sum of squared residuals (ssr).

        Parameters:
        ----------

        e_t_hat : array.
            Array with best estimation selection residuals.

        y_hat_name : string.
            Name of estimated stock.'''

        # e_t_hat to dataframe
        e_t_hat = pd.DataFrame(e_t_hat, columns=[['e_t_hat']]).dropna()
        
        # e_t_minus_1_hat
        e_t_minus_1_hat = e_t_hat.shift(1).dropna()
        e_t_minus_1_hat.columns = [['e_t_minus_1_hat']]
        
        # Preparing data
        dataframe_with_all = e_t_hat.join(e_t_minus_1_hat).dropna()
        
        # Matrix 'x'
        x = np.array(dataframe_with_all[['e_t_minus_1_hat']])
        
        # Estimation 'y'
        y = np.array(dataframe_with_all[['e_t_hat']])
        
        # Adding constant term
        x = sm.add_constant(x)
        
        # Fit data
        residuals_AR1 = sm.OLS(y, x).fit()
        
        # Residuals
        residuals = residuals_AR1.resid
        
        # Estimation coefficients
        C, B = residuals_AR1.params
        
        # Coefficients standard deviations
        a_sd, b_sd = residuals_AR1.bse
        
        # Coefficients t-statistics
        a_t_test, b_t_test = residuals_AR1.tvalues
        
        # Print dataframe 
        print_df = pd.DataFrame()
        
        # Print dataframe first column - variables
        print_df.index = ['Constant', f'(Lag 1, Res({y_hat_name}))']
        
        # Print dataframe second column - coefficient and constant
        print_df[f'Estimate Res({y_hat_name})'] = [C, B]
        
        # Print dataframe third column - coefficient and constant standard deviations
        print_df[f'SD of Estimate Res({y_hat_name})'] = [a_sd, b_sd]
        
        # Print dataframe forth column - coefficient and constant t-statistics
        print_df[f't-statistic Res({y_hat_name})'] = [a_t_test, b_t_test]
        
        return print_df, B, C, residuals_AR1.ssr, residuals_AR1.resid

    def pairs_trading_parameters(train_dataframe, B, C, ssr, resid, e_t):
        
        '''Calculating pairs trading parameters to be used in code sequence. It calculates 
        Tau, Theta (OU process), speed of the mean-reversion (halflife), sum of squared residuals (ssr)
        and sum of squared estimate of errors (sse first principles). It returns σ_eq and σ_eq_AV.

        Parameters:
        ----------

        train_dataframe : dataframe.
            Data training dataframe with both stocks adjusted close prices.

        B : float.
            AR(1) estimating B parameter value.
            
        C : float.
            AR(1) estimating C parameter value.

        ssr : float.
            Sum of Squared Errors value.
            
        resid : array.
            Array with AR(1) residuals.

        e_t : dataframe.
            Dataframe with residuals series.'''
        
        # Daily data - Tau
        tau = 1/252
        
        # Theta from Ornstein-Uhlenbeck process
        theta = -np.log(B)/tau
        
        # Calculating halflife
        halflife = np.log(2)/theta
        
        # Halflife in days
        halflife_days = halflife/tau
        
        # Calculating μ_e to OU process
        mu_e_OU = C / (1 - B)
        
        # Calculating sum of squared residuals of regression e_t (sse)
        sse_1 = ssr
        
        # Calculating sse from first principles
        sse_first_principles = np.sum(resid ** 2)

        # Calculating annualised variance
        annualised_variance = sse_first_principles * tau

        # Calculating denominator of σ_eq
        denominator = (1 - np.exp(- 2 * theta * tau))

        # Calculating σ_eq
        sigma_eq = np.sqrt(sse_first_principles * tau / denominator)

        # Calculating σ_OU
        sigma_OU = sigma_eq * np.sqrt(2 * theta)
        
        # Calculating standard deviation of residuals
        sample_diff_e_t = resid.std()

        # Calculating σ_eq_AV
        sigma_eq_AV = np.sqrt(resid.var() / tau * denominator / (2 * theta))
        
        ##########################################################################
        # Ornstein-Uhlenbeck process construction.                               #
        ##########################################################################
        random.seed(10000)
        n = len(e_t)
        T = len(train_dataframe) / 252 # How many years
        t = 252 # Daily
        dt = T/t
        z = random.standard_normal(n)
        
        # OU series construction to plot
        de_t = (- theta * (e_t.squeeze() - mu_e_OU) * dt + sigma_OU * sqrt(dt) * z)
        de_t = pd.DataFrame(de_t)
        
        # Print results
        print(f'Annualised variance = {annualised_variance:.2f}')
        print(f'Halflife in days = {halflife_days:.0f}')
        print()
        print(f'OU process parameters:')
        print(f'μ_e = {mu_e_OU:.4f}')
        print(f'σ = {sigma_OU:.4f}')
        print()
        print('Sigmas to trade: ')
        print(f'σ_eq = {sigma_eq:.4f}')
        print(f'σ_eq_AV = {sigma_eq_AV:.4f}')
        
        f = plt.figure()
        f.set_figwidth(12)
        f.set_figheight(6)

        # OU process plotting
        plt.plot(de_t)
        plt.grid(b=False)
        plt.axhline(mu_e_OU, label='μ_e to OU process', color='orange')
        plt.axhline(mu_e_OU + 1 * sigma_OU, label='μ_e ± 1.0 * σ_OU', color='red')
        plt.axhline(mu_e_OU - 1 * sigma_OU, color='red')
        plt.legend(bbox_to_anchor=(1.02, 0.5))
        plt.title('OU process', fontsize=14);
        
        plt.tight_layout()
        
        return sigma_eq, sigma_eq_AV

    def pairs_trading_backtest(two_dataframes, mu, beta, n_sigma, sigma, to_trade, spread, stop_bool, stop_z, plot):    
    
        '''Producing a backtest to a pairs trading strategy designing. It uses as input a dataframe with 
        both stocks to trade, the linear regression coefficient and constant, the number that multiplies 
        sigma (Z), the type of sigma you want to use (σ_eq or σ_eq_AV), how much money there is to use to 
        trade, spread value, a bool to use or not stop loss, the stop loss 'Z' value and a bool variable 
        to plot results or not.
        
        Backtesting uses reinvestments, and actual operation money to trade depends on previous operation 
        results.
        
        If there are trade signals, it plots 8 graphs: a graph of result by trade number, a graph of result 
        by date, a graph of drawdown by date, a graph of the residuals, μ_e and the respective σ (entrys 
        and stop when it is used) to turn it visible where there exists trades, stock 1 prices graph and 
        stock 2 prices and estimations, percentage returns graph and sharpe ratio graph. The function returns 
        the number of trades, total result of all trades, extremes values of residuals and a dataframe with 
        all days percentage results.

        Parameters:
        ----------

        two_dataframes : dataframe.
            Dataframe with all adjusted close prices of both stocks.

        mu : float.
            Value of constant given by best estimation linear regression to both stocks.
            
        beta : float.
            Value of coefficient given by best estimation linear regression to both stocks.

        n_sigma : float.
            Number that multiplies sigma, represented by 'Z'.
            
        sigma : float.
            Value of sigma, calculated at 'pairs_trading_parameters' function.

        to_trade : float.
            Represents how much money is there to trade.
            
        spread : float.
            Bid-ask spread value, if 0 then bid and ask values are equal its last price negotiation.

        stop_bool : bool.
            If 'True' your trading system is going to use a stop loss. If 'False' it isn't going to use.
            
        stop_z : float.
            If using stop loss, then there is going to exist a 'Z' value to stop loss. It must be bigger 
            than 'n_sigma' value.

        plot : bool.
            If 'True' then backtest will show plots. If 'False' it won't show up.'''
        
        # Values to trade mu_e and beta_coint
        a = mu
        b = beta
        
        # Number that multiplies sigma
        Z = n_sigma
        
        # Names of stock 'a' and stock 'b'
        stock1, stock2 = two_dataframes.columns
        
        ##########################################################################
        # In addiction there will be bid-ask spreads.                            #
        ##########################################################################
        
        # Bid-ask spread
        bid_ask_spread = spread
        
        ######################### Initializing variables #########################
        
        # Entry price of stock 'a'
        in_price_stock1 = 0
        
        # Entry price of stock 'b'
        in_price_stock2 = 0
        
        # Out price of stock 'a'
        out_price_stock1 = 0
        
        # Out price of stock 'b'
        out_price_stock2 = 0
        
        # List positions 'a' and 'b'
        buy_pos_stock1 = []
        buy_pos_stock2 = []
        sell_pos_stock1 = []
        sell_pos_stock2 = []
        
        # Variable to help calculating balance
        trade_number = 0
        
        # Money to trade stock 'a'
        money_to_trade_a = 0
        
        # Money to trade stock 'b'
        money_to_trade_b = 0
        
        # Number of stock 'a'
        number_to_trade_a = 0
        
        # Number of stock 'b'
        number_to_trade_b = 0
        
        # Total betas: mechanism to correctly ponder how much of stock 'a' and 'b'
        total_betas = 1 + abs(b)
        
        # Bool variables to indicate position
        portfolio_buy = False # Long stock2 and Short stock1
        portfolio_sell = False # Long stock1 and Short stock2
        
        # List with all entry dates
        vet_entrys = []
        
        # List with all out dates
        vet_out = []
        
        # List of all trades results
        vet_trade = []
        
        # List of 'real-time' results
        vet_result = []
        
        # List of balances
        vet_balance = []
        
        # List of 'real-time' results without date
        vet_result_without_date = []
        
        # List of balances without date
        vet_balance_without_date = []
        
        # Adjusted closes of stock 'a'
        price_stock1 = two_dataframes[stock1]
        
        # Adjusted closes of stock 'b'
        price_stock2 = two_dataframes[stock2]
        
        # Estimation of stock 'b' using stock 'a' as independent variable: stock_b = a + stock_a * b
        price_stock2_hat = a + price_stock1 * b
        
        # Estimation to dataframe
        price_stock2_hat = pd.DataFrame(price_stock2_hat)
        price_stock2_hat.columns = [f'{stock2}_hat']
        
        # Preparing data - dataframe with real adjusted closes and estimation of stock 'b'
        prices_and_hats_df = two_dataframes.join(price_stock2_hat)
        
        # Estimation residuals - residuals = stock 'b' - stock 'b' hat
        prices_and_hats_df[f'{stock2} resid'] = prices_and_hats_df[stock2] - prices_and_hats_df[f'{stock2}_hat']
        
        # Pairs trading upper entry band: μ_e + Z * σ
        prices_and_hats_df[f'μ_e + {Z} * σ'] = a + Z * sigma
        
        # Pairs trading lower entry band: μ_e - Z * σ
        prices_and_hats_df[f'μ_e - {Z} * σ'] = a - Z * sigma
        
        # Stop control
        stop_control = False
        
        ##########################################################################
        # When stop loss is enabled a black line is plotted in residuals graph   #
        # to indicate stop loss region. Operations are allowed after a stop only #
        # when residuals go back to region of μ_e + {Z} * σ and μ_e - {Z} * σ.   #
        ##########################################################################
        if stop_bool:
            
            # Stop validation
            if Z > stop_z:
                print(f'WARNING: stop loss Z ({stop_z}) must be bigger than entry Z ({Z}).')
                print(f'Assuming input entry Z ({Z}) and stop loss Z as double {Z * 2}.')
                stop_z = Z * 2
            
            # Pairs trading upper stop band: μ_e + Z * σ
            prices_and_hats_df[f'STOP LOSS: μ_e + {stop_z} * σ'] = a + stop_z * sigma

            # Pairs trading lower stop band: μ_e - Z * σ
            prices_and_hats_df[f'STOP LOSS: μ_e - {stop_z} * σ'] = a - stop_z * sigma
            
            # Stop control
            stop_control = False
        
        ##########################################################################
        # Bid-Ask spread based on last negotiation price.                        #
        ##########################################################################
        # Stock 'a' ask column
        prices_and_hats_df[f'{stock1}_ask'] = price_stock1 + bid_ask_spread / 2
        
        # Stock 'a' bid column
        prices_and_hats_df[f'{stock1}_bid'] = price_stock1 - bid_ask_spread / 2
        
        # Stock 'b' ask column
        prices_and_hats_df[f'{stock2}_ask'] = price_stock2 + bid_ask_spread / 2
        
        # Stock 'b' bid column
        prices_and_hats_df[f'{stock2}_bid'] = price_stock2 - bid_ask_spread / 2
        
        # Signal of trade column
        prices_and_hats_df['signal'] = 0
        
        # In position column
        prices_and_hats_df['position'] = 0
        
        # Trade result column
        prices_and_hats_df['result'] = 0
        
        # Balance column
        prices_and_hats_df['balance'] = to_trade
        
        # Sharpe ratio column
        prices_and_hats_df['sharpe'] = 0.0
        
        # Volatility column
        prices_and_hats_df['volatility'] = 0
        
        # Drawdown column
        prices_and_hats_df['drawdown'] = 0
        
        # Top identifier column
        prices_and_hats_df['top'] = 0
        
        # Percentage returns
        prices_and_hats_df['percentage returns'] = 0
        
        # Turn date index to a column
        prices_and_hats_df.reset_index(inplace=True)
        
        ##########################################################################  
        # Loop to construct trade signals. If residuals's less than lower band   #
        # than we'll 'buy portfolio' (+1), if it's bigger than upper band we'll  #
        # 'sell portfolio' (-1). Else, we'll do nothing (0).                     #
        ##########################################################################  
        for idx in range(len(prices_and_hats_df.index)): 
            
            if stop_bool:
                
                # Residuals between portfolio buy Z and stop Z
                if prices_and_hats_df[f'{stock2} resid'][idx] < prices_and_hats_df[f'μ_e - {Z} * σ'][idx] and prices_and_hats_df[f'{stock2} resid'][idx] > prices_and_hats_df[f'STOP LOSS: μ_e - {stop_z} * σ'][idx]:
                    prices_and_hats_df['signal'][idx] = 1

                # Residuals between portfolio sell Z and stop Z
                if prices_and_hats_df[f'{stock2} resid'][idx] > prices_and_hats_df[f'μ_e + {Z} * σ'][idx] and prices_and_hats_df[f'{stock2} resid'][idx] < prices_and_hats_df[f'STOP LOSS: μ_e + {stop_z} * σ'][idx]:
                    prices_and_hats_df['signal'][idx] = -1

                # Residuals between entry Z range
                if prices_and_hats_df[f'{stock2} resid'][idx] < prices_and_hats_df[f'μ_e + {Z} * σ'][idx] and prices_and_hats_df[f'{stock2} resid'][idx] > prices_and_hats_df[f'μ_e - {Z} * σ'][idx]:
                    prices_and_hats_df['signal'][idx] = 0
                    
                    if stop_control:
                        
                        # Reset stop loss
                        stop_control = False
                
            else:
                
                # Residuals below portfolio buy Z
                if prices_and_hats_df[f'{stock2} resid'][idx] < prices_and_hats_df[f'μ_e - {Z} * σ'][idx]:
                    prices_and_hats_df['signal'][idx] = 1
                
                # Residuals above portfolio sell Z
                if prices_and_hats_df[f'{stock2} resid'][idx] > prices_and_hats_df[f'μ_e + {Z} * σ'][idx]:
                    prices_and_hats_df['signal'][idx] = -1

                # Residuals between entry Z range
                if prices_and_hats_df[f'{stock2} resid'][idx] < prices_and_hats_df[f'μ_e + {Z} * σ'][idx] and prices_and_hats_df[f'{stock2} resid'][idx] > prices_and_hats_df[f'μ_e - {Z} * σ'][idx]:
                    prices_and_hats_df['signal'][idx] = 0
                
                    # Reset stop loss
                    stop_control = False

        ##########################################################################  
        # Construct trade positions. If there's a +1 signal, then we're          #
        # in a 'buy portfolio' position (+1), if there's a -1 signal, we're      #
        # 'sell portfolio' position (-1). If position is +1 and now residuals is #
        # bigger or equal μ_e, we're no long in position, as well as if position #
        # is -1 and now residuals is less or equal μ_e.                          #
        ##########################################################################  
            if idx < 1:
                continue

            prices_and_hats_df['position'][idx] = prices_and_hats_df['position'][idx - 1]

            if prices_and_hats_df['signal'][idx] == 1 and not stop_control:
                prices_and_hats_df['position'][idx] = 1

            if prices_and_hats_df['signal'][idx] == -1 and not stop_control:
                prices_and_hats_df['position'][idx] = -1

            if prices_and_hats_df['position'][idx - 1] == 1 and prices_and_hats_df[f'{stock2} resid'][idx] >= a:
                prices_and_hats_df['position'][idx] = 0

            if prices_and_hats_df['position'][idx - 1] == -1 and prices_and_hats_df[f'{stock2} resid'][idx] <= a:
                prices_and_hats_df['position'][idx] = 0
                
            if stop_bool:
                # Buy portfolio stop loss
                if prices_and_hats_df['position'][idx - 1] == 1 and prices_and_hats_df[f'{stock2} resid'][idx] <= prices_and_hats_df[f'STOP LOSS: μ_e - {stop_z} * σ'][idx]:
                    prices_and_hats_df['position'][idx] = 0
                    
                    # There will be another operation only when residuals are between entry points again
                    stop_control = True
                
                # Sell portfolio stop loss
                if prices_and_hats_df['position'][idx - 1] == -1 and prices_and_hats_df[f'{stock2} resid'][idx] >= prices_and_hats_df[f'STOP LOSS: μ_e + {stop_z} * σ'][idx]:
                    prices_and_hats_df['position'][idx] = 0
                    
                    # There will be another operation only when residuals are between entry points again
                    stop_control = True
                    
                # If test do not start between stop loss bounds
                if prices_and_hats_df[f'{stock2} resid'][idx] >= prices_and_hats_df[f'STOP LOSS: μ_e + {stop_z} * σ'][idx] or prices_and_hats_df[f'{stock2} resid'][idx] <= prices_and_hats_df[f'STOP LOSS: μ_e - {stop_z} * σ'][idx]:
                
                    # There will be another operation only when residuals are between entry points again
                    stop_control = True
                        
        ##########################################################################  
        # This part of loop calculates all results and append to lists.          #
        ##########################################################################
                
            ###################################################################### 
            # If we're not in position.                                          #
            ######################################################################
            if prices_and_hats_df['position'][idx] == 0:
                
                ##################################################################
                # If there was a 'buy portfolio' position and there isn't        #
                # anymore, then out price is the previous adjusted close for     #
                # both stocks and trade result is given by:                      #
                #                                                                #
                #  result = 1 * va * (out_a - in_a) - beta * vb * (out_b - in_b) #
                #                                                                #
                # where:                                                         #
                # - result is trade result in money;                             #
                # - out_a is the out price of stock 'a';                         #
                # - in_a is the in price of stock 'a';                           #
                # - va is the position value of stock 'a';                       #
                # - out_b is the out price of stock 'b';                         #
                # - in_b is the in price of stock 'b';                           #
                # - vb is the position value of stock 'b';                       #
                # - beta is the weight given by the previous linear regression   #
                # that determines how much capital to allocate at the estimated  #
                # stock 'b'.                                                     #
                #                                                                #
                # Results then is appended to a vector and all variables reset.  #
                ##################################################################
                if prices_and_hats_df['position'][idx - 1] == 1:
                    out_price_stock1 = prices_and_hats_df[f'{stock1}_ask'][idx - 1]
                    out_price_stock2 = prices_and_hats_df[f'{stock2}_bid'][idx - 1]
                    trade_result = (1 * number_to_trade_b * (out_price_stock2 - in_price_stock2) - b * number_to_trade_a * (out_price_stock1 - in_price_stock1))
                    sell_pos_stock1.append([out_price_stock1, prices_and_hats_df['Date'][idx]])
                    buy_pos_stock2.append([out_price_stock2, prices_and_hats_df['Date'][idx]])
                    portfolio_buy = False
                    portfolio_sell = False
                    in_price_stock1 = 0
                    in_price_stock2 = 0
                    out_price_stock1 = 0
                    out_price_stock2 = 0
                    vet_trade.append(trade_result)
                    trade_result = 0
                    vet_out.append(prices_and_hats_df['Date'][idx])

                ##################################################################
                # If there was a 'sell portfolio' position and there isn't       #
                # anymore, then out price is the previous adjusted close for     #
                # both stocks and trade result is given by:                      #
                #                                                                #
                #  result = beta * vb * (out_b - in_b) - 1 * va * (out_a - in_a) #
                #                                                                #
                # where:                                                         #
                # - result is trade result in money;                             #
                # - out_a is the out price of stock 'a';                         #
                # - in_a is the in price of stock 'a';                           #
                # - va is the position value of stock 'a';                       #
                # - out_b is the out price of stock 'b';                         #
                # - in_b is the in price of stock 'b';                           #
                # - vb is the position value of stock 'b';                       #
                # - beta is the weight given by the previous linear regression   #
                # that determines how much capital to allocate at the estimated  #
                # stock 'b'.                                                     #
                #                                                                #
                # Results then is appended to a vector and all variables reset.  #
                ##################################################################
                if prices_and_hats_df['position'][idx - 1] == -1:
                    out_price_stock1 = prices_and_hats_df[f'{stock1}_bid'][idx - 1]
                    out_price_stock2 = prices_and_hats_df[f'{stock2}_ask'][idx - 1]
                    trade_result = (b * number_to_trade_a * (out_price_stock1 - in_price_stock1) - 1 * number_to_trade_b * (out_price_stock2 - in_price_stock2))
                    buy_pos_stock1.append([out_price_stock1, prices_and_hats_df['Date'][idx]])
                    sell_pos_stock2.append([out_price_stock2, prices_and_hats_df['Date'][idx]])
                    portfolio_buy = False
                    portfolio_sell = False
                    in_price_stock1 = 0
                    in_price_stock2 = 0
                    out_price_stock1 = 0
                    out_price_stock2 = 0
                    vet_trade.append(trade_result)    
                    trade_result = 0
                    vet_out.append(prices_and_hats_df['Date'][idx])
                    
            ###################################################################### 
            # No positions.                                                      #
            ###################################################################### 
            if portfolio_sell == False and portfolio_buy == False:
                
                ##################################################################
                # If we were not in position and now we 'buy portfolio'.         #
                # Now variable portfolio_buy is True and the in prices are       #    
                # adjusted closes of stock 'a' and stock 'b' right now.          #
                ##################################################################
                if prices_and_hats_df['position'][idx - 1] == 0 and prices_and_hats_df['position'][idx] == 1:
                    # Separate capital in two - stock 'a' and stock 'b'
                    money_to_trade_a = prices_and_hats_df['balance'][idx - 1] * abs(b) / total_betas
                    money_to_trade_b = prices_and_hats_df['balance'][idx - 1] * 1 / total_betas
                    number_to_trade_a = money_to_trade_a / prices_and_hats_df[f'{stock1}_bid'][idx]
                    number_to_trade_b = money_to_trade_b / prices_and_hats_df[f'{stock2}_ask'][idx]
                    in_price_stock1 = prices_and_hats_df[f'{stock1}_bid'][idx]
                    in_price_stock2 = prices_and_hats_df[f'{stock2}_ask'][idx]
                    portfolio_buy = True
                    portfolio_sell = False
                    
                    sell_pos_stock1.append([in_price_stock1, prices_and_hats_df['Date'][idx]])
                    buy_pos_stock2.append([in_price_stock2, prices_and_hats_df['Date'][idx]])
                    vet_entrys.append(prices_and_hats_df['Date'][idx])

                ##################################################################
                # If we were not in position and now we 'sell portfolio'.        #
                # Now variable portfolio_buy is True and the in prices are       #
                # adjusted closes of stock 'a' and stock 'b' right now.          #
                ##################################################################
                if prices_and_hats_df['position'][idx - 1] == 0 and prices_and_hats_df['position'][idx] == -1:
                    money_to_trade_a = prices_and_hats_df['balance'][idx - 1] * abs(b) / total_betas
                    money_to_trade_b = prices_and_hats_df['balance'][idx - 1] * 1 / total_betas
                    number_to_trade_a = money_to_trade_a / prices_and_hats_df[f'{stock1}_ask'][idx]
                    number_to_trade_b = money_to_trade_b / prices_and_hats_df[f'{stock2}_bid'][idx]
                    in_price_stock1 = prices_and_hats_df[f'{stock1}_ask'][idx]
                    in_price_stock2 = prices_and_hats_df[f'{stock2}_bid'][idx]
                    portfolio_buy = False
                    portfolio_sell = True

                    buy_pos_stock1.append([in_price_stock1, prices_and_hats_df['Date'][idx]])
                    sell_pos_stock2.append([in_price_stock2, prices_and_hats_df['Date'][idx]])
                    vet_entrys.append(prices_and_hats_df['Date'][idx])

            ###################################################################### 
            # If we bought portfolio.                                            #
            ######################################################################
            if portfolio_buy:

                ##################################################################
                # Every day result is updated, calculated by:                    #
                #                                                                #
                #  result = 1 * va * (act_a - in_a) - beta * vb * (act_b - in_b) #
                #                                                                #
                # where:                                                         #
                # - result is trade result in money;                             #
                # - act_a is actual price of stock 'a';                          #
                # - in_a is the in price of stock 'a';                           #
                # - va is the position value of stock 'a';                       #
                # - act_b is actual price of stock 'b';                          #
                # - in_b is the in price of stock 'b';                           #
                # - vb is the position value of stock 'b';                       #
                # - beta is the weight given by the previous linear regression   #
                # that determines how much capital to allocate at the estimated  #
                # stock 'b'.                                                     #
                ##################################################################
                prices_and_hats_df['result'][idx] = (1 * number_to_trade_b * (prices_and_hats_df[stock2][idx] - in_price_stock2) - b * number_to_trade_a * (prices_and_hats_df[stock1][idx] - in_price_stock1))

                sell_pos_stock1.append([prices_and_hats_df[stock1][idx], prices_and_hats_df['Date'][idx]])
                buy_pos_stock2.append([prices_and_hats_df[stock2][idx], prices_and_hats_df['Date'][idx]])
                
                ##################################################################
                # To avoid repetition, make sure that actual result is different #
                # from previous and append it to lists mentioned above.          #
                ##################################################################
                if prices_and_hats_df['result'][idx] != prices_and_hats_df['result'][idx - 1]:
                    vet_result.append([prices_and_hats_df['result'][idx], prices_and_hats_df['Date'][idx]])
                    vet_result_without_date.append(prices_and_hats_df['result'][idx])
                    
                ##################################################################
                # If test is finishing and we're on position, finish the trade.  #
                ##################################################################
                if idx >= len(prices_and_hats_df.index) - 1:
                    out_price_stock1 = prices_and_hats_df[f'{stock1}_ask'][idx - 1]
                    out_price_stock2 = prices_and_hats_df[f'{stock2}_bid'][idx - 1]
                    trade_result = (1 * number_to_trade_b * (out_price_stock2 - in_price_stock2) - b * number_to_trade_a * (out_price_stock1 - in_price_stock1))
                    vet_trade.append(trade_result) 
                    vet_out.append(prices_and_hats_df['Date'][idx])
                    sell_pos_stock1.append([out_price_stock1, prices_and_hats_df['Date'][idx]])
                    buy_pos_stock2.append([out_price_stock2, prices_and_hats_df['Date'][idx]])

            ###################################################################### 
            # If we sold portfolio.                                              #
            ######################################################################
            if portfolio_sell:
                
                ##################################################################
                # Every day result is updated, calculated by:                    #
                #                                                                #
                #  result = beta * vb * (act_b - in_b) - 1 * va * (act_a - in_a) #
                #                                                                #
                # where:                                                         #
                # - result is trade result in money;                             #
                # - act_a is actual price of stock 'a';                          #
                # - in_a is the in price of stock 'a';                           #
                # - va is the position value of stock 'a';                       #
                # - act_b is actual price of stock 'b';                          #
                # - in_b is the in price of stock 'b';                           #
                # - vb is the position value of stock 'b';                       #
                # - beta is the weight given by the previous linear regression   #
                # that determines how much capital to allocate at the estimated  #
                # stock 'b'.                                                     #
                ##################################################################
                prices_and_hats_df['result'][idx] = (b * number_to_trade_a * (prices_and_hats_df[stock1][idx] - in_price_stock1) - 1 * number_to_trade_b * (prices_and_hats_df[stock2][idx] - in_price_stock2))

                buy_pos_stock1.append([prices_and_hats_df[stock1][idx], prices_and_hats_df['Date'][idx]])
                sell_pos_stock2.append([prices_and_hats_df[stock2][idx], prices_and_hats_df['Date'][idx]])
                
                ##################################################################
                # To avoid repetition, make sure that actual result is different #
                # from previous and append it to lists mentioned above.          #
                ##################################################################
                if prices_and_hats_df['result'][idx] != prices_and_hats_df['result'][idx - 1]:
                    vet_result.append([prices_and_hats_df['result'][idx], prices_and_hats_df['Date'][idx]])
                    vet_result_without_date.append(prices_and_hats_df['result'][idx])
                    
                ##################################################################
                # If test is finishing and we're on position, finish the trade.  #
                ##################################################################
                if idx >= len(prices_and_hats_df.index) - 1:
                    out_price_stock1 = prices_and_hats_df[f'{stock1}_bid'][idx - 1]
                    out_price_stock2 = prices_and_hats_df[f'{stock2}_ask'][idx - 1]
                    trade_result = (b * number_to_trade_a * (out_price_stock1 - in_price_stock1) - 1 * number_to_trade_b * (out_price_stock2 - in_price_stock2))
                    vet_trade.append(trade_result) 
                    vet_out.append(prices_and_hats_df['Date'][idx])
                    buy_pos_stock1.append([out_price_stock1, prices_and_hats_df['Date'][idx]])
                    sell_pos_stock2.append([out_price_stock2, prices_and_hats_df['Date'][idx]])
                    
        ##########################################################################  
        # This part of loop calculates balance.                                  #
        ##########################################################################
            prices_and_hats_df['balance'][idx] = prices_and_hats_df['balance'][idx - 1]

            ###################################################################### 
            # In position.                                                       #
            ######################################################################
            if prices_and_hats_df['position'][idx] != 0:

                if trade_number < 1:                
                    prices_and_hats_df['balance'][idx] = to_trade + prices_and_hats_df['result'][idx]
                    vet_balance.append([prices_and_hats_df['balance'][idx], prices_and_hats_df['Date'][idx]])
                    
                    if prices_and_hats_df['balance'][idx] != prices_and_hats_df['balance'][idx - 1]:
                        vet_balance_without_date.append(prices_and_hats_df['balance'][idx])
                else:
                    prices_and_hats_df['balance'][idx] = prices_and_hats_df['balance'][idx - 1] + (prices_and_hats_df['result'][idx] - prices_and_hats_df['result'][idx - 1])
                    vet_balance.append([prices_and_hats_df['balance'][idx], prices_and_hats_df['Date'][idx]])
                    
                    if prices_and_hats_df['balance'][idx] != prices_and_hats_df['balance'][idx - 1]:
                        vet_balance_without_date.append(prices_and_hats_df['balance'][idx])
                        
            ###################################################################### 
            # No position.                                                       #
            ######################################################################
            else:            
                if prices_and_hats_df['position'][idx - 1] != 0:
                    trade_number = trade_number + 1
                    
        ##########################################################################  
        # This part of loop calculates drawdowns.                                #
        ##########################################################################
                
            ###################################################################### 
            # Top monitoring.                                                    #
            ######################################################################
            top = prices_and_hats_df['balance'][idx]
            
            if top >= prices_and_hats_df['top'][idx - 1]:
                top = top
            else:
                top = prices_and_hats_df['top'][idx - 1]
                
            prices_and_hats_df['top'][idx] = top

            ###################################################################### 
            # Drawdown monitoring.                                               #
            ######################################################################
            drawdown = prices_and_hats_df['top'][idx] - prices_and_hats_df['balance'][idx]

            if drawdown >= prices_and_hats_df['drawdown'][idx - 1]:
                drawdown = drawdown
            else:
                drawdown = prices_and_hats_df['drawdown'][idx - 1]
                
            prices_and_hats_df['drawdown'][idx] = - drawdown  
        
        # Percentage returns dataframe
        prices_and_hats_df['percentage returns'] = (prices_and_hats_df['balance'] - to_trade) / to_trade
        
        # Control counter
        control_counter = 0
        
        # Control list
        percentage_ret_list = []
        
        ##########################################################################  
        # Loop to calculate rolling sharpe ratio.                                #
        ##########################################################################
        for idx in range(len(prices_and_hats_df.index)):
            
            sharpe_ratio_value = 0
            
            if idx < 1:
                continue
                
            # First position entry - this part appends percentage returns
            if control_counter < 1:
                if prices_and_hats_df['position'][idx - 1] == 0 and prices_and_hats_df['position'][idx] != 0:
                    control_counter = idx
            else:
                percentage_ret_list.append(prices_and_hats_df['percentage returns'][idx])
                
                # Stabilize Sharpe Ratio calculating it after 30 returns
                if np.std(percentage_ret_list) != 0.0 and idx > control_counter + 30:
                    
                    # Calculating mean
                    returns_mean = np.mean(percentage_ret_list)
                    
                    # Calculating standard deviation
                    return_std_dev = np.std(percentage_ret_list)
                    
                    # SR = mean / standard_deviation
                    sharpe_ratio_value = returns_mean / return_std_dev
                    
            prices_and_hats_df['sharpe'][idx] = sharpe_ratio_value
            
        # Balances to dataframe  
        result_df = pd.DataFrame(vet_balance, columns=['Balance', 'Date'])
        result_df_x = result_df['Date']
        result_df_y = result_df['Balance']
        
        # Positions stocks
        buy_pos_stock1 = pd.DataFrame(buy_pos_stock1)
        buy_pos_stock2 = pd.DataFrame(buy_pos_stock2)
        sell_pos_stock1 = pd.DataFrame(sell_pos_stock1)
        sell_pos_stock2 = pd.DataFrame(sell_pos_stock2)
        
        # P&L construction
        vet_trade_df = pd.DataFrame(vet_trade)
        positive_trades_df = pd.DataFrame()
        negative_trades_df = pd.DataFrame()
        
        # Printing bool control
        bool_control_trades = False
        
        # If there was buy positions to stock 'a'
        if not buy_pos_stock1.empty:
            buy_pos_stock1.columns = [f'price_buy_pos_{stock1}', 'Date']
        
        # If there was buy positions to stock 'b'
        if not buy_pos_stock2.empty:
            buy_pos_stock2.columns = [f'price_buy_pos_{stock2}', 'Date']
            
        # If there was sell positions to stock 'a'
        if not sell_pos_stock1.empty:
            sell_pos_stock1.columns = [f'price_sell_pos_{stock1}', 'Date']
            
        # If there was sell positions to stock 'b'
        if not sell_pos_stock2.empty:
            sell_pos_stock2.columns = [f'price_sell_pos_{stock2}', 'Date']
        
        # If there was trades at the period
        if not vet_trade_df.empty:
            result_no_date_df = pd.DataFrame(vet_balance_without_date)
            vet_trade_df.columns = ['Return by trade']
            
            # All positive trades
            positive_trades_df = vet_trade_df[vet_trade_df > 0].dropna()
            positive_trades_df = np.array(positive_trades_df)
                
            # All negative trades
            negative_trades_df = vet_trade_df[vet_trade_df < 0].dropna()
            negative_trades_df = np.array(negative_trades_df)
            
            # Sum of positive trades
            positive_trades_df = np.sum(positive_trades_df)
            
            # Sum of negative trades
            negative_trades_df = np.sum(negative_trades_df)
            
            bool_control_trades = True
        
        # Percentage dataframe
        date_df = pd.DataFrame(prices_and_hats_df['Date'])
        perc_df = pd.DataFrame(prices_and_hats_df['balance'].pct_change())
        percentage_df = date_df.join(perc_df)
        percentage_df.columns = ['Date', 'percentage returns']
        percentage_df = percentage_df.set_index('Date')
        
        # If user set True to plotting
        if plot:
            # Print results
            print(f'Number of trades: {len(vet_trade_df)} | Backtest result: {sum(vet_trade):.2f}')
            
            print()
            
            # If there was complete trades
            if bool_control_trades:
                print(f'Profits: {positive_trades_df:.2f} | Losses: {negative_trades_df:.2f}')
            
            # If there was complete trades
            if not result_df.empty:
                
                f = plt.figure()
                f.set_figwidth(13)
                f.set_figheight(10)
                
                # Result by trade graph
                plt.subplot(4, 2, 1)
                plt.plot(result_no_date_df, color='green', linestyle='-')
                plt.grid(b=False)
                plt.title('Balance curve by trade', fontsize=14);

                # Result by date graph
                plt.subplot(4, 2, 2)
                plt.plot(result_df_x, result_df_y, color='purple', linestyle='-')
                plt.grid(b=False)
                plt.title('Balance curve by date', fontsize=14);
                
                ##################################################################
                # Plot green lines to all entry dates.                           #
                ##################################################################
                for i in vet_entrys:
                    plt.axvline(x=i, color='g', linestyle='-', label='entrys')
                    
                ##################################################################
                # Plot red lines to all entry dates.                             #
                ##################################################################
                for i in vet_out:
                    plt.axvline(x=i, color='r', linestyle='-', label='out')

                # Drawdown by date graph
                plt.subplot(4, 2, 3)
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df['drawdown'], color='red', linestyle='-')
                plt.grid(b=False)
                plt.title('Drawdown curve by date', fontsize=14);

                # Residuals graph
                plt.subplot(4, 2, 4)
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df[f'{stock2} resid'], color='b', linestyle='-')
                plt.grid(b=False)
                plt.axhline(y=a, color='y', linestyle='-', label='μ_e')
                
                # Plot stop loss if it was at backtesting, else just plot entry points σ
                if stop_bool:
                    plt.title('Residuals with σ (entry and out points) and μ_e (out points)', fontsize=14)
                    plt.axhline(y=a + stop_z * sigma, color='black', linestyle='-', label=f'Stop loss: μ_e ± {stop_z} * σ')
                    plt.axhline(y=a - stop_z * sigma, color='black', linestyle='-')
                else:
                    plt.title('Residuals with σ (entry points) and μ_e (out points)', fontsize=14)
                
                plt.axhline(y=a + Z * sigma, color='r', linestyle='-', label=f'μ_e ± {Z} * σ')
                plt.axhline(y=a - Z * sigma, color='r', linestyle='-');
                
                ##################################################################
                # Plot green lines to all entry dates.                           #
                ##################################################################
                for i in vet_entrys:
                    plt.axvline(x=i, color='g', linestyle='-', label='entrys')
                    
                ##################################################################
                # Plot red lines to all out dates.                               #
                ##################################################################
                for i in vet_out:
                    plt.axvline(x=i, color='r', linestyle='-', label='out')
                    
                # Stock 'a' graph
                plt.subplot(4, 2, 5)
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df[stock1], color='orange')
                
                # If there was buy positions to stock 'a'
                if not buy_pos_stock1.empty:
                    plt.scatter(buy_pos_stock1['Date'], buy_pos_stock1[f'price_buy_pos_{stock1}'], color='green')
                    
                # If there was sell positions to stock 'a'
                if not sell_pos_stock1.empty:
                    plt.scatter(sell_pos_stock1['Date'], sell_pos_stock1[f'price_sell_pos_{stock1}'], color='red')
                    
                plt.grid(b=False)
                plt.title(f'{stock1} graph', fontsize=14);
                
                # Stock 'b' graph
                plt.subplot(4, 2, 6)
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df[stock2], color='orange')
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df[f'{stock2}_hat'], '-.', color='b')
                
                # If there was buy positions to stock 'b'
                if not buy_pos_stock2.empty:
                    plt.scatter(buy_pos_stock2['Date'], buy_pos_stock2[f'price_buy_pos_{stock2}'], color='green')
                
                # If there was sell positions to stock 'b'
                if not sell_pos_stock2.empty:
                    plt.scatter(sell_pos_stock2['Date'], sell_pos_stock2[f'price_sell_pos_{stock2}'], color='red')
                    
                plt.grid(b=False)
                plt.title(f'{stock2} graph (orange) and {stock2}_hat graph (blue)', fontsize=14);
                    
                # Percentage returns graph
                plt.subplot(4, 2, 7)
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df['percentage returns'], color='g', linestyle='-')
                plt.grid(b=False)
                plt.title('Percentage returns by date', fontsize=14);
                
                # Sharpe ratio graph
                plt.subplot(4, 2, 8)
                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df['sharpe'], color='orange', linestyle='-')
                plt.grid(b=False)
                plt.title('Sharpe ratio by date', fontsize=14);
                
                plt.tight_layout()

            # If there weren't any trades - plot only residuals graph
            elif result_df.empty:
                
                f = plt.figure()
                f.set_figwidth(13)
                f.set_figheight(8)

                plt.plot(prices_and_hats_df['Date'], prices_and_hats_df[f'{stock2} resid'], color='b', linestyle='-')
                plt.grid(b=False)
                plt.title('Residuals with σ (entry points) and μ_e (out points)', fontsize=14)
                plt.axhline(y=a, color='y', linestyle='-', label='μ_e')
                plt.axhline(y=a + Z * sigma, color='r', linestyle='-', label=f'μ_e ± {n_sigma} * σ')
                plt.axhline(y=a - Z * sigma, color='r', linestyle='-');
                
                # Plot stop loss if it was at backtesting, else just plot entry points σ
                if stop_bool:
                    plt.title('Residuals with σ (entry and out points) and μ_e (out points)', fontsize=14)
                    plt.axhline(y=a + stop_z * sigma, color='black', linestyle='-', label=f'Stop loss: μ_e ± {stop_z} * σ')
                    plt.axhline(y=a - stop_z * sigma, color='black', linestyle='-')
                else:
                    plt.title('Residuals with σ (entry points) and μ_e (out points)', fontsize=14)
                
                plt.tight_layout()
        
        return len(vet_trade_df), sum(vet_trade), prices_and_hats_df[f'{stock2} resid'].min(), prices_and_hats_df[f'{stock2} resid'].max(), percentage_df

    def optimize_pairs_z(two_dataframes, const, coeff, sigma, by_value, to_trade):

        '''This function optimizes Z using by values given by inputs. Other inputs are: dataframe with both 
        prices, sigma value and money to trade. The function calculates automatically the maximum value of 
        Z to test all possible values without any unnecessary value.

        Parameters:
        ----------

        two_dataframes : dataframe.
            Dataframe with all adjusted close prices of both stocks.

        const : float.
            Value of constant given by best estimation linear regression to both stocks.
            
        coeff : float.
            Value of coefficient given by best estimation linear regression to both stocks.

        sigma : float.
            Value of sigma, calculated at 'pairs_trading_parameters' function.
            
        by_value : float.
            Value that will be used in iteration.

        to_trade : float.
            Represents how much money is there to trade.'''
        
        # Lists
        df = []
        best_return = []
        
        # Initializing variables
        bigger_return = 0
        best_z = 0

        # Minimum and maximum residuals
        none_var1, none_var2, min_resid, max_resid, none_var3 = PairsTrading.pairs_trading_backtest(two_dataframes, const, coeff, 
                                                                            0, sigma, to_trade, 0, False, 0, False)
        
        # Integer to loop
        min_value = int(min_resid * 1000 / sigma)
        max_value = int(max_resid * 1000 / sigma)
        
        # When to finish optimization
        until_value = 0
        
        if abs(max_value) > abs(min_value):
            until_value = abs(max_value)
        else:
            until_value = abs(min_value)
        
        # by_value validation
        if by_value < until_value:
            by_value = int(by_value * 1000) # Between 0.01 and 'until_value'
        else:
            print(f'by_value must be between {0} and {until_value}')
        
        ##########################################################################  
        # Loop all Z values.                                                     #
        ##########################################################################  
        for i in range(0, until_value, by_value):
            # Re-setting loop
            i = i / 1000
            
            # Backtesting all Z values
            trades_number, test_return, none_var1, none_var2, none_var3 = PairsTrading.pairs_trading_backtest(two_dataframes, const, coeff, 
                                                                i, sigma, to_trade, 0, False, 0, False)
            
            # List with all results
            df.append([trades_number, test_return, i])
        
        # List do dataframe with columns: number of trades, return of trades and Z value
        df = pd.DataFrame(df)
        df.columns = ['trades_number', 'test_return', 'Z']
        
        # Best return column
        df['best_return'] = df['test_return']
        df['best_z'] = df['Z']
        
        ##########################################################################  
        # Loop dataframe to estimate the best result and respective Z value.     #
        ##########################################################################  
        for idx in range(len(df.index)): 
            
            if idx < 1 or idx >= len(df.index):
                continue
                
            bigger_return = df['test_return'][idx]
            best_z = df['Z'][idx]
            
            if bigger_return > df['best_return'][idx - 1]:
                bigger_return = bigger_return
                best_z = best_z
            else:
                bigger_return = df['best_return'][idx - 1]
                best_z = df['best_z'][idx - 1]
                
            df['best_return'][idx] = bigger_return
            df['best_z'][idx] = best_z
                
        # Append the best return and respective Z value
        best_return.append([bigger_return, best_z])
        
        # List to dataframe of best result
        best_return = pd.DataFrame(best_return)
        
        # Best return value
        return_1 = best_return.iloc[0, 0]
        
        # Best Z value
        z_1 = best_return.iloc[0, 1]
        
        # Print results
        print(f'Best return is {return_1:.2f} with Z of {z_1}')
        
        f = plt.figure()
        f.set_figwidth(12)
        f.set_figheight(5)
        
        plt.subplot(1,2,1)
        plt.plot(df['Z'], df['test_return'], color='r', linestyle='-')
        plt.grid(b=False)
        plt.title('Tests returns for each Z', fontsize=14);
        
        plt.subplot(1,2,2)
        plt.plot(df['Z'], df['trades_number'], color='r', linestyle='-')
        plt.grid(b=False)
        plt.title('Tests number of trades for each Z', fontsize=14);
        
        plt.tight_layout()
        
        return df, return_1, z_1

    def optimize_pairs_stop_loss_z(two_dataframes, const, coeff, sigma, Z, by_value, to_trade):    

        '''This function optimizes Z stop loss using by values given by inputs. Other inputs are: dataframe 
        with both prices, sigma value and money to trade. The function calculates automatically the maximum 
        value of Z to test all possible values without any unnecessary value.

        Parameters:
        ----------

        two_dataframes : dataframe.
            Dataframe with all adjusted close prices of both stocks.

        const : float.
            Value of constant given by best estimation linear regression to both stocks.
            
        coeff : float.
            Value of coefficient given by best estimation linear regression to both stocks.

        sigma : float.
            Value of sigma, calculated at 'pairs_trading_parameters' function.
            
        Z : float.
            Value of best 'Z' given by 'optimize_pairs_z' function.
            
        by_value : float.
            Value that will be used in iteration.

        to_trade : float.
            Represents how much money is there to trade.'''
        
        # Lists
        df = []
        best_return = []
        
        # Initializing variables
        bigger_return = 0
        best_z = 0

        # Minimum and maximum residuals
        none_var1, none_var2, min_resid, max_resid, none_var3 = PairsTrading.pairs_trading_backtest(two_dataframes, const, coeff, 
                                                                            Z, sigma, to_trade, 0, False, 0, False)
        
        # Integer to loop
        min_value = int(min_resid * 1000 / sigma)
        max_value = int(max_resid * 1000 / sigma)
        
        # When to finish optimization
        until_value = 0
        
        if abs(max_value) > abs(min_value):
            until_value = abs(max_value)
        else:
            until_value = abs(min_value)
        
        # by_value validation
        if by_value < until_value / 1000:
            by_value = int(by_value * 1000) # Between 0.01 and 'until_value'
        else:
            by_value = int(0.01 * 1000)
            print(f'by_value must be between {Z} and {until_value}, assuming {0.01}.')
            
        # From value to begin optimization
        from_value = int((Z * 1000 + (until_value - (Z * 1000)) / 2))   # Optimize from half distance between 
                                                                    # Z to entry and extreme values of
                                                                    # residuals.
        
        ##########################################################################  
        # Loop all Z values.                                                     #
        ##########################################################################  
        for i in range(from_value, until_value, by_value):
            # Re-setting loop
            i = i / 1000
            
            # Backtesting all Z values
            trades_number, test_return, none_var1, none_var2, none_var3 = PairsTrading.pairs_trading_backtest(two_dataframes, const, coeff, 
                                                                Z, sigma, to_trade, 0, True, i, False)
            
            # List with all results
            df.append([trades_number, test_return, i])
        
        # List do dataframe with columns: number of trades, return of trades and Z value
        df = pd.DataFrame(df)
        df.columns = ['trades_number', 'test_return', 'Z_stop_loss']
        
        # Best return column
        df['best_return'] = df['test_return']
        df['best_z_stop_loss'] = df['Z_stop_loss']
        
        ##########################################################################  
        # Loop dataframe to estimate the best result and respective Z value.     #
        ##########################################################################  
        for idx in range(len(df.index)): 
            
            if idx < 1 or idx >= len(df.index):
                continue
                
            bigger_return = df['test_return'][idx]
            best_z = df['Z_stop_loss'][idx]
            
            if bigger_return > df['best_return'][idx - 1]:
                bigger_return = bigger_return
                best_z = best_z
            else:
                bigger_return = df['best_return'][idx - 1]
                best_z = df['best_z_stop_loss'][idx - 1]
                
            df['best_return'][idx] = bigger_return
            df['best_z_stop_loss'][idx] = best_z
                
        # Append the best return and respective Z value
        best_return.append([bigger_return, best_z])
        
        # List to dataframe of best result
        best_return = pd.DataFrame(best_return)
        
        # Best return value
        return_1 = best_return.iloc[0, 0]
        
        # Best Z value
        z_1 = best_return.iloc[0, 1]
        
        # Print results
        print(f'Best return is {return_1:.2f} with Z to stop loss of {z_1}')
        
        f = plt.figure()
        f.set_figwidth(12)
        f.set_figheight(5)
        
        plt.subplot(1,2,1)
        plt.plot(df['Z_stop_loss'], df['test_return'], color='r', linestyle='-')
        plt.grid(b=False)
        plt.title('Tests returns for each Z to stop loss', fontsize=14);
        
        plt.subplot(1,2,2)
        plt.plot(df['Z_stop_loss'], df['trades_number'], color='r', linestyle='-')
        plt.grid(b=False)
        plt.title('Tests number of trades for each Z to stop loss', fontsize=14);
        
        plt.tight_layout()
        
        return df, return_1, z_1

    def bid_ask_influence(init, final, by, two_dataframes, const, coeff, Z, sigma, to_trade):

        '''Function to elucidate bid-ask spread influence with a simple type of calculation.

        Parameters:
        ----------
        
        init : float.
            Value that will begin iteration.
        
        final : float.
            Value that will end iteration.
        
        by : float.
            Value that will be used in iteration.

        two_dataframes : dataframe.
            Dataframe with all adjusted close prices of both stocks.

        const : float.
            Value of constant given by best estimation linear regression to both stocks.
            
        coeff : float.
            Value of coefficient given by best estimation linear regression to both stocks.
            
        Z : float.
            Value of best 'Z' given by 'optimize_pairs_z' function.
            
        sigma : float.
            Value of sigma, calculated at 'pairs_trading_parameters' function.

        to_trade : float.
            Represents how much money is there to trade.'''
        
        # Integers to loop
        init = int(init * 100)
        final = int(final * 100)
        by = int(by * 100)
        
        # List of values
        vet_spread_result = []
        
        ##########################################################################  
        # Loop dataframe to test all bid-ask spreads.                            #
        ########################################################################## 
        for k in range(init, final, by):
            k = k / 100
            b_a_numb_trades, b_a_result, none_var1, none_var2, none_var3 = PairsTrading.pairs_trading_backtest(two_dataframes, const, coeff, 
                                                                                    Z, sigma, to_trade, k, False, 0, False)
            vet_spread_result.append([k, b_a_numb_trades, b_a_result])

        # List to dataframe
        vet_spread_result = pd.DataFrame(vet_spread_result)
        vet_spread_result.columns = ['Bid-Ask spread', 'Number of trades', 'Result']

        # Plot
        f = plt.figure()
        f.set_figwidth(7)
        f.set_figheight(4)
        plt.plot(vet_spread_result['Bid-Ask spread'], vet_spread_result['Result'], color='r', linestyle='-')
        plt.grid(b=False)
        plt.title('Bid-Ask influence to result', fontsize=14);
        
        return vet_spread_result