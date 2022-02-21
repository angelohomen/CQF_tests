import pandas as pd
import numpy as np
from numpy import *
from tabulate import tabulate
import matplotlib.pyplot as plt

class ExoticOptions:
    def __init__(self, ):
        print('Exotic options pricing.')
    
    def __del__(self, ):
        print('Exotic options succeeded.')

    def payoff_mean(int_rate, time_exp, first_df, sec_df):
        value = (exp(- int_rate * time_exp) * mean(maximum(first_df - sec_df, 0))).mean()
        return value

    def payoff(int_rate, time_exp, first_df, sec_df):
        value = exp(- int_rate * time_exp) * mean(maximum(first_df - sec_df, 0))
        return value

    def monte_carlo_simulation(initial_spot, average, sigma, horizon, timesteps, number_sim):
        random.seed(10000) 
        s = initial_spot
        r = average
        T = horizon
        t = timesteps
        n = number_sim
        dt = T/t
        S = zeros((t + 1, n))
        S[0] = s

        for i in range(1, t + 1):
            z = random.standard_normal(n)
            S[i] = S[i - 1] * exp((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z)  

        return S

    def continuous_arithmetic_avrg(dictionary, dataframe):
        for col in dataframe.columns:
            vet = []
            for index in range(len(dataframe[col])):
                temp = index
                summ = 0
                while temp >= 0:
                    summ = summ + dataframe[col][temp]
                    temp = temp - 1
                vet.append(summ/(index + 1))
            dictionary[col] = vet
        
        return dictionary

    def continuous_geometric_avrg(dictionary, dataframe):
        for col in dataframe.columns:
            vet = []
            for index in range(len(dataframe[col])):
                temp = index
                mult = 1
                while temp >= 0:
                    mult = mult * (dataframe[col][temp] ** (1 / (index + 1)))
                    temp = temp - 1
                vet.append(mult)
            dictionary[col] = vet
            
        return dictionary

    def discrete_arithmetic_avrg(dictionary, dataframe, periods):
        for col in dataframe.columns:
            vet = []
            for index in range(len(dataframe[col])):
                temp = index
                summ = 0
                if index % periods == 0:
                    control = index
                    avg = 0
                    for new_temp in range(index, (index + periods), 1):
                        if new_temp < len(dataframe[col]):
                            summ = summ + dataframe[col][new_temp]
                            avg = summ/(periods)
                        else:
                            avg = summ/(len(dataframe[col]) - control)
                vet.append(avg)
            dictionary[col] = vet
            
        return dictionary

    def discrete_geometric_avrg(dictionary, dataframe, periods):
        for col in dataframe.columns:
            vet = []
            for index in range(len(dataframe[col])):
                temp = index
                mult = 1
                if index % periods == 0:
                    control = index
                    for new_temp in range(index, (index + periods), 1):
                        avg = 0
                        if new_temp < len(dataframe[col]):
                            mult = mult * (dataframe[col][new_temp] ** (1 / (periods)))
                            avg = mult
                        else:
                            mult = 1
                            initial_range = len(dataframe[col])
                            final_range = control
                            for last_temps in range(initial_range, final_range, - 1):
                                if last_temps <= len(dataframe[col]):
                                    mult = mult * ((dataframe[col][last_temps - 1]) ** (1 / (initial_range - final_range)))
                                    avg = mult
                vet.append(avg)
            dictionary[col] = vet
        
        return dictionary

    def continuous_maximum(dictionary, dataframe):
        for col in dataframe.columns:
            vet = []
            for index in range(len(dataframe[col])):
                if index == 0:
                    higher = dataframe[col][index]
                if higher < dataframe[col][index]:
                    higher = dataframe[col][index] 
                else:
                    higher = higher
                vet.append(higher)
            dictionary[col] = vet
        
        return dictionary

    def discrete_maximum(dictionary, dataframe, periods):
        for col in dataframe.columns:
            vet = []
            higher = dataframe[col][0]
            for index in range(len(dataframe[col])):
                if index % periods == 0:
                    if higher < dataframe[col][index]:
                        higher = dataframe[col][index] 
                    else:
                        higher = higher
                vet.append(higher)
            dictionary[col] = vet
        
        return dictionary

    def varying_parameters(So, r, vol, T, t, sims, period, E, parameter, initial_range, final_range, by):
    
        AC = {}
        AD = {}
        GC = {}
        GD = {}
        MC = {}
        MD = {}
        CAARC = []
        CGARC = []
        DAARC = []
        DGARC = []
        CAARP = []
        CGARP = []
        DAARP = []
        DGARP = []
        CAASC = []
        CGASC = []
        DAASC = []
        DGASC = []
        CAASP = []
        CGASP = []
        DAASP = []
        DGASP = []
        CMRC = []
        DMRC = []
        CMSC = []
        DMSC = []
        CMRP = []
        DMRP = []
        CMSP = []
        DMSP = []
        
        if parameter == 'E':
            
            S = ExoticOptions.monte_carlo_simulation(So, r, vol, T, t, sims)
            path = pd.DataFrame(S)

            for iterator in range(int(initial_range * 100), int(final_range * 100 + 1), int(by * 100)):
                
                iterator = iterator / 100

                AC = ExoticOptions.continuous_arithmetic_avrg(AC, path)
                art_cont_avg = pd.DataFrame(AC)

                GC = ExoticOptions.continuous_geometric_avrg(GC, path)
                geo_cont_avg = pd.DataFrame(GC)

                AD = ExoticOptions.discrete_arithmetic_avrg(AD, path, period)
                art_disc_avg = pd.DataFrame(AD)

                GD = ExoticOptions.discrete_geometric_avrg(GD, path, period)
                geo_disc_avg = pd.DataFrame(GD)

                MC = ExoticOptions.continuous_maximum(MC, path)
                max_cont = pd.DataFrame(MC)

                MD = ExoticOptions.discrete_maximum(MD, path, period)
                max_disc = pd.DataFrame(MD)

                #Asian options
                cont_art_avg_rate_call = ExoticOptions.payoff_mean(r, T, art_cont_avg, iterator)
                CAARC.append([iterator, cont_art_avg_rate_call])

                cont_geo_avg_rate_call = ExoticOptions.payoff_mean(r, T, geo_cont_avg, iterator)
                CGARC.append([iterator, cont_geo_avg_rate_call])

                disc_art_avg_rate_call = ExoticOptions.payoff_mean(r, T, art_disc_avg, iterator)
                DAARC.append([iterator, disc_art_avg_rate_call])

                disc_geo_avg_rate_call = ExoticOptions.payoff_mean(r, T, geo_disc_avg, iterator)
                DGARC.append([iterator, disc_geo_avg_rate_call])

                cont_art_avg_rate_put = ExoticOptions.payoff_mean(r, T, iterator, art_cont_avg)
                CAARP.append([iterator, cont_art_avg_rate_put])

                cont_geo_avg_rate_put = ExoticOptions.payoff_mean(r, T, iterator, geo_cont_avg)
                CGARP.append([iterator, cont_geo_avg_rate_put])

                disc_art_avg_rate_put = ExoticOptions.payoff_mean(r, T, iterator, art_disc_avg)
                DAARP.append([iterator, disc_art_avg_rate_put])

                disc_geo_avg_rate_put = ExoticOptions.payoff_mean(r, T, iterator, geo_disc_avg)
                DGARP.append([iterator, disc_geo_avg_rate_put])

                cont_art_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, art_cont_avg)
                CAASC.append([iterator, cont_art_avg_strike_call])

                cont_geo_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, geo_cont_avg)
                CGASC.append([iterator, cont_geo_avg_strike_call])

                disc_art_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, art_disc_avg)
                DAASC.append([iterator, disc_art_avg_strike_call])

                disc_geo_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, geo_disc_avg)
                DGASC.append([iterator, disc_geo_avg_strike_call])

                cont_art_avg_strike_put = ExoticOptions.payoff_mean(r, T, art_cont_avg, path)
                CAASP.append([iterator, cont_art_avg_strike_put])

                cont_geo_avg_strike_put = ExoticOptions.payoff_mean(r, T, geo_cont_avg, path)
                CGASP.append([iterator, cont_geo_avg_strike_put])

                disc_art_avg_strike_put = ExoticOptions.payoff_mean(r, T, art_disc_avg, path)
                DAASP.append([iterator, disc_art_avg_strike_put])

                disc_geo_avg_strike_put = ExoticOptions.payoff_mean(r, T, geo_disc_avg, path)
                DGASP.append([iterator, disc_geo_avg_strike_put])

                #Lookback options
                cont_max_rate_call = ExoticOptions.payoff_mean(r, T, max_cont, iterator)
                CMRC.append([iterator, cont_max_rate_call])

                disc_max_rate_call = ExoticOptions.payoff_mean(r, T, max_disc, iterator)
                DMRC.append([iterator, disc_max_rate_call])

                cont_max_rate_put = ExoticOptions.payoff_mean(r, T, iterator, max_cont)
                CMSC.append([iterator, cont_max_rate_put])

                disc_max_rate_put = ExoticOptions.payoff_mean(r, T, iterator, max_disc)
                DMSC.append([iterator, disc_max_rate_put])

                cont_max_strike_call = ExoticOptions.payoff_mean(r, T, path, max_cont)
                CMRP.append([iterator, cont_max_strike_call])

                disc_max_strike_call = ExoticOptions.payoff_mean(r, T, path, max_disc)
                DMRP.append([iterator, disc_max_strike_call])

                cont_max_strike_put = ExoticOptions.payoff_mean(r, T, max_cont, path)  
                CMSP.append([iterator, cont_max_strike_put])

                disc_max_strike_put = ExoticOptions.payoff_mean(r, T, max_disc, path)
                DMSP.append([iterator, disc_max_strike_put])

        else:
            
            for iterator in range(int(initial_range * 100), int(final_range * 100 + 1), int(by * 100)):
                
                iterator = iterator / 100
                
                path = pd.DataFrame()
                
                if parameter == 'vol':
                    S = ExoticOptions.monte_carlo_simulation(So, r, iterator, T, t, sims)
                    path = pd.DataFrame(S)
                if parameter == 'r':
                    S = ExoticOptions.monte_carlo_simulation(So, iterator, vol, T, t, sims)
                    path = pd.DataFrame(S)
                if parameter == 'T':
                    S = ExoticOptions.monte_carlo_simulation(So, r, vol, iterator, t, sims)
                    path = pd.DataFrame(S)
                    
                AC = ExoticOptions.continuous_arithmetic_avrg(AC, path)
                art_cont_avg = pd.DataFrame(AC)

                GC = ExoticOptions.continuous_geometric_avrg(GC, path)
                geo_cont_avg = pd.DataFrame(GC)

                AD = ExoticOptions.discrete_arithmetic_avrg(AD, path, period)
                art_disc_avg = pd.DataFrame(AD)

                GD = ExoticOptions.discrete_geometric_avrg(GD, path, period)
                geo_disc_avg = pd.DataFrame(GD)

                MC = ExoticOptions.continuous_maximum(MC, path)
                max_cont = pd.DataFrame(MC)

                MD = ExoticOptions.discrete_maximum(MD, path, period)
                max_disc = pd.DataFrame(MD)

                #Asian options
                cont_art_avg_rate_call = ExoticOptions.payoff_mean(r, T, art_cont_avg, E)
                CAARC.append([iterator, cont_art_avg_rate_call])

                cont_geo_avg_rate_call = ExoticOptions.payoff_mean(r, T, geo_cont_avg, E)
                CGARC.append([iterator, cont_geo_avg_rate_call])

                disc_art_avg_rate_call = ExoticOptions.payoff_mean(r, T, art_disc_avg, E)
                DAARC.append([iterator, disc_art_avg_rate_call])

                disc_geo_avg_rate_call = ExoticOptions.payoff_mean(r, T, geo_disc_avg, E)
                DGARC.append([iterator, disc_geo_avg_rate_call])

                cont_art_avg_rate_put = ExoticOptions.payoff_mean(r, T, E, art_cont_avg)
                CAARP.append([iterator, cont_art_avg_rate_put])

                cont_geo_avg_rate_put = ExoticOptions.payoff_mean(r, T, E, geo_cont_avg)
                CGARP.append([iterator, cont_geo_avg_rate_put])

                disc_art_avg_rate_put = ExoticOptions.payoff_mean(r, T, E, art_disc_avg)
                DAARP.append([iterator, disc_art_avg_rate_put])

                disc_geo_avg_rate_put = ExoticOptions.payoff_mean(r, T, E, geo_disc_avg)
                DGARP.append([iterator, disc_geo_avg_rate_put])

                cont_art_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, art_cont_avg)
                CAASC.append([iterator, cont_art_avg_strike_call])

                cont_geo_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, geo_cont_avg)
                CGASC.append([iterator, cont_geo_avg_strike_call])

                disc_art_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, art_disc_avg)
                DAASC.append([iterator, disc_art_avg_strike_call])

                disc_geo_avg_strike_call = ExoticOptions.payoff_mean(r, T, path, geo_disc_avg)
                DGASC.append([iterator, disc_geo_avg_strike_call])

                cont_art_avg_strike_put = ExoticOptions.payoff_mean(r, T, art_cont_avg, path)
                CAASP.append([iterator, cont_art_avg_strike_put])

                cont_geo_avg_strike_put = ExoticOptions.payoff_mean(r, T, geo_cont_avg, path)
                CGASP.append([iterator, cont_geo_avg_strike_put])

                disc_art_avg_strike_put = ExoticOptions.payoff_mean(r, T, art_disc_avg, path)
                DAASP.append([iterator, disc_art_avg_strike_put])

                disc_geo_avg_strike_put = ExoticOptions.payoff_mean(r, T, geo_disc_avg, path)
                DGASP.append([iterator, disc_geo_avg_strike_put])

                #Lookback options
                cont_max_rate_call = ExoticOptions.payoff_mean(r, T, max_cont, E)
                CMRC.append([iterator, cont_max_rate_call])

                disc_max_rate_call = ExoticOptions.payoff_mean(r, T, max_disc, E)
                DMRC.append([iterator, disc_max_rate_call])

                cont_max_rate_put = ExoticOptions.payoff_mean(r, T, E, max_cont)
                CMSC.append([iterator, cont_max_rate_put])

                disc_max_rate_put = ExoticOptions.payoff_mean(r, T, E, max_disc)
                DMSC.append([iterator, disc_max_rate_put])

                cont_max_strike_call = ExoticOptions.payoff_mean(r, T, path, max_cont)
                CMRP.append([iterator, cont_max_strike_call])

                disc_max_strike_call = ExoticOptions.payoff_mean(r, T, path, max_disc)
                DMRP.append([iterator, disc_max_strike_call])

                cont_max_strike_put = ExoticOptions.payoff_mean(r, T, max_cont, path)  
                CMSP.append([iterator, cont_max_strike_put])

                disc_max_strike_put = ExoticOptions.payoff_mean(r, T, max_disc, path)
                DMSP.append([iterator, disc_max_strike_put])
            
        # Strike varying plot - Arithmetic Averages
        CAARCdf = pd.DataFrame(CAARC)
        CAARCx = CAARCdf[0]
        CAARCy = CAARCdf[1]

        DAARCdf = pd.DataFrame(DAARC)
        DAARCx = DAARCdf[0]
        DAARCy = DAARCdf[1]

        CAASCdf = pd.DataFrame(CAASC)
        CAASCx = CAASCdf[0]
        CAASCy = CAASCdf[1]

        DAASCdf = pd.DataFrame(DAASC)
        DAASCx = DAASCdf[0]
        DAASCy = DAASCdf[1]

        CAARPdf = pd.DataFrame(CAARP)
        CAARPx = CAARPdf[0]
        CAARPy = CAARPdf[1]

        DAARPdf = pd.DataFrame(DAARP)
        DAARPx = DAARPdf[0]
        DAARPy = DAARPdf[1]

        CAASPdf = pd.DataFrame(CAASP)
        CAASPx = CAASPdf[0]
        CAASPy = CAASPdf[1]

        DAASPdf = pd.DataFrame(DAASP)
        DAASPx = DAASPdf[0]
        DAASPy = DAASPdf[1]
        
        # Strike varying plot - Geometric Averages
        CGARCdf = pd.DataFrame(CGARC)
        CGARCx = CGARCdf[0]
        CGARCy = CGARCdf[1]

        DGARCdf = pd.DataFrame(DGARC)
        DGARCx = DGARCdf[0]
        DGARCy = DGARCdf[1]

        CGASCdf = pd.DataFrame(CGASC)
        CGASCx = CGASCdf[0]
        CGASCy = CGASCdf[1]

        DGASCdf = pd.DataFrame(DGASC)
        DGASCx = DGASCdf[0]
        DGASCy = DGASCdf[1]

        CGARPdf = pd.DataFrame(CGARP)
        CGARPx = CGARPdf[0]
        CGARPy = CGARPdf[1]

        DGARPdf = pd.DataFrame(DGARP)
        DGARPx = DGARPdf[0]
        DGARPy = DGARPdf[1]

        CGASPdf = pd.DataFrame(CGASP)
        CGASPx = CGASPdf[0]
        CGASPy = CGASPdf[1]

        DGASPdf = pd.DataFrame(DGASP)
        DGASPx = DGASPdf[0]
        DGASPy = DGASPdf[1]
        
        # Strike varying plot - Maximums
        CMRCdf = pd.DataFrame(CMRC)
        CMRCx = CMRCdf[0]
        CMRCy = CMRCdf[1]

        DMRCdf = pd.DataFrame(DMRC)
        DMRCx = DMRCdf[0]
        DMRCy = DMRCdf[1]

        CMSCdf = pd.DataFrame(CMSC)
        CMSCx = CMSCdf[0]
        CMSCy = CMSCdf[1]

        DMSCdf = pd.DataFrame(DMSC)
        DMSCx = DMSCdf[0]
        DMSCy = DMSCdf[1]

        CMRPdf = pd.DataFrame(CMRP)
        CMRPx = CMRPdf[0]
        CMRPy = CMRPdf[1]

        DMRPdf = pd.DataFrame(DMRP)
        DMRPx = DMRPdf[0]
        DMRPy = DMRPdf[1]

        CMSPdf = pd.DataFrame(CMSP)
        CMSPx = CMSPdf[0]
        CMSPy = CMSPdf[1]

        DMSPdf = pd.DataFrame(DMSP)
        DMSPx = DMSPdf[0]
        DMSPy = DMSPdf[1]
        
        title = ''
        string = ''
        
        if parameter == 'E':
            string = f'Strike from {int(initial_range)} to {int(final_range)}'
            title = 'Strike'
        if parameter == 'vol':
            string = f'Volatility from {int(initial_range * 100)}% to {int(final_range * 100)}%'
            title = 'Volatility'
        if parameter == 'T':
            string = f'Time-to-expiry from {int(initial_range)} to {int(final_range)} years'
            title = 'Time-to-expiry'
        if parameter == 'r':
            string = f'Risk-free Interest Rate from {int(initial_range * 100)}% to {int(final_range * 100)}%'
            title = 'Risk-free Interest Rate'
        
        figsize = (8, 10) 
        axs = plt.figure(figsize=figsize, constrained_layout=True)
        axs = axs.subplots(4, 2)
        axs[0, 0].plot(CAARCx, CAARCy, color = 'b')
        axs[0, 0].set_title('Continuous Moving Average\nRate Call')
        axs[1, 0].plot(DAARCx, DAARCy, color = 'b')
        axs[1, 0].set_title('Discrete Moving Average\nRate Call')
        axs[2, 0].plot(CAASCx, CAASCy, color = 'b')
        axs[2, 0].set_title('Continuous Moving Average\nStrike Call')
        axs[3, 0].plot(DAASCx, DAASCy, color = 'b')
        axs[3, 0].set_title('Discrete Moving Average\nStrike Call')
        axs[0, 1].plot(CAARPx, CAARPy, color = 'b')
        axs[0, 1].set_title('Continuous Moving Average\nRate Put')
        axs[1, 1].plot(DAARPx, DAARPy, color = 'b')
        axs[1, 1].set_title('Discrete Moving Average\nRate Put')
        axs[2, 1].plot(CAASPx, CAASPy, color = 'b')
        axs[2, 1].set_title('Continuous Moving Average\nStrike Put')
        axs[3, 1].plot(DAASPx, DAASPy, color = 'b')
        axs[3, 1].set_title('Discrete Moving Average\nStrike Put')
        plt.suptitle(f'Asian options based on Arithmetic Average varying\n{title}', fontsize = 18)
        
        for ax in axs.flat:
            ax.set(xlabel=f'{string}', ylabel='Option payoff')

        figsize = (8, 10) 
        axs1 = plt.figure(figsize=figsize, constrained_layout=True)
        axs1 = axs1.subplots(4, 2)
        axs1[0, 0].plot(CGARCx, CGARCy, color = 'b')
        axs1[0, 0].set_title('Continuous Moving Average\nRate Call')
        axs1[1, 0].plot(DGARCx, DGARCy, color = 'b')
        axs1[1, 0].set_title('Discrete Moving Average\nRate Call')
        axs1[2, 0].plot(CGASCx, CGASCy, color = 'b')
        axs1[2, 0].set_title('Continuous Moving Average\nStrike Call')
        axs1[3, 0].plot(DGASCx, DGASCy, color = 'b')
        axs1[3, 0].set_title('Discrete Moving Average\nStrike Call')
        axs1[0, 1].plot(CGARPx, CGARPy, color = 'b')
        axs1[0, 1].set_title('Continuous Moving Average\nRate Put')
        axs1[1, 1].plot(DGARPx, DGARPy, color = 'b')
        axs1[1, 1].set_title('Discrete Moving Average\nRate Put')
        axs1[2, 1].plot(CGASPx, CGASPy, color = 'b')
        axs1[2, 1].set_title('Continuous Moving Average\nStrike Put')
        axs1[3, 1].plot(DGASPx, DGASPy, color = 'b')
        axs1[3, 1].set_title('Discrete Moving Average\nStrike Put')
        plt.suptitle(f'Asian options based on Geometric Average varying\n{title}', fontsize = 18)
        
        for ax1 in axs1.flat:
            ax1.set(xlabel=f'{string}', ylabel='Option payoff')

        figsize = (8, 10) 
        axs2 = plt.figure(figsize=figsize, constrained_layout=True)
        axs2 = axs2.subplots(4, 2)
        axs2[0, 0].plot(CMRCx, CMRCy, color = 'b')
        axs2[0, 0].set_title('Continuous Maximum\nRate Call')
        axs2[1, 0].plot(DMRCx, DMRCy, color = 'b')
        axs2[1, 0].set_title('Discrete Maximum\nRate Call')
        axs2[2, 0].plot(CMSCx, CMSCy, color = 'b')
        axs2[2, 0].set_title('Continuous Maximum\nStrike Call')
        axs2[3, 0].plot(DMSCx, DMSCy, color = 'b')
        axs2[3, 0].set_title('Discrete Maximum\nStrike Call')
        axs2[0, 1].plot(CMRPx, CMRPy, color = 'b')
        axs2[0, 1].set_title('Continuous Maximum\nRate Put')
        axs2[1, 1].plot(DMRPx, DMRPy, color = 'b')
        axs2[1, 1].set_title('Discrete Maximum\nRate Put')
        axs2[2, 1].plot(CMSPx, CMSPy, color = 'b')
        axs2[2, 1].set_title('Continuous Maximum\nStrike Put')
        axs2[3, 1].plot(DMSPx, DMSPy, color = 'b')
        axs2[3, 1].set_title('Discrete Maximum\nStrike Put')
        plt.suptitle(f'Lookback options varying\n{title}', fontsize = 18)
        
        for ax2 in axs2.flat:
            ax2.set(xlabel=f'{string}', ylabel='Option payoff')

