import pickle
import time
import scipy.stats as scipy_stats
import numpy as np
from survey_stats.basic_model import Formula, Formulas, Variable_Types, Variable, Variables
from survey_stats.data_process import Data, Sample
from survey_stats.functions import number_of_digits, seconds_to_days_hms
from survey_stats.linear_regressions import ols

import pandas as pd

class Model:
    def __init__(self, dep_var, formula):
        self.dep_var = dep_var
        self.formula = formula

    def __str__(self):
        res = ''
        for i, formula in enumerate(Formula(self.formula).split().formulas):
            if formula == '1':
                res += f' + c({i})'
            else:
                res += f' + c({i})*{formula}'
        return self.dep_var + ' = ' + res[3:]
    
    motors = ['statsmodels',]
    def estimate(self, sample:Sample, print_equation:bool=True, motor:str='statsmodels'):
        if self.formula != '':
            indep_num = Formula(self.formula).split().calculate_all(sample.get_data(), skip_collinear=True)
            dep_num = Formula(self.dep_var).calculate(sample.get_data(),skip_collinear=True)
            if len(dep_num.variables()) > 1:
                raise ValueError(f"Error! variable {self.dep_var} is not binary.")
            dep_name = dep_num.variables()[0]
            indep_names = indep_num.variables()
            indep_num.add_data(dep_num)
            indep_num.dropna()
            dep_num = indep_num.select_variables([dep_name])
            vs = set([v for v in dep_num.values[self.dep_var].values() if not np.isnan(v)])
            dep_num = dep_num.add_dummies([[(dep_name, '=', v)] for v in vs])
            indep_num.drop([dep_name])
            y = dep_num.to_numpy()
            x = indep_num.to_numpy()
            import warnings
            if motor == 'statsmodels':
                import statsmodels.api as sm
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    warnings.filterwarnings("ignore", category=RuntimeWarning) 
                    model = sm.MNLogit(y, x)
                    model.exog_names[:] = indep_names
                    res = model.fit(disp=0, skip_hessian= 0, method_kwargs={'warn_convergence': False})
                indep_coefs = res.params
                cov_var_coefs = res.cov_params()
                table = res.summary()
            df_resid = res.df_resid
            df_model = res.df_model
            r2 = res.prsquared
            r2adj = 1 - (1-r2)*(len(dep_num)-1)/(len(dep_num)-len(indep_names)-1)
            if print_equation:
                print(table)
            return Equation(dep_name, indep_names, indep_coefs, cov_var_coefs, df_resid, df_model,r2,r2adj, table, sample,y,x)

    @staticmethod
    def __observation(sample: Sample, formulas: list[str]):
        sts, ens = [], []
        for var in formulas:
            indices = [i for i, value in sample.data.values[var].items(
            ) if not np.isnan(value) and i in sample.index]
            sts.append(indices[0])
            ens.append(indices[-1])
        sts.sort()
        ens.sort()
        st, en = sts[-1], ens[0]
        return sample.index.index(en)-sample.index.index(st)+1

    progress, start_time, left_time = 0, time.perf_counter(), time.perf_counter()
    @staticmethod
    def __estimate_skip_collinear(sample: Sample, dep_var, vars: list[str], cors: dict, method:str='min_indep',
                        print_equation: bool = True, min_df: int = 10, print_progress: bool = True):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            if print_progress:
                Model.progress += 1
                Model.left_time = time.perf_counter()-Model.start_time
                Model.remain_time = Model.left_time/Model.progress * len(vars)
                print(f'Progress: {Model.progress}. Number of Variables: {len(vars)}. left time: {seconds_to_days_hms(Model.left_time)}. remain time: {seconds_to_days_hms(Model.remain_time)}.'+' '*20, end='\r')
            if vars == []:
                return []
            try:
                if Model.__observation(sample, vars) >= len(vars)+min_df:
                    f = '+'.join(vars)
                    eq = Model(dep_var, f).estimate(sample, print_equation)
                else:
                    raise
                if print_progress:
                    print()
                return eq
            except:
                # remove most correlation with dependent variable
                if method == 'max_dep':
                    var_min, cor_min = '', 1
                    for var in vars:
                        if cors.values[var][dep_var].imag == 0:
                            if cors.values[var][dep_var].real < cor_min:
                                cor_min = cors.values[var][dep_var].real
                                var_min = var
                    if var_min=='':
                        var_min = vars[-1]
                    vars.remove(var_min)
                else:
                # remove minimum correlation with indepenent variables, this don't complete
                    var_max, cor_max = '', 0
                    for var in vars:
                        try:
                            mean_cor = sum([cors.values[var][v] for v in vars if v != var 
                                                        and cors.values[var][v].imag == 0
                                            and not np.isnan(cors.values[var][v].real)])/(len([v for v in vars if v != var
                                                                                            and cors.values[var][v].imag == 0
                                                                                            and not np.isnan(cors.values[var][v].real)]))
                            # print(var, mean_cor)
                            if not np.isnan(mean_cor.real):
                                if mean_cor.real > cor_max:
                                    var_max, cor_max = var, mean_cor.real
                        except:
                            pass
                    if var_max == '':
                        var_max = vars[-1]
                    vars.remove(var_max)
                return Model.__estimate_skip_collinear(sample, dep_var, vars, cors, method, print_equation, min_df, print_progress)

    def estimate_skip_collinear(self, sample: Sample, print_equation: bool = True, 
                min_df: int = 10, method: str = 'min_indep', print_progress: bool = True):
        vars = Formula(self.formula).split().formulas
        vars.append(self.dep_var)
        data = Formulas(vars).calculate_all(sample.data)
        sample = Sample(data, sample.index)
        cors = Variables([Variable(v) for v in vars]).stats.correlation(sample)
        vars.remove(self.dep_var)
        Model.progress, Model.start_time, Model.left_time = 0, time.perf_counter(), time.perf_counter()
        return Model.__estimate_skip_collinear(sample, self.dep_var, vars, cors, method, print_equation, min_df, print_progress)

    def estimate_constrained(self, sample: Sample, vars_no:int, min_obs:int=10, max_obs:int=100, max_r2:float=1,
                            no_constant:bool=False,
                            print_equation: bool = True, 
                            motor: str ='statsmodels'):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            vars = Formula(self.formula).split().formulas
            vars.append(self.dep_var)
            data = Formulas(vars).calculate_all(sample.data)
            sample = Sample(data, sample.index[-max_obs:])
            vars.remove(self.dep_var)
            cors = []
            for var in vars:
                cor = Variables([Variable(self.dep_var), Variable(var)]
                                ).stats.correlation(sample
                                    ).values[self.dep_var][var]
                if cor.imag != 0:
                    cors.append(0)
                else:
                    cors.append(cor.real)
            cors_sorted = sorted(cors, reverse=True)
            vars_, cors_ = vars.copy(), cors.copy()
            indep_vars, no=[], vars_no
            for cor in cors_sorted:
                if no == 0:
                    break
                else:
                    var = vars_[cors_.index(cor)]
                    r2 = True
                    if len(indep_vars)>1 and max_r2<1:
                        f = '1+' + '+'.join(indep_vars)
                        try:
                            r2 = ols.Model(var, f).estimate(sample, False).r2 <= max_r2
                        except Exception as e:
                            pass
                    if Variable(var).stats.count(sample) >= min_obs and r2:
                        indep_vars.append(var)
                        no -= 1
                    vars_.remove(var)
                    cors_.remove(cor)
            f = '' if no_constant else '1+'
            f += '+'.join(indep_vars)
            return Model(self.dep_var, f).estimate(sample, print_equation, motor)

class Equation:
    def __init__(self, dep_var:str, indep_vars:list[str], indep_coefs:np.ndarray, 
                    cov_var_coefs:np.ndarray, df_resid:int, df_model:int,r2:float,r2adj:float, table:str, sample:Sample,y,x) -> None:
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.indep_coefs = indep_coefs
        self.cov_var_coefs = cov_var_coefs
        self.df_resid = df_resid
        self.df_model = df_model
        self.r2 = r2
        self.r2adj = r2adj
        self.table = table
        self.sample = sample
        self.y = y
        self.x = x
    
    def __str__(self):
        len_dep_var = max([len(x) for x in [self.dep_var, '(p-value)']])
        eq, sds, ts, ps = '', '', '', ''
        for i, var in enumerate(self.indep_vars):
            len_pre = max([len(x) for x in [eq, sds, ts, ps]])
            eq, sds, ts, ps = [x.ljust(len_pre) for x in [eq, sds, ts, ps]]
            c = float(self.indep_coefs[i])
            sd = float(self.cov_var_coefs[i][i] ** 0.5)
            z = c / sd
            p = (1- scipy_stats.norm.cdf(abs(z), loc=0, scale=1))*2
            len_indep_var = int(max([number_of_digits(x) for x in [c,sd,z,p]]))
            if var == '1':
                if c > 0:
                    eq += ' +  ' + f'{c:.4f}'.center(len_indep_var)
                elif c < 0:
                    eq += ' -  ' + f'{-c:.4f}'.center(len_indep_var)
            else:
                if c > 0:
                    eq += ' +  ' + f'{c:.4f}*{var}'.center(len_indep_var)
                elif c < 0:
                    eq += ' -  ' + f'{-c:.4f}*{var}'.center(len_indep_var)
            sds += '   ' + f'({sd:.4f})'.ljust(len_indep_var)
            ts += '   ' + f'({z:.4f})'.ljust(len_indep_var)
            ps += '   ' + f'({p:.4f})'.ljust(len_indep_var)
        eq = self.dep_var.center(len_dep_var) + ' = ' + eq[3:]
        sds = '(std.)'.center(len_dep_var) + sds
        ts = '(z)'.center(len_dep_var) + ts
        ps = '(p-value)'.center(len_dep_var) + ps
        return f'{eq}\n{sds}\n{ts}\n{ps}'

    def goodness_of_fit(self, sample:Sample):
        y_arr = Formula(self.dep_var).calculate(sample.get_data()).to_numpy()
        x_arr = Formulas(self.indep_vars).calculate_all(sample.get_data()).to_numpy()
        yf_prob = 1/(1+np.exp(-np.dot(x_arr, self.indep_coefs))).T[0]
        yf_arr = np.round(yf_prob, 0)
        counts = [[sum(yf_arr*y_arr), sum(yf_arr*(1-y_arr))],
                [sum((1-yf_arr)*y_arr), sum((1-yf_arr)*(1-y_arr))]]
        max_a_width = len(f'{max([max(row) for row in counts])}')
        column_width = 5 + max_a_width + 5
        table_width = column_width * 2
        total_width = table_width + 16*2
        res = 'Confusion Matrix'.center(total_width, '-') + '\n'
        res += 'observations'.center(total_width) + '\n'
        res += ('1'.center(column_width) + '0'.center(column_width)).center(total_width) + '\n'
        res += ('-'*column_width*2).center(total_width) + '\n'
        res += 'forecasts 1' + ' '*4 + '|' + (f'{int(counts[0][0])}'.center(column_width) + '|' +
                            f'{int(counts[0][1])}'.center(column_width-1)).center(table_width) + '|\n'
        res += ('-'*column_width*2).center(total_width) + '\n'
        res += '          0' + ' '*4 + '|' + (f'{int(counts[1][0])}'.center(column_width) + '|' +
                            f'{int(counts[1][1])}'.center(column_width-1)).center(table_width) + '|\n'
        res += ('-'*column_width*2).center(total_width) + '\n'
        res += '\n\n'
        res += f"Count's R2 = {float((counts[0][0]+counts[1][1])/sum([sum(row) for row in counts])):.4f}" + '\n'
        bs = float(sum((yf_prob-y_arr)**2)/len(y_arr))
        res += f'Brier Score (BS) = {bs:.4f}' + '\n'
        y_mean = float(sum(y_arr)/len(y_arr))
        efron = float(1 - sum((y_arr-yf_prob)**2)/sum((y_arr-y_mean)**2))
        res += f"Efron's R2 = {efron:.4f}" + '\n'
        res += '-' * total_width
        return res

    def forecast(self, sample:Sample):
        data = Formulas(self.indep_vars).calculate_all(sample.data)
        values = {}
        for var in self.indep_vars:
            if not var in values.keys():
                values[var] = {}
            for i in sample.index:
                if var == '1':
                    values[var][i] = 1
                elif var in data.variables():
                    values[var][i] = data.values[var][i]
                else:
                    values[var][i] = np.nan
        indep_num = Data(values=values).to_numpy() 

        import warnings
        import statsmodels.api as sm
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = sm.MNLogit(self.y, self.x)
            res_num = model.predict(self.indep_coefs, indep_num)

        values = {}
        for i, index in enumerate(res_num):
            for j, p in enumerate(index):
                if not f'{self.dep_var}_{j}' in values.keys():
                    values[f'{self.dep_var}_{j}'] = {}
                values[f'{self.dep_var}_{j}'][sample.index[i]] = p
        return Data(values = values)

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print('Results were saved successfully')

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'rb') as f:
            eq = pickle.load(f)
        return eq
