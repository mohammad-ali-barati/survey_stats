from __future__ import annotations
import time
import numpy as np
import scipy.stats as scipy_stats

from survey_stats.linear_regressions import ols
from survey_stats.data_process import Data, Sample
from survey_stats.basic_model import Formula, Formulas, Variable, Variables
from survey_stats.functions import seconds_to_days_hms

class Model:
    def __init__(self, dep_var: str, formula: str, lags:list[int],
                    do_lags_to_regressors:bool=False):
        self.dep_var = dep_var
        self.lags = lags
        self.do_lags_to_regressors = do_lags_to_regressors
        new_formula = ''
        l = 0
        for formula in Formula(formula).split().formulas:
            if new_formula == '':
                if formula == '1':
                    new_formula += formula
                for l in self.lags:
                    if new_formula == '':
                        new_formula += f'lag({self.dep_var},{l})'
                    else:
                        new_formula += f'+lag({self.dep_var},{l})'
            else:
                new_formula += f'+{formula}'
            if self.do_lags_to_regressors and formula != '1':
                for l in self.lags:
                    new_formula += f'+lag({formula},{l})'
        self.formula = new_formula
        
    
    def estimate(self, sample: Sample, print_equation: bool = True)->Equation:
        eq = ols.Model(self.dep_var, self.formula).estimate(sample, print_equation)
        return Equation(eq.dep_var, self.lags, eq.indep_vars,
                        eq.indep_coefs, eq.cov_var_coefs,
                        eq.df, eq.mse, eq.r2, eq.r2adj,
                        eq.table, eq.sample)
    
    def estimate_most_correlated(self, sample:Sample, max_variables:int=10, min_correlated:int=0,
                                min_observation:int=20,
                                print_equation:bool=True,
                                print_progress:bool=True):
        data = Formulas(self.formula.split('+')).calculate_all(sample.data, print_progress=False)
        data.add_data(sample.data.select_variables([self.dep_var]))
        vars = [Variable(var) for var in data.variables()]
        cors = []
        for var in vars[:-1]:
            if var != '1':
                try:
                    cor = Variables([var, vars[-1]]).stats.correlation(
                            Sample(data, sample.index), False).values[var.name][self.dep_var]
                    if cor.imag != 0:
                        cors.append((var,0))
                    else:
                        cors.append((var.name, abs(cor.real)))
                except Exception as e:
                    if print_progress:
                        print(f"Error in correlation of {var.name}. {e}")
        cors.sort(key=lambda i: i[1])
        i, vars = 0, []
        for var, cor in cors:
            if i > max_variables:
                break
            if cor >= min_correlated and \
                Variable(var).stats.count(Sample(data, sample.index)) >= min_observation:
                i += 1
                vars.append(var)
        if '1+' in self.formula.replace(' ','') or '+1+' in self.formula.replace(' ',''):
            f = '1+'
        else:
            f = ''
        f += '+'.join(vars[:-1])
        eq = Model(self.dep_var, f, self.lags, self.do_lags_to_regressors).estimate(sample, print_equation)
        return eq

    @staticmethod
    def __most_significant(dep_var: str, formula: str, lags:list[int], sample: Sample, min_significant=1, print_equation=True):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            removes, k = 1, 0
            while k < 100:
                k += 1
                p_values, t_students = [], []
                try:
                    eq = Model(dep_var, formula, lags).estimate(sample, False)
                    for i, coef in enumerate(eq.indep_coefs):
                        t_student = abs(coef/eq.cov_var_coefs[i][i]**0.5)
                        t_students.append(t_student)
                        p_value = (1 - scipy_stats.t.cdf(t_student, eq.df))*2
                        p_values.append(p_value)
                    if p_values == []:
                        break
                    removes = 1
                except:
                    removes += 1
                if p_values != []:
                    if (np.isnan(max(p_values)) or max(p_values) >= min_significant):
                        # decrease formula
                        t_students = [1 if np.isnan(
                            t) else t for t in t_students]
                        formula_new = ''
                        removed = False
                        for i, var in enumerate(eq.indep_vars):
                            if (not t_students[i] in sorted(t_students)[:removes]) or (t_students[i] == 1 and removed):
                                formula_new += var if formula_new == '' else f'+{var}'
                            else:
                                removed = True
                        if formula_new == formula:
                            break
                        formula = formula_new
                    else:
                        break
            if p_values != []:
                if print_equation:
                    print(f"The number of iterations: {k}")
                    print(eq.table)
                return eq
            else:
                raise ValueError(
                    f"Error! there isn't any model with minimum significant of {min_significant}.")

    def estimate_most_significant(self, sample: Sample, min_significant=1, print_equation=True):
        eq = Model.__most_significant(
            self.dep_var, self.formula, self.lags, sample, min_significant, print_equation)
        return eq


    #TODO estimate_skip_collinear() must be check and repair.
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
    def __estimate_skip_collinear(sample: Sample, dep_var, lags, vars: list[str], cors: dict, method:str='min_indep',
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
                    eq = Model(dep_var, f, lags).estimate(sample, print_equation)
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
                return Model.__estimate_skip_collinear(sample, dep_var, lags, vars, cors, method, print_equation, min_df, print_progress)

    def estimate_skip_collinear(self, sample: Sample, print_equation: bool = True, 
                min_df: int = 10, method: str = 'min_indep', print_progress: bool = True):
        vars = Formula(self.formula).split().formulas
        vars.append(self.dep_var)
        data = Formulas(vars).calculate_all(sample.data)
        sample = Sample(data, sample.index)
        cors = Variables([Variable(v) for v in vars]).stats.correlation(sample)
        vars.remove(self.dep_var)
        Model.progress, Model.start_time, Model.left_time = 0, time.perf_counter(), time.perf_counter()
        return Model.__estimate_skip_collinear(sample, self.dep_var, self.lags, vars, cors, method, print_equation, min_df, print_progress)

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
            indep_vars, no=[], 0
            for cor in cors_sorted:
                if no >= vars_no:
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
                        no += 1
                    vars_.remove(var)
                    cors_.remove(cor)
            f = '' if no_constant else '1+'
            f += '+'.join(indep_vars)
            return Model(self.dep_var, f).estimate(sample, print_equation, motor)

class Equation:
    def __init__(self, dep_var: str, lags:int, indep_vars: list[str], 
                    indep_coefs: list[float], cov_var_coefs: list[list], 
                    df: int, mse:float, r2: float, r2adj: float, 
                    table: str, sample: Sample) -> None:
        self.dep_var = dep_var
        self.lags = lags
        self.indep_vars = indep_vars
        self.indep_coefs = indep_coefs
        self.cov_var_coefs = cov_var_coefs
        self.df =df
        self.mse = mse
        self.r2 = r2
        self.r2adj = r2adj
        self.table = table
        self.sample = sample

    def forecast(self, sample: Sample, is_dynamic: bool = True) -> Data:
        if not self.dep_var in sample.data.variables():
            data = Formula(self.dep_var).calculate(sample.data)
        else:
            data = sample.data
        if self.indep_vars[0]=='1':
            indep_lags = self.indep_vars[1:len(self.lags)+1]
            indep_exos = ['1'] + self.indep_vars[len(self.lags)+1:]
            coefs_lags = self.indep_coefs[1:len(self.lags)+1]
            coefs_exos = [self.indep_coefs[0]] + list(self.indep_coefs[len(self.lags)+1:])
        else:
            indep_lags = self.indep_vars[:len(self.lags)]
            indep_exos = self.indep_vars[len(self.lags):]
            coefs_lags = self.indep_coefs[:len(self.lags)]
            coefs_exos = self.indep_coefs[len(self.lags):]
        exos_data = Formulas(indep_exos).calculate_all(sample.data)
        forecast_data = Data(sample.data.type, {})
        forecast_data.values[self.dep_var + '_f'] = {}
        for i in sample.data.index():
            if not i in sample.index:       #not np.isnan(data.values[self.dep_var][i]) and 
                forecast_data.values[self.dep_var + '_f'][i] = data.values[self.dep_var][i]
            elif i in sample.index:
                n = sample.data.index().index(i)
                try:
                    yf = 0
                    for j,lag in enumerate(self.lags): 
                        if is_dynamic:
                            yf += forecast_data.values[self.dep_var + '_f'][sample.data.index()[n-lag]] * float(coefs_lags[j])
                        else:
                            yf += sample.data.values[self.dep_var][sample.data.index()[n-lag]] * float(coefs_lags[j])
                    for j,var in enumerate(indep_exos):
                        yf += exos_data.values[var][i] * float(coefs_exos[j])
                    forecast_data.values[self.dep_var + '_f'][i] = yf
                except:
                    forecast_data.values[self.dep_var + '_f'][i] = np.nan
        return forecast_data.select_index(sample.index)

    def forecast_std(self, sample: Sample) -> Data:
        if not self.dep_var in self.sample.data.variables():
            data = Formula(self.dep_var).calculate(self.sample.data)
        else:
            data = self.sample.data
        if self.indep_vars[0]=='1':
            indep_lags = self.indep_vars[1:len(self.lags)+1]
            indep_exos = ['1'] + self.indep_vars[len(self.lags)+1:]
            coefs_lags = self.indep_coefs[1:len(self.lags)+1]
            coefs_exos = [self.indep_coefs[0]] + list(self.indep_coefs[len(self.lags)+1:])
        else:
            indep_lags = self.indep_vars[:len(self.lags)]
            indep_exos = self.indep_vars[len(self.lags):]
            coefs_lags = self.indep_coefs[:len(self.lags)]
            coefs_exos = self.indep_coefs[len(self.lags):]
        exos_data = Formulas(indep_exos).calculate_all(sample.data)
        
        std_data = Data(sample.data.type, {})
        std_data.values[self.dep_var + '_std'] = {}
        for i in sample.data.index():
            if not np.isnan(data.values[self.dep_var][i]) and not i in sample.index:
                std_data.values[self.dep_var + '_std'][i] = 0
            elif i in sample.index:
                n = sample.data.index().index(i)
                var_yf = 0
                for j,lag in enumerate(self.lags): 
                    var_yf += (std_data.values[self.dep_var + '_std'][sample.data.index()[n-lag]] \
                                * float(coefs_lags[j]))**2 + self.mse
                std_data.values[self.dep_var + '_std'][i] = var_yf**0.5
        return std_data.select_index(sample.index)

