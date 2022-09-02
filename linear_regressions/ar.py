
import numpy as np

from survey_stats.linear_regressions import ols
from survey_stats.data_process import Data, Sample
from survey_stats.basic_model import Formula, Formulas

class Model(ols.Model):
    def __init__(self, dep_var: str, formula: str, lags:list[int]):
        super().__init__(dep_var, formula)
        self.lags = lags
        new_formula = ''
        l = 0
        for formula in Formula(self.formula).split().formulas:
            if new_formula == '':
                if formula == '1':
                    new_formula += formula
                for l in self.lags:
                    if new_formula == '':
                        new_formula += f'lag({self.dep_var},{l})'
                    else:
                        new_formula += f'+lag({self.dep_var},{l})'
                if formula != '1':
                    new_formula += f'+{formula}'
            else:
                new_formula += f'+{formula}'
        self.formula = new_formula
    
    def estimate(self, sample: Sample, print_equation: bool = True):
        eq = super().estimate(sample, print_equation)
        return Equation(eq.dep_var, self.lags, eq.indep_vars,
                    eq.indep_coefs, eq.cov_var_coefs,
                    eq.df, eq.mse, eq.r2,eq.r2adj,
                    eq.table, eq.sample)
    
    def estimate_most_significant(self, sample: Sample, min_significant=1, print_equation=True):
        eq = super().estimate_most_significant(sample, min_significant, print_equation)
        new_lags = []
        for var in eq.indep_vars:
            for lag in self.lags:
                if f'lag({self.dep_var},{lag})' == var:
                    new_lags.append(lag)
                    break
        self.lags = new_lags

        return Equation(eq.dep_var, self.lags, eq.indep_vars,
                    eq.indep_coefs, eq.cov_var_coefs,
                    eq.df, eq.mse, eq.r2,eq.r2adj,
                    eq.table, eq.sample)
    
    def estimate_skip_collinear(self, sample: Sample, print_equation: bool = True, min_df:int=10, print_progress:bool=True):
        try:
            eq = super().estimate_skip_collinear(sample, print_equation, min_df, print_progress)
            new_lags = []
            for var in eq.indep_vars:
                for lag in self.lags:
                    if f'lag({self.dep_var},{lag})' == var:
                        new_lags.append(lag)
                        break
            self.lags = new_lags
            return Equation(eq.dep_var, self.lags, eq.indep_vars,
                        eq.indep_coefs, eq.cov_var_coefs,
                        eq.df, eq.mse, eq.r2,eq.r2adj,
                        eq.table, eq.sample)
        except Exception as e:
            print(f'Error! dependent variable: {self.dep_var}, formula: {self.formula}. {e}')

class Equation(ols.Equation):
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

    def forecast(self, sample: Sample, is_dynamic:bool=True) -> Data:
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
                            yf += forecast_data.values[self.dep_var][sample.data.index()[n-lag]] * float(coefs_lags[j])
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

