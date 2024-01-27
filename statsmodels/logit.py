import pickle
import time
import scipy.stats as scipy_stats
import numpy as np
from survey_stats.basic_model import Formula
from survey_stats.data_process import Data, Sample
from survey_stats.functions import number_of_digits, seconds_to_days_hms
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


class Model:
    def __init__(self, dep_var:str, formula:str='', indep_vars:str=[], has_constant:bool=True):
        self.dep_var = dep_var
        formula = formula.replace(' ','')
        if has_constant and formula!='' and (not '1+'!=formula[:2]) and (not '+1+' in formula) and (not '+1'!=formula[-2:]):
            formula = '1+' + formula
        self.formula = formula
        self.indep_vars = list(set(indep_vars))
        self.has_constant = has_constant

    def __str__(self):
        if self.formula != '':
            res = ''
            for i, formula in enumerate(Formula(self.formula).split().formulas):
                if formula == '1':
                    res += f' + c({i})'
                else:
                    res += f' + c({i})*{formula}'
            return self.dep_var + ' = ' + res[3:]
        elif self.indep_vars != []:
            res = 'c(0) + '
            for i, formula in enumerate(self.indep_vars):
                res += f' + c({i+1})*{formula}'
            return self.dep_var + ' = ' + res[3:]
        return ''

    def estimate(self, sample:Sample, print_progress:bool=False, indent:int=0, max_lenght:int=-1):
        if self.formula != '':
            if sample.weights == '1':
                data = sample.data.select_variables([v for v in sample.data.variables() if v in self.formula or v in self.dep_var]).select_index(sample.index.copy())
            elif sample.weights in data.variables():
                data = sample.data.select_variables([v for v in sample.data.variables() if v in self.formula or v in self.dep_var or v == sample.weights]).select_index(sample.index.copy())
            indep_num = Formula(self.formula).split().calculate_all(data)
            indep_names = indep_num.variables()
            self.indep_vars = indep_names.copy()
            if self.has_constant:
                self.indep_vars.remove('1')
            dep_num = Formula(self.dep_var).calculate(data)
            indep_num.add_data(dep_num)
            data = indep_num
            if sample.weights != '1':
                w_num = sample.data.select_variables([sample.weights])
                data.dropna()
                w_num = data.select_variables([sample.weights])
            else:
                data.dropna()
            y_arr = data.select_variables([self.dep_var]).to_numpy()
            x_arr = data.select_variables(indep_names).to_numpy()          
        elif self.indep_vars != []:
            if sample.weights == '1':
                data = sample.data.select_variables([self.dep_var]+self.indep_vars).select_index(sample.index.copy())
            elif sample.weights in data.variables():
                data = sample.data.select_variables([self.dep_var]+self.indep_vars+[sample.weights]).select_index(sample.index.copy())
            data.dropna()
            if sample.weights != '1':
                w_num = data.select_variables([sample.weights])
            data.add_a_variable('1', [1 for i in data.index()])
            y_arr = data.select_variables([self.dep_var]).to_numpy()
            x_arr = data.select_variables(['1']+self.indep_vars).to_numpy()
        else:
            raise ValueError(f"Error! formula or indep_vars are not defined!")

        if sample.weights == '1':
            w = 1
            try:
                regr = sm.Logit(y_arr ,x_arr)
                regr.exog_names[:] = indep_names
                model = regr.fit(disp=0, skip_hessian= 0, method_kwargs={'warn_convergence': False})
                indep_coefs = list(model.params)
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        else:
            w = np.array([w for _,w in w_num.values[sample.weights].items()])
            try:
                regr = sm.GLM(y_arr, x_arr, family=sm.families.Binomial(), freq_weights=w)
                regr.exog_names[:] = indep_names
                model = regr.fit(disp=0, skip_hessian= 0, method_kwargs={'warn_convergence': False})
                indep_coefs = list(model.params)
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        eq = Equation(self.dep_var, self.formula, self.indep_vars, self.has_constant, indep_coefs, model,
                                                    x_arr, y_arr,w)
        if print_progress:
            if max_lenght==-1 or max_lenght<3:
                print(' '*indent+str(eq))
            else:
                print(' '*indent+str(eq)[:max_lenght-3]+'...')
        return eq
    
        

class Equation:
    def __init__(self, dep_var:str, formula:str, indep_vars:list[str], has_constant:bool, indep_coefs:list[float],
                    model,
                    x_arr:np.ndarray=None, y_arr:np.ndarray=None,
                    w:np.ndarray=None) -> None:
        self.dep_var = dep_var
        self.formula = formula
        self.indep_vars = indep_vars
        self.has_constant = has_constant
        self.indep_coefs = indep_coefs
        self.model = model
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.w = w
        self.params =self.params(self.indep_coefs, model, x_arr, y_arr, w)

    class params:
        def __init__(self, indep_coefs:list[float], model,
                x_arr:np.ndarray=None, y_arr:np.ndarray=None,
                w:np.ndarray=None) -> None:
            self.indep_coefs = indep_coefs
            self.model = model
            self.x_arr = x_arr
            self.y_arr =y_arr
            self.w = w
        
        def cov_var_coefs(self):
            return self.model.cov_params()


    def __str__(self):
        res = ''
        if self.has_constant:
            coef=self.indep_coefs[0]
            if abs(coef)<.1 or abs(coef)>1000:
                res += f'{coef:.2e} '
            else:
                res += f'{coef:.2f} '
            i = 1
        else:
            i = 0
        for var in self.indep_vars:
            coef=self.indep_coefs[i]
            if abs(coef)<.1 or abs(coef)>1000:
                if coef<0:
                    res += f'- {-coef:.2e} {var} '
                else:
                    res += f'+ {coef:.2e} {var} '
            else:
                if coef<0:
                    res += f'- {-coef:.2f} {var} '
                else:
                    res += f'+ {coef:.2f} {var} '
            i += 1
        if self.has_constant:
            res = f'{self.dep_var} = ' + res
        else:
            res = f'{self.dep_var} = ' + res[1:]
        return res.strip()

    def table(self):
        return self.model.summary().as_text()

    def forecast(self, sample:Sample):
        if self.formula != '':
            data = sample.data.select_variables([v for v in sample.data.variables() if v in self.formula]).select_index(sample.index.copy())
            indep_vars = self.indep_vars.copy()
            if self.has_constant:
                indep_vars = ['1'] +indep_vars
            if indep_vars[0] in data.variables():
                indep_data = data.select_variables(indep_vars[0])
            else:
                indep_data = Formula(indep_vars[0]).calculate(sample.data)
            for indep in indep_vars[1:]:   
                if indep in data.variables():
                    indep_data.add_data(data.select_variables(indep))
                else:
                    try:
                        indep_data.add_data(Formula(indep).calculate(sample.data))
                    except:
                        raise ValueError(f"'{indep}' is not in sample.")
        else:
            data = sample.data.select_variables(self.indep_vars).select_index(sample.index.copy())
            indep_vars = self.indep_vars.copy()
            if self.has_constant:
                indep_vars = ['1'] +indep_vars
                indep_data = Data(values={'1':{i:1 for i in sample.index}})
            if indep_vars[0] in data.variables():
                indep_data = data.select_variables(indep_vars[0])
            elif indep_vars[0] != '1':
                raise ValueError(f"'{indep_vars[0]}' is not in sample.")
            for indep in indep_vars[1:]:   
                if indep in data.variables():
                    indep_data.add_data(data.select_variables(indep))
                else:
                    raise ValueError(f"'{indep}' is not in sample.")

        x_arr = indep_data.to_numpy()
        y_f = self.model.predict(x_arr)
        res = Data(sample.data.type, values={f'{self.dep_var}_f':{}})
        for j, i in enumerate(indep_data.index()):
            res.values[f'{self.dep_var}_f'][i] = y_f[j]
        return res

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print('Results were saved successfully')

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'rb') as f:
            eq = pickle.load(f)
        return eq
