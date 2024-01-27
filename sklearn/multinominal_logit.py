import pickle, time
import numpy as np
import scipy.stats as scipy_stats
from survey_stats.data_process import Data, Sample
from survey_stats.basic_model import Formula
from survey_stats.functions import subsets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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
            elif sample.weights in sample.data.variables():
                data = sample.data.select_variables([v for v in sample.data.variables() if v in self.formula or v in self.dep_var or v == sample.weights]).select_index(sample.index.copy())
            indep_num = Formula(self.formula).split().calculate_all(data)
            indep_names = indep_num.variables()
            self.indep_vars = indep_names.copy()
            # if self.has_constant:
            #     self.indep_vars.remove('1')
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
            elif sample.weights in sample.data.variables():
                data = sample.data.select_variables([self.dep_var]+self.indep_vars+[sample.weights]).select_index(sample.index.copy())
            data.dropna()
            if sample.weights != '1':
                w_num = data.select_variables([sample.weights])
            y_arr = data.select_variables([self.dep_var]).to_numpy()
            x_arr = data.select_variables(self.indep_vars).to_numpy()
        else:
            raise ValueError(f"Error! formula or indep_vars are not defined!")

        if sample.weights == '1':
            w = 1
            try:
                regr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0, fit_intercept=self.has_constant)
                regr.fit(x_arr, y_arr)
                if self.has_constant:
                    indep_coefs = [[x] + list(regr.coef_[i]) for i, x in enumerate(regr.intercept_)]
                else:
                    indep_coefs = [list(x) for x in regr.coef_]
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        else:
            w = np.array([w for _,w in w_num.values[sample.weights].items()])
            try:
                regr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=0, fit_intercept=self.has_constant)
                regr.fit(x_arr, y_arr, w)
                if self.has_constant:
                    indep_coefs = [[x] + list(regr.coef_[i]) for i, x in enumerate(regr.intercept_)]
                else:
                    indep_coefs = [list(x) for x in regr.coef_]
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        eq = Equation(self.dep_var, self.formula, self.indep_vars, self.has_constant, indep_coefs, regr,
                                                    x_arr, y_arr,w)
        if print_progress:
            print(' '*indent+eq.to_str(max_lenght))
        return eq

    def estimate_skip_collinear(self, sample:Sample, print_progress:bool=False, indent:int=0, subindent:int=5, max_lenght:int=-1):
        if print_progress:
            print(' '*indent+'estimating model of '+self.dep_var)
        if self.formula != '':
            vars_number = len(vars:=self.formula.split('+'))
            if vars_number<=25:
                subvars = sorted(subsets(vars), key=lambda x:len(x), reverse=True)
                for i, regressors in enumerate(subvars):
                    try:
                        return Model(self.dep_var, '+'.join(regressors)).estimate(sample, print_progress, indent+subindent, max_lenght)
                    except:
                        if print_progress:
                            print(' '*(indent+subindent)+f'{i} of {len(subvars)}. number of variables: {len(regressors)}')
            else:
                subvars = sorted(subsets(vars,deep=len(vars),randomly=True), key=lambda x:len(x), reverse=True)
                for i, regressors in enumerate(subvars):
                    try:
                        return Model(self.dep_var, '+'.join(regressors)).estimate(sample, print_progress, indent+subindent, max_lenght)
                    except:
                        if print_progress:
                            print(' '*(indent+subindent)+f'{i} of {len(subvars)}. number of variables: {len(regressors)}')
        elif self.indep_vars != []:
            vars_number = self.has_constant + len(vars:=self.indep_vars)
            if vars_number<=25:
                subvars = sorted(subsets(vars), key=lambda x:len(x), reverse=True)
                for i, regressors in enumerate(subvars):
                    try:
                        return Model(self.dep_var, indep_vars=regressors, has_constant=self.has_constant).estimate(sample, print_progress, indent+subindent, max_lenght)
                    except:
                        if print_progress:
                            print(' '*(indent+subindent)+f'{i} of {len(subvars)}. number of variables: {len(regressors)}')
            else:
                subvars = sorted(subsets(vars,deep=len(vars),randomly=True), key=lambda x:len(x), reverse=True)
                for i, regressors in enumerate(subvars):
                    try:
                        return Model(self.dep_var, indep_vars=regressors, has_constant=self.has_constant).estimate(sample, print_progress, indent+subindent, max_lenght)
                    except:
                        if print_progress:
                            print(' '*(indent+subindent)+f'{i} of {len(subvars)}. number of variables: {len(regressors)}')
        else:
            raise ValueError(f"Error! formula or indep_vars are not defined!")
        
        return 

class Equation:
    def __init__(self, dep_var:str, formula:str, indep_vars:list[str], has_constant:bool, indep_coefs:list[float], model,
                    x_arr:np.ndarray=None, y_arr:np.ndarray=None,
                    w:np.ndarray=None) -> None:
        self.dep_var = dep_var
        self.formula = formula
        self.indep_vars = indep_vars
        self.has_constant = has_constant
        self.indep_coefs = indep_coefs
        self.model = model
        self.x_arr = x_arr
        self.y_arr =y_arr
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
        def score(self):
            return self.model.score(self.x_arr, self.y_arr)
        
        def confusion_matrix(self, to_text_actual_predicted:bool=True):
            y_pred = self.model.predict(self.x_arr)
            cm = confusion_matrix(self.y_arr, y_pred)
            if not to_text_actual_predicted:
                return cm
            else:
                vals = sorted(list(set([int(x) for x in self.y_arr])))
                cols = max(len('zero'), *[len(f'{x:,d}') for r in cm for x in r])+2
                res = 'confusion_matrix'.center(len(vals)*cols+4) + '\n'
                res += ' '+'-'*((len(vals)+1)*(cols+1)-1) + '\n'
                res += '|' + ''.center(cols) + '|' + 'Predicted'.center(len(vals)*(cols+1)-1) + '|\n'
                # title
                res += '|' + 'Actual'.center(cols) 
                for predicts in vals:
                    res += '|' + str(predicts).center(cols)
                res += '|\n'
                res += ' '+'-'*((len(vals)+1)*(cols+1)-1) + '\n'
                # values
                for i, actuals in enumerate(vals):
                    res += '|' + str(actuals).center(cols)
                    for j, predicts in enumerate(vals):
                        res += '|' + f'{cm[i][j]:,d}'.center(cols)
                    res += '|\n'
                res += ' '+'-'*((len(vals)+1)*(cols+1)-1) + '\n'
                return res
      
        def classification_report(self):
            y_pred = self.model.predict(self.x_arr)
            return classification_report(self.y_arr, y_pred)

    def __str__(self):
        return self.to_str()
    
    def to_str(self, max_lenght:int=-1):
        dep_vals = sorted(list({x for y in self.y_arr for x in y}))
        res = ''
        for j, coefs in enumerate(self.indep_coefs):
            row = ''
            if self.has_constant:
                coef=coefs[0]
                if abs(coef)<.1 or abs(coef)>1000:
                    row += f'{coef:.2e} '
                else:
                    row += f'{coef:.2f} '
                i = 1
            else:
                i = 0
            for var in self.indep_vars:
                coef=coefs[i]
                if abs(coef)<.1 or abs(coef)>1000:
                    if coef<0:
                        row += f'- {-coef:.2e} {var} '
                    else:
                        row += f'+ {coef:.2e} {var} '
                else:
                    if coef<0:
                        row += f'- {-coef:.2f} {var} '
                    else:
                        row += f'+ {coef:.2f} {var} '
                i += 1
            if self.has_constant:
                row = f'{self.dep_var}={dep_vals[j]} = ' + row
            else:
                row = f'{self.dep_var}={dep_vals[j]} = ' + row[1:]
            if max_lenght<3:
                res += row + '\n'
            else:
                res += row[:max_lenght-3] + '...\n'
        return res.strip()

    def save(self, file_path:str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print('Results were saved successfully')

    @classmethod
    def load(cls, file_path:str):
        with open(file_path, 'rb') as f:
            eq = pickle.load(f)
        return eq

    def forecast(self, sample:Sample)->Data:
        if self.formula != '':
            data = sample.data.select_variables([v for v in sample.data.variables() if v in self.formula]).select_index(sample.index.copy())
            indep_vars = self.indep_vars.copy()
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
            if indep_vars[0] in data.variables():
                indep_data = data.select_variables(indep_vars[0])
            elif indep_vars[0] != '1':
                raise ValueError(f"'{indep_vars[0]}' is not in sample.")
            for indep in indep_vars[1:]:   
                if indep in data.variables():
                    indep_data.add_data(data.select_variables(indep))
                else:
                    raise ValueError(f"'{indep}' is not in sample.")
        vals = sorted(list(set([int(x) for x in self.y_arr])))
        x_arr = indep_data.to_numpy()
        y_f = self.model.predict_proba(x_arr)
        res = Data(sample.data.type, values={f'{self.dep_var}={v}':{} for v in vals})
        for j, i in enumerate(indep_data.index()):
            for k, y in enumerate(y_f[j]):
                res.values[f'{self.dep_var}={vals[k]}'][i] = y
        return res
