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
            elif sample.weights in data.variables():
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
            elif sample.weights in data.variables():
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
                regr = LogisticRegression(solver='liblinear', random_state=0, fit_intercept=self.has_constant)
                regr.fit(x_arr, y_arr)
                if self.has_constant:
                    indep_coefs = [regr.intercept_[0]] + list(regr.coef_[0])
                else:
                    indep_coefs = list(regr.coef_[0])
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        else:
            w = np.array([w for _,w in w_num.values[sample.weights].items()])
            try:
                regr = LogisticRegression(solver='liblinear', random_state=0, fit_intercept=self.has_constant)
                regr.fit(x_arr, y_arr, w)
                if self.has_constant:
                    indep_coefs = [regr.intercept_[0]] + list(regr.coef_[0])
                else:
                    indep_coefs = list(regr.coef_[0])
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        eq = Equation(self.dep_var, self.formula, self.indep_vars, self.has_constant, indep_coefs, regr,
                                                    x_arr, y_arr,w)
        if print_progress:
            if max_lenght==-1 or max_lenght<3:
                print(' '*indent+str(eq))
            else:
                print(' '*indent+str(eq)[:max_lenght-3]+'...')
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
        
        def confusion_matrix(self, to_text_True_Positive:bool=False,
                                    to_text_actual_predicted:bool=True):
            y_pred = self.model.predict(self.x_arr)
            cm = confusion_matrix(self.y_arr, y_pred)
            if not to_text_True_Positive and not to_text_actual_predicted:
                return cm
            elif to_text_True_Positive:
                cols = max(len('Positive'), *[len(f'{x:,d}') for r in cm for x in r])+6
                res = 'confusion_matrix'.center(3*cols+4) + '\n'
                res += ' '+'-'*(3*cols+2) + '\n'
                res += '|' + ''.center(cols) + '|' + 'Negative'.center(cols) + '|' + 'Positive'.center(cols) + '|\n'
                res += ' '+'-'*(3*cols+2) + '\n'
                res += '|' + 'True'.center(cols) + '|' + f'{cm[0][0]:,d}'.center(cols) + '|' + f'{cm[1][1]:,d}'.center(cols) + '|\n'
                res += '|' + 'False'.center(cols) + '|' + f'{cm[1][0]:,d}'.center(cols) + '|' + f'{cm[0][1]:,d}'.center(cols) + '|\n'
                res += ' '+'-'*(3*cols+2) + '\n'
                res += 'True = correctly predicted\n'
                res += 'False = wrongly predicted\n'
                res += 'Positive = predicted one\n'
                res += 'Negative = predicted zero\n'
                return res
            else:
                cols = max(len('zero'), *[len(f'{x:,d}') for r in cm for x in r])+6
                res = 'confusion_matrix'.center(3*cols+4) + '\n'
                res += ' '+'-'*(3*cols+2) + '\n'
                res += '|' + ''.center(cols) + '|' + 'Predicted'.center(2*cols+1) + '|\n'
                res += '|' + 'Actual'.center(cols) + '|' + 'Zero'.center(cols) + '|' + 'One'.center(cols) + '|\n'
                res += ' '+'-'*(3*cols+2) + '\n'
                res += '|' + 'Zero'.center(cols) + '|' + f'{cm[0][0]:,d}'.center(cols) + '|' + f'{cm[0][1]:,d}'.center(cols) + '|\n'
                res += '|' + 'One'.center(cols) + '|' + f'{cm[1][0]:,d}'.center(cols) + '|' + f'{cm[1][1]:,d}'.center(cols) + '|\n'
                res += ' '+'-'*(3*cols+2) + '\n'
                return res
      
        def classification_report(self):
            y_pred = self.model.predict(self.x_arr)
            return classification_report(self.y_arr, y_pred)

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
        y_f = self.model.predict_proba(x_arr).T[1]
        res = Data(sample.data.type, values={f'{self.dep_var}_f':{}})
        for j, i in enumerate(indep_data.index()):
            res.values[f'{self.dep_var}_f'][i] = y_f[j]
        return res
