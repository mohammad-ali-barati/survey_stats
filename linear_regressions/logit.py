import pickle
import time
import scipy.stats as scipy_stats
import numpy as np
from survey_stats.basic_model import Formula, Formulas, Variable_Types, Variable, Variables
from survey_stats.data_process import Data, Sample
from survey_stats.functions import number_of_digits, seconds_to_days_hms



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
    
    motors = ['statsmodels', 'sklearn']
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
            indep_num.drop([dep_name])
            y = dep_num.to_numpy()
            x = indep_num.to_numpy()
            import warnings
            if motor == 'statsmodels':
                import statsmodels.api as sm
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model = sm.Logit(y ,x)
                    model.exog_names[:] = indep_names
                    res = model.fit(disp=0, skip_hessian= 0, method_kwargs={'warn_convergence': False})
                indep_coefs = res.params
                cov_var_coefs = res.cov_params()
                table = res.summary()
            elif motor == 'sklearn':
                from sklearn.linear_model import LogisticRegression
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    fit = LogisticRegression(random_state=0, fit_intercept=True).fit(x, y)
                indep_coefs = fit.coef_[0]
                predProbs = fit.predict_proba(x)
                X_design = np.hstack([np.ones((x.shape[0], 1)), x])
                V = np.diagflat(np.product(predProbs, axis=1))
                cov_var_coefs = np.linalg.inv(
                    np.dot(np.dot(X_design.T, V), X_design))
                table = ''
            if print_equation:
                print(table)
            return Equation(dep_name, indep_names, indep_coefs, cov_var_coefs, table, sample)

    progress, start_time, left_time = 0, time.perf_counter(), time.perf_counter()
    @staticmethod
    def __estimate_skip_collinear(sample: Sample, dep_var, vars: list[str], cors: dict, print_equation: bool = True, min_df: int = 10, print_progress: bool = True):
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
                # remove most collinear variable, this don't complete
                var_min, cor_min = '', 1
                for var in vars:
                    if cors[var].imag == 0:
                        if cors[var].real < cor_min:
                            cor_min = cors[var].real
                            var_min = var
                vars.remove(var_min)
                return Model.__estimate_skip_collinear(sample, dep_var, vars, cors, print_equation, min_df, print_progress)

    def estimate_skip_collinear(self, sample: Sample, print_equation: bool = True, min_df: int = 10, print_progress: bool = True):
        vars = Formula(self.formula).split().formulas
        # print(vars)
        vars.append(self.dep_var)
        # print(vars)
        data = Formulas(vars).calculate_all(sample.data)
        # print(data)
        sample = Sample(data, sample.index)
        cors = {}
        for var in vars:
            cors[var] = Variables([Variable(self.dep_var), Variable(
                var)]).stats.correlation(sample).values[self.dep_var][var]
        vars.remove(self.dep_var)
        Model.progress, Model.start_time, Model.left_time = 0, time.perf_counter(), time.perf_counter()
        return Model.__estimate_skip_collinear(sample, self.dep_var, vars, cors, print_equation, min_df, print_progress)

    @staticmethod
    def __most_significant(dep_var: str, formula: str, sample: Sample, min_significant=1, print_equation=True):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            removes, k = 1, 0
            while k < 100:
                k += 1
                p_values, t_students = [], []
                try:
                    eq = Model(dep_var, formula).estimate(sample, False)
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
        return Model.__most_significant(self.dep_var, self.formula, sample, min_significant, print_equation)

        

class Equation:
    def __init__(self, dep_var:str, indep_vars:list[str], indep_coefs:np.ndarray, 
                    cov_var_coefs:np.ndarray, table:str, sample:Sample) -> None:
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.indep_coefs = indep_coefs
        self.cov_var_coefs = cov_var_coefs
        self.table = table
        self.sample = sample
    
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
                    print(c, len_indep_var)
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
        indep_num = Formulas(self.indep_vars).calculate_all(sample.get_data(), skip_collinear=True)
        indep_num.dropna()
        x_arr = indep_num.to_numpy()
        
        yf = 1/(1+np.exp(-np.dot(x_arr, self.indep_coefs)))

        res = Data(sample.data.type, {self.dep_var + '_f':{}})
        j = 0
        for i in sample.index:
            if i in indep_num.index():
                res.values[self.dep_var + '_f'][i] = yf[j]
                j += 1
            else:
                res.values[self.dep_var + '_f'][i] = np.nan
        
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
