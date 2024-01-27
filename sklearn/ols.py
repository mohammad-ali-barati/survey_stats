import pickle, time
import numpy as np
import scipy.stats as scipy_stats
from survey_stats.data_process import Data, Sample
from survey_stats.basic_model import Formula
from survey_stats.functions import subsets
from sklearn.linear_model import LinearRegression
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
                regr = LinearRegression()
                regr.fit(x_arr, y_arr)
                indep_coefs = [regr.intercept_[0]] + list(regr.coef_[0][1:])
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        else:
            w = np.array([w for _,w in w_num.values[sample.weights].items()])
            try:
                regr = LinearRegression()
                regr.fit(x_arr, y_arr, w)
                indep_coefs = [regr.intercept_[0]] + list(regr.coef_[0][1:])
            except Exception as e:
                raise ValueError(f"Error! Near-singular matrix! {e}")
        eq = Equation(self.dep_var, self.formula, self.indep_vars, self.has_constant, indep_coefs,
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

    def estimate_most_significant(self, sample:Sample, min_significant = 1,
                                  print_progress = True, indent:int=0,subindent:int=5, max_lenght:int=-1):
        if print_progress:
            print(' '*indent+'estimating most significant models of '+self.dep_var)
        if self.formula != '':
            indep_vars = self.formula.split('+')
            while len(indep_vars)>0:
                eq = Model(self.dep_var, '+'.join(indep_vars)).estimate(sample, False, indent+subindent, max_lenght)
                indep_vars = ['1'] + eq.indep_vars if eq.has_constant else eq.indep_vars
                p_values = eq.params.p_values()
                var_droped, prob_droped = sorted([(indep_vars[i] , p) for i,p in enumerate(p_values)],
                                  key=lambda x: x[1], reverse=True)[0]
                if prob_droped<=min_significant:
                    if print_progress:
                        if max_lenght==-1 or max_lenght<3:
                            print(' '*(indent+subindent)+str(eq))
                        else:
                            print(' '*(indent+subindent)+str(eq)[:max_lenght-3]+'...')
                    break
                if print_progress:
                    print(' '*(indent+subindent)+f'droped: {var_droped}. number of parameters: {len(indep_vars)}.')
                indep_vars = [v for v in indep_vars if v!=var_droped]
            else:
                raise ValueError(f"Error! there isn't any model with minimum significant of {min_significant}.")
            return eq
        elif self.indep_vars != []:
            indep_vars = ['1'] + self.indep_vars.copy() if self.has_constant else self.indep_vars.copy()
            while len(indep_vars)>0:
                has_constant = False
                if '1' in indep_vars:
                    has_constant = True
                    indep_vars.remove('1')
                eq = Model(self.dep_var, indep_vars=indep_vars,
                           has_constant=has_constant).estimate(sample, False, indent+subindent, max_lenght)
                indep_vars = ['1'] + eq.indep_vars if eq.has_constant else eq.indep_vars
                p_values = eq.params.p_values()
                var_droped, prob_droped = sorted([(indep_vars[i] , p) for i,p in enumerate(p_values)],
                                  key=lambda x: x[1], reverse=True)[0]
                if prob_droped<=min_significant:
                    if print_progress:
                        if max_lenght==-1 or max_lenght<3:
                            print(' '*(indent+subindent)+str(eq))
                        else:
                            print(' '*(indent+subindent)+str(eq)[:max_lenght-3]+'...')
                    break
                if print_progress:
                    print(' '*(indent+subindent)+f'droped: {var_droped}. number of parameters: {len(indep_vars)}.')
                indep_vars = [v for v in indep_vars if v!=var_droped]
            else:
                raise ValueError(f"Error! there isn't any model with minimum significant of {min_significant}.")
            return eq
        else:
            raise ValueError(f"Error! formula or indep_vars are not defined!")

class Equation:
    def __init__(self, dep_var:str, formula:str, indep_vars:list[str], has_constant:bool, indep_coefs:list[float],
                    x_arr:np.ndarray=None, y_arr:np.ndarray=None,
                    w:np.ndarray=None) -> None:
        self.dep_var = dep_var
        self.formula = formula
        self.indep_vars = indep_vars
        self.has_constant = has_constant
        self.indep_coefs = indep_coefs
        self.x_arr = x_arr
        self.y_arr =y_arr
        self.w = w
        self.params =self.params(self.indep_coefs, x_arr, y_arr, w)

    class params:
        def __init__(self, indep_coefs:list[float],
                x_arr:np.ndarray=None, y_arr:np.ndarray=None,
                w:np.ndarray=None) -> None:
            self.indep_coefs = indep_coefs
            self.x_arr = x_arr
            self.y_arr =y_arr
            self.w = w

        def obs(self):
            return self.y_arr.shape[0]

        def tss(self):
            if self.w == 1:
                return float(np.dot(self.y_arr.T,self.y_arr)-(np.sum(self.y_arr)**2)/len(self.y_arr))
            else:
                n = len(self.y_arr)
                ys, ys2, n = 0, 0, 0
                for i, y in enumerate(self.y_arr):
                    ys += y * self.w[i]
                    ys2 += (y ** 2) * self.w[i]
                    n += self.w[i]
                return float(ys2 - (ys ** 2)/n)
        
        def df_total(self):
            return self.y_arr.shape[0] - 1
        
        def mst(self):
            return self.tss()/self.df_total()

        def rss(self):
            yf = np.dot(self.x_arr, self.indep_coefs)
            if self.w == 1:
                return float(np.dot(yf.T,yf)-(np.sum(yf)**2)/len(yf))
            else:
                n = len(self.y_arr)
                yfs, yfs2, n = 0, 0, 0
                for i, y in enumerate(self.y_arr):
                    yfs += yf[i] * self.w[i]
                    yfs2 += (yf[i] ** 2) * self.w[i]
                    n += self.w[i]
                return float(yfs2 - (yfs ** 2)/n)

        def df_reg(self):
            return self.x_arr.shape[1] - 1

        def msr(self):
            return self.rss()/self.df_reg()
  
        def ess(self):
            return self.tss() - self.rss()
        
        def df_err(self):
            return self.df_total() - self.df_reg()
        
        def mse(self):
            return float(self.ess() / self.df_err())
        
        def cov_var_coefs(self):
            x_x_inv = np.linalg.inv(np.dot(self.x_arr.T, self.x_arr))
            return np.dot(self.mse(), x_x_inv)
        
        def r2(self):
            return float(self.rss()/self.tss())

        def df(self):
            return self.df_total()-self.df_reg()-1
        
        def r2adj(self):
            return 1-(1-self.r2())*self.df_total()/self.df()

        def p_values(self):
            return [(1 - scipy_stats.t.cdf(abs(coefi / self.cov_var_coefs()[i][i]**0.5), self.df()))*2
                            for i, coefi in enumerate(self.indep_coefs)]

        def f(self):
            return self.msr()/self.mse()

        def f_prob(self):
            return 1 - scipy_stats.f.cdf(self.f(), self.df_reg(), self.df_err())

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

    def anova(self):
        def format_floats(x:float):
            if abs(x)<.1 or abs(x)>1000:
                x = f'{x:.2e}'
            else:
                x = f'{x:.2f}'
            return x
        def format_ints(x:int):
            return f'{x:,d}'
        len_title = len('regression')
        # ss
        rss, ess = self.params.rss(), self.params.ess()
        tss = rss + ess
        # df
        df_reg, df_err= self.params.df_reg(), self.params.df_err()
        df_total = df_reg + df_err
        # ms
        msr, mse = rss/df_reg, ess/df_err
        # f
        f = msr/mse
        f_prob = 1 - scipy_stats.f.cdf(f, df_reg, df_err)

        # to string format
        rss, ess, tss = format_floats(rss), format_floats(ess), format_floats(tss)
        df_reg, df_err, df_total = format_ints(df_reg), format_ints(df_err), format_ints(df_total)
        msr, mse = format_floats(msr), format_floats(mse)
        f, f_prob = f'f={format_floats(f)}', f'p-value={format_floats(f_prob)}'
        # length
        len_ss = max(len('sum of squares'), len(rss), len(ess), len(tss))
        len_df = max(len('df.'), len(df_reg), len(df_err), len(df_total))
        len_ms = max(len('mean of squares'), len(msr), len(mse))
        len_f = max(len(f), len(f_prob))
        len_total = 1 + len_title + 1 + len_ss + 1 + len_df + 1 + len_ms + 1 + len_f + 1

        # res
        res = ' ' + 'ANOVA'.center(len_total) + ' \n'
        res += ' ' + '-'*len_total + ' \n'
        res += '|' + ''.center(len_title) + '|' + 'sum of squares'.center(len_ss) + '|' + 'df.'.center(len_df) + '|' + 'mean of squares'.center(len_ms) + '|' + ''.center(len_f) + '|\n'
        res += ' ' + '-'*len_total + ' \n'
        res += '|' + 'regression'.center(len_title) + '|' + rss.center(len_ss) + '|' + df_reg.center(len_df) + '|' + msr.center(len_ms) + '|' + f.center(len_f) + '|\n'
        res += '|' + 'error'.center(len_title) + '|' + ess.center(len_ss) + '|' + df_err.center(len_df) + '|' + mse.center(len_ms) + '|' + f_prob.center(len_f) + '|\n'
        res += ' ' + '-'*len_total + ' \n'
        res += '|' + 'total'.center(len_title) + '|' + tss.center(len_ss) + '|' + df_total.center(len_df) + '|' + ''.center(len_ms) + '|' + ''.center(len_f) + '|\n'
        res += ' ' + '-'*len_total + ' \n'
        return res

    def table(self):
        def format_floats(x:float):
            if abs(x)<.001 or abs(x)>1000:
                x = f'{x:.4e}'
            else:
                x = f'{x:.4f}'
            return x
        def format_ints(x:int):
            return f'{x:,d}'
        res = f'Method: Ordinary Least Square (OLS)\n'
        res += f'Number of observations: {format_ints(self.params.obs())}\n'
        res += f'Degrees of freedom: {format_ints(self.params.df())}\n'
        res += f'R2: {format_floats(self.params.r2())}\n'
        res += f'Adjusted R2: {format_floats(self.params.r2adj())}\n'
        res += f'\nDependent variable: {self.dep_var}'
        len_var = max([len(var) for var in self.indep_vars])
        coefs = [format_floats(c) for c in self.indep_coefs]
        len_coefs = len(max(coefs))
        cov_vars = self.params.cov_var_coefs()
        sds = [format_floats(cov_vars[i][i]**.5) for i in range(len(cov_vars))]
        len_sd = len(max(sds))
        ts = [format_floats(self.indep_coefs[i]/cov_vars[i][i]**.5) for i in range(len(cov_vars))]
        len_t = len(max(ts))
        p_values = self.params.p_values()
        ps = [format_floats(p) for p in p_values]
        len_p_value = len(max(ps))
        
        len_var, len_coefs, len_sd, len_t, len_p_value = [
            i+5 for i in [len_var, len_coefs, len_sd, len_t, len_p_value]]
        len_total = len_var + len_coefs + len_sd + len_t + len_p_value + 4
        res += '\n'
        res += 'Results of Linear Regression\n'
        res += ' ' + '-'*len_total + ' \n'
        res += '|' + 'Variables'.center(len_var) + '|' + 'Coefs.'.center(len_coefs) + '|' + 'std.'.center(
            len_sd) + '|' + 't'.center(len_t) + '|' + 'p-value'.center(len_p_value) + '|\n'
        res += ' ' + '-'*len_total + ' \n'

        indep_names = ['1'] + self.indep_vars if self.has_constant else self.indep_vars
        for i, var in enumerate(indep_names):
            res += '|' + str(var).center(len_var) + '|' + f'{coefs[i]}'.center(len_coefs) + '|' + f'{sds[i]}'.center(
                len_sd) + '|' + f'{ts[i]}'.center(len_t) + '|' + f'{ps[i]}'.center(len_p_value) + '|\n'
        res += ' ' + '-'*len_total + ' \n'
        res += '\n'
        return res

    def save(self, file_path:str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print('Results were saved successfully')

    @classmethod
    def load(cls, file_path:str):
        with open(file_path, 'rb') as f:
            eq = pickle.load(f)
        return eq
    
    def wald_test(self, conditions:dict):
        '''
        condition = {0:0, 3:1, ...}\n
        H0: coef[0] = 0 and coef[3] = 1 and ...\n
        '''
        R = []
        n = len(self.indep_coefs)
        for cond in conditions:
            row = []
            for i in range(n):
                if cond == i:
                    row.append(1)
                else:
                    row.append(0)
            R.append(row)
        R = np.array(R)
        r = np.array([list(conditions.values())]).T
        coefs = np.array(self.indep_coefs)
        V = np.array(self.params.cov_var_coefs())
        A = np.dot(R,coefs) - r
        B = np.linalg.inv(np.dot(np.dot(R, V/n),np.transpose(R)))
        W = np.dot(np.dot(np.transpose(A), B), A)
        p_value = 1-scipy_stats.chi2.cdf(W[0], n)
        if 0.05<p_value<=.1:
            result = f'At the 90% confidence level, the coefficients are significantly different from the determined values at the same time.'
        elif 0.01<p_value<=.05:
            result = f'At the 95% confidence level, the coefficients are significantly different from the determined values at the same time.'
        elif p_value<=.01:
            result = f'At the 99% confidence level, the coefficients are significantly different from the determined values at the same time.'
        else:
            result = f'the coefficients are not significantly different from the determined values at the same time.'

        return {'wald':W[0][0], 'p_value':p_value[0], 'result': result}

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
  
        forecast_data = Data(sample.data.type, {})
        forecast_data.values[self.dep_var + '_f'] = {}
        for i in sample.index:
            forecast_data.values[self.dep_var + '_f'][i] = sum([indep_data.values[var][i] * self.indep_coefs[j]
                                                                for j, var in enumerate(self.indep_vars)])
        return forecast_data
