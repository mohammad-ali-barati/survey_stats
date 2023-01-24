import pickle, time
import numpy as np
import scipy.stats as scipy_stats
from survey_stats.data_process import Data, Sample
from survey_stats.basic_model import Variable, Variables, Formula, Formulas
from survey_stats.functions import number_of_digits, seconds_to_days_hms

class Model:
    def __init__(self, dep_var:str, formula:str):
        self.dep_var = dep_var
        self.formula = formula.replace(' ','')

    def __str__(self):
        res = ''
        for i, formula in enumerate(Formula(self.formula).split().formulas):
            if formula == '1':
                res += f' + c({i})'
            else:
                res += f' + c({i})*{formula}'
        return self.dep_var + ' = ' + res[3:]

    def __table(self, indep_names, coefs, cov_var_coefs, r2, r2adj, df, obs):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            res = f'Method: Ordinary Least Square (OLS)\n'
            res += f'Number of observations: {obs}\n'
            res += f'Degrees of freedom: {df}\n'
            res += f'R2: {r2:.4f}\n'
            res += f'Adjusted R2: {r2adj:.4f}\n'
            res += f'\nDependent variable: {self.dep_var}'
            len_var = int(max([len(var) for var in indep_names]))
            len_coefs = int(max([number_of_digits(c)+5 for c in coefs]))
            len_sd = 5
            for i in range(len(cov_var_coefs)):
                try:
                    if number_of_digits(cov_var_coefs[i][i]**0.5) + 5 > len_sd:
                        len_sd = number_of_digits(cov_var_coefs[i][i]**0.5) + 5
                except:
                    pass
            # len_sd = int(max([number_of_digits(cov_var_coefs[i][i]**0.5) + 5
            #                     for i in range(len(cov_var_coefs))]))
            len_t = 5
            for i in range(len(cov_var_coefs)):
                try:
                    if number_of_digits(coefs[i]/cov_var_coefs[i][i]**0.5) + 5 > len_t:
                        len_t = number_of_digits(
                            coefs[i]/cov_var_coefs[i][i]**0.5) + 5
                except:
                    pass
            # len_t = int(max([number_of_digits(coefs[i]/cov_var_coefs[i][i]**0.5)+5
            #                     for i in range(len(cov_var_coefs))]))
            len_p_value = 5
            for i in range(len(cov_var_coefs)):
                try:
                    if (1 - scipy_stats.t.cdf(abs(coefs[i]/cov_var_coefs[i][i]**0.5), df))*2 > len_p_value:
                        len_p_value = (
                            1 - scipy_stats.t.cdf(abs(coefs[i]/cov_var_coefs[i][i]**0.5), df))*2
                except:
                    pass
            # len_p_value = int(max([(1 - scipy_stats.t.cdf(abs(coefs[i]/cov_var_coefs[i][i]**0.5), df))*2
            #                    for i in range(len(cov_var_coefs))]))
            # add margin to columns equals to 5 left and right.
            len_var, len_coefs, len_sd, len_t, len_p_value = [
                i+10 for i in [len_var, len_coefs, len_sd, len_t, len_p_value]]
            len_total = len_var + len_coefs + len_sd + len_t + len_p_value + 4
            res += '\n'
            res += 'Results of Linear Regression\n'
            res += ' ' + '-'*len_total + ' \n'
            res += '|' + 'Variables'.center(len_var) + '|' + 'Coefs.'.center(len_coefs) + '|' + 'std.'.center(
                len_sd) + '|' + 't'.center(len_t) + '|' + 'p-value'.center(len_p_value) + '|\n'
            res += ' ' + '-'*len_total + ' \n'
            for i, var in enumerate(indep_names):
                # i = indep_names.index(var)
                # for i in range(len(cov_var_coefs)):
                sd = float(cov_var_coefs[i][i]**0.5)
                t = float(coefs[i] / sd)
                p_value = (1 - scipy_stats.t.cdf(abs(t), df))*2
                c = float(coefs[i])
                res += '|' + str(var).center(len_var) + '|' + f'{c:.4f}'.center(len_coefs) + '|' + f'{sd:.4f}'.center(
                    len_sd) + '|' + f'{t:.4f}'.center(len_t) + '|' + f'{p_value:.4f}'.center(len_p_value) + '|\n'
            res += ' ' + '-'*len_total + ' \n'
            res += '\n'
            return res

    def estimate(self, sample:Sample, print_equation:bool = True):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            warnings.filterwarnings("ignore", category=RuntimeWarning) 
            if self.formula != '':
                indep_num = Formula(self.formula).split().calculate_all(sample.get_data(), skip_collinear=True)
                indep_names = indep_num.variables()
                dep_num = Formula(self.dep_var).calculate(sample.get_data())
                data = Data(sample.data.type, {})
                data.add_data(indep_num)
                data.add_data(dep_num)
                if sample.weights != '1':
                    w_num = sample.data.select_variables([sample.weights])
                    data.dropna()
                    w_num = data.select_variables([sample.weights])
                    dep_num = data.select_variables([self.dep_var])
                    indep_num = data.select_variables(indep_names)
                else:
                    data.dropna()
                    dep_num = data.select_variables([self.dep_var])
                    indep_num = data.select_variables(indep_names)
                obs = len(data)
                if len(dep_num)>len(indep_num.variables()):
                    x_arr = indep_num.to_numpy()
                    y_arr = dep_num.to_numpy()
                    if sample.weights == '1':
                        x_x = np.dot(x_arr.T, x_arr)
                        if np.linalg.det(x_x) != 0:
                            x_x_inv = np.linalg.inv(x_x)
                            x_y = np.dot(x_arr.T, y_arr)
                            # coeficient vector
                            indep_coefs = np.dot(x_x_inv, x_y)
                            
                            # ANOVA
                            tss = float(np.dot(y_arr.T,y_arr)-(np.sum(y_arr)**2)/len(y_arr))
                            df_total = y_arr.shape[0] - 1
                            mst = tss / df_total

                            yf = np.dot(x_arr, indep_coefs)
                            rss = float(np.dot(yf.T,yf)-(np.sum(yf)**2)/len(yf))
                            df_reg = x_arr.shape[1] - 1
                            # if df_reg==0:
                            #     print(indep_coefs)
                            # msr = rss/df_reg

                            ess = tss - rss
                            df_err = df_total - df_reg
                            mse = float(ess / df_err)
                            cov_var_coefs = np.dot(mse, x_x_inv)
                            # r2 and adjusted r2
                            r2 = float(rss/tss)
                            df = df_total-df_reg-1
                            r2adj = 1-(1-r2)*df_total/df
                        else:
                            raise ValueError(f"Error! Near-singular matrix!")
                    else:
                        w_list = [w for _,w in w_num.values[sample.weights].items()]
                        w = np.diag(w_list)
                        x_w_x = np.dot(np.dot(x_arr.T, w),x_arr)
                        if np.linalg.det(x_w_x) != 0:
                            x_w_x_inv = np.linalg.inv(x_w_x)
                            x_w_y = np.dot(np.dot(x_arr.T,w), y_arr)
                            # coeficient vector
                            indep_coefs = np.dot(x_w_x_inv, x_w_y)
                            yf = np.dot(x_arr, indep_coefs)
                            # ANOVA
                            n = len(y_arr)
                            ys = 0
                            ys2 = 0
                            yfs = 0
                            yfs2 = 0
                            n = 0
                            j = 0
                            for i in range(len(y_arr)):
                                ys += y_arr[j] * w_list[i]
                                ys2 += (y_arr[j] ** 2) * w_list[i]
                                yfs += yf[j] * w_list[i]
                                yfs2 += (yf[j] ** 2) * w_list[i]
                                n += w_list[i]
                                j += 1

                            tss = float(ys2 - (ys ** 2)/n)
                            df_total = y_arr.shape[0] - 1

                            rss = float(yfs2 - (yfs ** 2)/n)
                            df_reg = x_arr.shape[1] - 1

                            ess = tss - rss
                            df_err = df_total - df_reg
                            mse = float(ess / df_err)

                            cov_var_coefs = np.dot(mse, x_w_x_inv)
                            # r2 and adjusted r2
                            r2 = rss/tss
                            df = df_total-df_reg-1
                            r2adj = 1-(1-r2)*df_total/df
                        else:
                            raise ValueError(f"Error! Near-singular matrix!")
                    table = self.__table(indep_names, indep_coefs, cov_var_coefs, r2, r2adj, df, obs)
                    if print_equation:
                        print(table)
                    indep_coefs = [i[0] for i in indep_coefs]
                    return Equation(self.dep_var, indep_names, indep_coefs, cov_var_coefs, df, mse, r2, r2adj, table, sample)
                else:
                    raise ValueError(f"Error! number of observation ({len(dep_num)}) is not greater then number of parametes ({len(indep_num.variables())}).")
            else:
                raise ValueError(f"Error! formula is empty!")

    @staticmethod
    def __observation(sample:Sample, formulas:list[str]):
        sts, ens=[],[]
        for var in formulas:
            indices = [i for i, value in sample.data.values[var].items() if not np.isnan(value) and i in sample.index]
            sts.append(indices[0])
            ens.append(indices[-1])
        sts.sort()
        ens.sort()
        st, en = sts[-1], ens[0]
        return sample.index.index(en)-sample.index.index(st)+1

    progress, start_time, left_time = 0, time.perf_counter(), time.perf_counter()
    @staticmethod
    def __estimate_skip_collinear(sample:Sample, dep_var, vars:list[str], cors:dict, print_equation:bool = True, min_df:int=10, print_progress:bool=True):
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
                if Model.__observation(sample, vars)>=len(vars)+min_df:
                    f='+'.join(vars)
                    eq =  Model(dep_var, f).estimate(sample, print_equation)
                else:
                    raise
                if print_progress:
                    print()
                return eq
            except:
                # remove most collinear variable, this don't complete
                var_min, cor_min='',1
                for var in vars:
                    if cors[var].imag ==0:
                        if cors[var].real<cor_min:
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
            cors[var] = Variables([Variable(self.dep_var), Variable(var)]).stats.correlation(sample).values[self.dep_var][var]
        vars.remove(self.dep_var)
        Model.progress, Model.start_time, Model.left_time = 0, time.perf_counter(), time.perf_counter()
        return Model.__estimate_skip_collinear(sample, self.dep_var, vars, cors, print_equation, min_df, print_progress)

    @staticmethod
    def __most_significant(dep_var:str, formula:str, sample:Sample, min_significant = 1, print_equation = True):
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
                        p_value = (1- scipy_stats.t.cdf(t_student, eq.df))*2
                        p_values.append(p_value)
                    if p_values == []:
                        break
                    removes = 1
                except:
                    removes += 1
                if p_values != []:
                    if (np.isnan(max(p_values)) or max(p_values) >= min_significant):
                        # decrease formula
                        t_students = [1 if np.isnan(t) else t for t in t_students]
                        formula_new = ''
                        removed = False
                        for i, var in enumerate(eq.indep_vars):
                            if (not t_students[i] in sorted(t_students)[:removes]) or (t_students[i]==1 and removed):
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
                raise ValueError(f"Error! there isn't any model with minimum significant of {min_significant}.")

    def estimate_most_significant(self, sample:Sample, min_significant = 1, print_equation = True):
        eq = Model.__most_significant(self.dep_var, self.formula, sample, min_significant, print_equation)
        return eq
        
class Equation:
    def __init__(self, dep_var:str, indep_vars:list[str], indep_coefs:list[float], 
                    cov_var_coefs:list[list], df:int, mse:float, r2:float, r2adj:float, 
                    table:str, sample:Sample) -> None:
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.indep_coefs = indep_coefs
        self.cov_var_coefs = cov_var_coefs
        self.df =df
        self.mse = mse
        self.r2 = r2
        self.r2adj = r2adj
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
            if sd == 0:
                t = np.nan
                p = np.nan
            else:
                t = c / sd
                p = (1- scipy_stats.t.cdf(abs(t),self.df))*2
            len_indep_var = int(max([number_of_digits(x) for x in [c,sd,t,p]]))
            if var == '1':
                if c > 0:
                    eq += ' + ' + f'{c:.4f}'.center(len_indep_var)
                elif c < 0:
                    eq += ' - ' + f'{-c:.4f}'.center(len_indep_var)
            else:
                if c > 0:
                    eq += ' + ' + f'{c:.4f}*{var}'.center(len_indep_var)
                elif c < 0:
                    eq += ' - ' + f'{-c:.4f}*{var}'.center(len_indep_var)
            sds += '   ' + f'({sd:.4f})'.ljust(len_indep_var)
            ts += '   ' + f'({t:.4f})'.ljust(len_indep_var)
            ps += '   ' + f'({p:.4f})'.ljust(len_indep_var)
        eq = self.dep_var.center(len_dep_var) + ' = ' + eq[1:]
        sds = '(std.)'.center(len_dep_var) + sds
        ts = '(t)'.center(len_dep_var) + ts
        ps = '(p-value)'.center(len_dep_var) + ps
        return f'{eq}\n{sds}\n{ts}\n{ps}'

    def forecast(self, sample:Sample)->Data:
        indep_num = Formulas(self.indep_vars).calculate_all(sample.data)
        if self.dep_var in self.sample.data.variables():
            dep_num = sample.data
        else:
            dep_num = Formula(self.dep_var).calculate(sample.data)
        forecast_data = Data(sample.data.type, {})
        forecast_data.values[self.dep_var + '_f'] = {}
        for i in sample.data.index():
            if not np.isnan(dep_num.values[self.dep_var][i]) and not i in sample.index:
                forecast_data.values[self.dep_var + '_f'][i] = dep_num.values[self.dep_var][i]
            elif i in sample.index:
                yf = 0
                for j,var in enumerate(self.indep_vars):
                    yf += indep_num.values[var][i] * float(self.indep_coefs[j])
                forecast_data.values[self.dep_var + '_f'][i] = yf
        return forecast_data.select_index(sample.index)
        

    def goodness_of_fit(self, sample:Sample=None, print_equation:bool=True):
        if sample == None:
            sample = self.sample
        data = sample.get_data().select_variables(self.dep_var)
        data.add_data(self.forecast(sample))
        data.add_data(Formula(sample.weights).calculate(sample.get_data()))

        # ANOVA
        ys, ys2, yfs, yfs2, ws, n = 0,0,0,0,0,0
        for i in sample.index:
            y = data.values[self.dep_var][i]
            yf = data.values[self.dep_var + '_f'][i]
            w = data.values[sample.weights][i]
            if not np.isnan(yf) and not np.isnan(y):
                ys += y * w
                ys2 += (y ** 2) * w
                yfs += yf * w
                yfs2 += (yf ** 2) * w
                ws += w
                n += 1

        tss = ys2 - (ys ** 2)/n
        df_total = n - 1

        rss = yfs2 - (yfs ** 2)/n
        df_reg = len(self.indep_coefs) - 1
        msr = rss/df_reg

        ess = tss - rss
        df_err = df_total - df_reg
        mse = ess / df_err

        # r2 and adjusted r2
        df = df_total-df_reg-1
        

        res = {}
        res['r2'] = rss/tss
        res['r2adj'] = 1-(1-res['r2'])*df_total/df
        res['f'] = msr / mse
        res['p-value'] = 1 - scipy_stats.f.cdf(res['f'], df_reg, df_err)

        len_title = len('regression')
        len_ss = max(len('sum of squares'), len(
            f'{rss:.4f}'), len(f'{ess:.4f}'), len(f'{tss:.4f}'))
        len_df = max(len('df.'), len(str(df_reg)), len(str(df_err)), len(str(df_total)))
        len_ms = max(len('mean of squares'), len(
            f'{msr:.4f}'), len(f'{mse:.4f}'))
        len_f = max(len(f"F = {res['f']:.4f}"), len(f"p-value = {res['p-value']:.4f}"))
        len_total = len_title + len_ss + len_df + len_ms + len_f + 4
        anova = ' ' + 'ANOVA'.center(len_total) + ' \n'
        anova += ' ' + '-'*len_total + ' \n'
        anova += '|' + ''.center(len_title) + '|' + 'sum of squares'.center(len_ss) + '|' \
                     + 'df.'.center(len_df) + '|' + 'mean of squares'.center(len_ms) + '|' \
                     + ''.center(len_f) + '|\n'
        anova += ' ' + '-'*len_total + ' \n'
        anova += '|' + 'regression'.center(len_title) + '|' + f'{rss:.4f}'.center(len_ss) + '|' \
                     + str(df_reg).center(len_df) + '|' + f'{msr:.4f}'.center(len_ms) + '|' \
                     + f"F = {res['f']:.4f}".center(len_f) + '|\n'
        anova += '|' + 'error'.center(len_title) + '|' + f'{ess:.4f}'.center(len_ss) + '|' \
                     + str(df_err).center(len_df) + '|' + f'{mse:.4f}'.center(len_ms) + '|' \
                     + f"p-value = {res['p-value']:.4f}".center(len_f) + '|\n'
        anova += ' ' + '-'*len_total + ' \n'
        anova += '|' + 'total'.center(len_title) + '|' + f'{tss:.4f}'.center(len_ss) + '|' \
                     + str(df_total).center(len_df) + '|' + ''.center(len_ms) + '|' \
                     + ''.center(len_f) + '|\n'
        anova += ' ' + '-'*len_total + ' \n'
        res['anova'] = anova
        if print_equation:
            print(anova)
            print('R2 = ', res['r2'])
            print('Adjusted R2 = ', res['r2adj'])
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
    
    """
    condition = {0:0, 3:1, ...}
    H0: coef[0] = 0
    H0: coef[3] = 1
    ...
    """
    def wald_test(self, conditions:dict):
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
        V = np.array(self.cov_var_coefs)
        A = np.dot(R,coefs) - r
        B = np.linalg.inv(np.dot(np.dot(R, V/n),np.transpose(R)))
        W = np.dot(np.dot(np.transpose(A), B), A)
        p_value = 1-scipy_stats.chi2.cdf(W[0], n)
        return {'W':W[0][0], 'p_value':p_value[0]}