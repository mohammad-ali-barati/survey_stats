import pickle
import numpy as np
import scipy.stats as scipy_stats
from _models.data_process import Data, Sample
from _models.basic_model import Variable_Types, Variable, Formula, Formulas
from _models.functions import number_of_digits

class Model:
    def __init__(self, dep_var:str, formula:str):
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
    def __table(indep_names, coefs, cov_var_coefs, r2, r2adj, df):
        res = ''
        len_var = int(max([len(var) for var in indep_names]))
        len_coefs = int(max([number_of_digits(c)+5 for c in coefs]))
        len_sd = int(max([number_of_digits(cov_var_coefs[i][i]**0.5) + 5
                            for i in range(len(cov_var_coefs))]))
        len_t = int(max([number_of_digits(coefs[i]/cov_var_coefs[i][i]**0.5)+5
                            for i in range(len(cov_var_coefs))]))
        len_p_value = int(max([(1 - scipy_stats.t.cdf(abs(coefs[i]/cov_var_coefs[i][i]**0.5), df))*2
                           for i in range(len(cov_var_coefs))]))
        # add margin to columns equals to 5 left and right.
        len_var, len_coefs, len_sd, len_t, len_p_value = [
            i+10 for i in [len_var, len_coefs, len_sd, len_t, len_p_value]]
        len_total = len_var + len_coefs + len_sd + len_t + len_p_value + 4
        res = '\n'
        res += 'Results of Linear Regression\n'
        res += ' ' + '-'*len_total + ' \n'
        res += '|' + 'Variables'.center(len_var) + '|' + 'Coefs.'.center(len_coefs) + '|' + 'std.'.center(
            len_sd) + '|' + 't'.center(len_t) + '|' + 'p-value'.center(len_p_value) + '|\n'
        res += ' ' + '-'*len_total + ' \n'
        for i in range(len(cov_var_coefs)):
            sd = float(cov_var_coefs[i][i]**0.5)
            t = float(coefs[i] / sd)
            p_value = (1- scipy_stats.t.cdf(abs(t),df))*2
            c = float(coefs[i])
            res += '|' + str(indep_names[i]).center(len_var) + '|' + f'{c:.4f}'.center(len_coefs) + '|' + f'{sd:.4f}'.center(
                len_sd) + '|' + f'{t:.4f}'.center(len_t) + '|' + f'{p_value:.4f}'.center(len_p_value) + '|\n'
        res += ' ' + '-'*len_total + ' \n'
        res += f'R2: {r2:.4f}\n'
        res += f'Adjusted R2: {r2adj:.4f}\n'
        res += '\n'
        return res

    def estimate(self, sample:Sample, do_print:bool = True):
        # if Variable.from_data(sample.data, self.dep_var).type != Variable_Types.numeric:
        #     raise ValueError(f"Error! dependent variable '{self.dep_var}' must be nuemric but it is {Variable.from_data(sample.data, self.dep_var).type}.")
        if self.formula != '':
            indep_num = Formula(self.formula).split().calculate_all(sample.get_data(), skip_collinear=True)
            dep_num = Formula(self.dep_var).calculate(sample.get_data())
            data = Data(sample.data.type)
            data.add_data(indep_num)
            data.add_data(dep_num)
            if sample.weights != '1':
                w_num = sample.data.select_variables(sample.weights)
                data.add_data(w_num)
                data.dropna()
                w_num = data.select_variables([sample.weights])
                dep_num = data.select_variables([self.dep_var])
                data.drop([self.dep_var, sample.weights])
                indep_num = data
            else:
                data.dropna()
                dep_num = data.select_variables([self.dep_var])
                data.drop([self.dep_var])
                indep_num = data
            
            indep_names = indep_num.variables()
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
                    msr = rss/df_reg

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
                # print(w)
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
            table = Model.__table(indep_names, indep_coefs, cov_var_coefs, r2, r2adj, df)
            if do_print:
                print(table)
            return Equation(self.dep_var, indep_names, indep_coefs, cov_var_coefs, df, r2, r2adj, table, sample)
        else:
            raise ValueError(f"Error! formula is empty!")

    def best_estimate(self, sample:Sample, min_significant = 1, do_print = True):
        eq = self.estimate(sample, False)
        cov = eq.cov_var_coefs
        coefs = eq.indep_coefs
        j = 0
        func = ''
        nonsign = False
        n_nonsign = 0
        for s in cov:
            t = coefs[j]/s[j]**0.5
            p_value = (1- scipy_stats.t.cdf(abs(t), eq.df))*2
            if p_value <= min_significant:
                func += ' + ' + eq.indep_vars[j]
            elif nonsign:
                func += ' + ' + eq.indep_vars[j]
                n_nonsign += 1
            else:
                n_nonsign += 1
                nonsign = True
            j += 1
        if n_nonsign == 0:
            return self.estimate(sample, do_print)
        else:
            self.function = func[3:]
            return self.best_estimate(sample, min_significant, do_print)

class Equation:
    def __init__(self, dep_var:str, indep_vars:list[str], indep_coefs:list[float], 
                    cov_var_coefs:list[list], df:int, r2:float, r2adj:float, 
                    table:str, sample:Sample) -> None:
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.indep_coefs = indep_coefs
        self.cov_var_coefs = cov_var_coefs
        self.df =df
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
            t = c / sd
            p = (1- scipy_stats.t.cdf(abs(t),self.df))*2
            len_indep_var = int(max([number_of_digits(x) for x in [c,sd,t,p]]))
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
            ts += '   ' + f'({t:.4f})'.ljust(len_indep_var)
            ps += '   ' + f'({p:.4f})'.ljust(len_indep_var)
        eq = self.dep_var.center(len_dep_var) + ' = ' + eq[3:]
        sds = '(std.)'.center(len_dep_var) + sds
        ts = '(t)'.center(len_dep_var) + ts
        ps = '(p-value)'.center(len_dep_var) + ps
        return f'{eq}\n{sds}\n{ts}\n{ps}'

    def forecast(self, sample:Sample)->Data:
        indep_num = Formulas(self.indep_vars).calculate_all(sample.get_data())
        data = Data(sample.data.type)
        data.add_data(indep_num)
        if sample.weights != '1':
            w_num = sample.data.select_variables(sample.weights)
            data.add_data(w_num)
            data.dropna()
            w_num = data.select_variables([sample.weights])
            data.drop([self.dep_var, sample.weights])
            indep_num = data
        else:
            data.dropna()
            data.drop([self.dep_var])
            indep_num = data
        x_arr = indep_num.to_numpy()

        yf = np.dot(x_arr, self.indep_coefs)
        forecast_data = Data(sample.data.type, {})
        j = 0
        for i in sample.index:
            if not (self.dep_var + '_f') in forecast_data.values:
                forecast_data.values[self.dep_var + '_f'] = {}
            if i in data.index():
                forecast_data.values[self.dep_var + '_f'][i] = float(yf[j])
                j += 1
            else:
                forecast_data.values[self.dep_var + '_f'][i] = np.nan
        return forecast_data

    def goodness_of_fit(self, sample:Sample=None, do_print:bool=True):
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
        if do_print:
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