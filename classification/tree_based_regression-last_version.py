from __future__ import annotations
import numpy as np
import scipy.stats as scipy_stats

import os
import math
import statistics
import pickle

from scipy.stats.morestats import Mean, Std_dev

from survey_stats.data_process import *
from survey_stats.functions import *
from survey_stats.basic_model import *

class Methods:
    # numeric
    mse = 'mse'
    poisson = 'poisson'
    mae = 'mae'
    p_value = 'p-value'
    # categorical
    gini = 'gini'
    entropy = 'entropy'
    misclassification = 'misclassification'

class Node:
    def __init__(self, sample:Sample, dep_var:Variable, parent:Node = None, 
                split_var:Variable=None, relation:str=None, split_value=None, 
                t:float = None, p_value:float=None) -> None:
        self.sample = sample
        self.dep_var = dep_var
        self.parent = parent
        self.split_var = split_var
        self.relation = relation
        self.split_value = split_value
        self.t = t
        self.p_value = p_value

    def depth(self):
        if self.parent != None:
            return self.parent.depth() + 1
        else:
            return 0

    def parents(self) -> list:
        if self.parent == None:
            return []
        else:
            ps = self.parent.parents()
            ps.append(self.parent)
            return ps

    def childs(self, nodes:list) -> None:
        chs= []
        for node in nodes:
            if node.parent == self:
                chs.append(node)
        return chs

    def __str__(self) -> str:
        return self.to_summary_str()

    def to_full_str(self):
        res = ''
        if self.dep_var.type == Variable_Types.numeric:
            mean = self.dep_var.stats.mean(self.sample)
            mse = self.dep_var.stats.std(self.sample)**2
            res += f'n = {len(self.sample.index)}, {self.dep_var.name}: mean = {mean:.4f}, mse = {mse:.4f}'
        elif self.dep_var.type == Variable_Types.categorical:
            stats = self.dep_var.stats.distribution(self.sample)
            dist = '{'
            for v in stats:
                dist += f"{v}:{stats[v]['count']}, "
            dist = dist[:-2] + '}'
            res += f'n = {len(self.sample.index)}, {self.dep_var.name} = {dist}'
        
        
        if self.split_var != None:
            res += f', {self.split_var.name} '
            if self.split_var.type == Variable_Types.numeric:
                if is_numeric(self.split_value):
                    res += f'{self.relation} {self.split_value:.4f}'
                else:
                    raise ValueError(f"Error! Probably {self.split_var.name} type is 'categorical', but now it is 'numeric'!")
            elif self.split_var.type == 'categorical':
                if self.relation == '==':
                    res += f'{self.relation} {self.split_value}'
                else:
                    values = [v for v in self.split_var.values(self.sample) if not v in self.split_value]
                    res += f'== {values}'
            if self.relation == '<=' or self.relation == '==':
                if self.dep_var.type == Variable_Types.numeric:
                    res += f', t = {self.t:.4f}, p-value = {self.p_value:.4f}'
                elif self.dep_var.type == Variable_Types.categorical:
                    res += f', chi2 = {self.t:.4f}, p-value = {self.p_value:.4f}'
        return res

    def to_summary_str(self):
        res = f'n = {len(self.sample.index)}'
        
        if self.split_var != None:
            res += f', {self.split_var.name} '
            if self.split_var.type == Variable_Types.numeric:
                if is_numeric(self.split_value):
                    res += f'{self.relation} {self.split_value:.4f}'
                else:
                    raise ValueError(f"Error! Probably {self.split_var.name} type is 'categorical', but now it is 'numeric'!")
            elif self.split_var.type == 'categorical':
                if self.relation == '==':
                    res += f'{self.relation} {self.split_value}'
                else:
                    values = [v for v in self.split_var.values(self.sample) if not v in self.split_value]
                    res += f'== {values}'
        return res

    # value_split for numeric=[15,26,32, ...], for categorical=[['a'], ['a','b'], ...] 
    def stats_split(self, var_split:Variable, value_split:Union[float, list], method:str = Methods.mse):
        if self.dep_var.type == Variable_Types.numeric:
            is_poisson = True
            #region left and right
            ws = 0
            s_left, s2_left, w_left, n_left = 0,0,0,0
            s_right, s2_right, w_right, n_right = 0,0,0,0
            yw_left, yw_right = [], []
            for i in self.sample.index:
                w = 1 if self.sample.weights == '1' else self.sample.data.values[self.sample.weights][i]
                y = self.sample.data.values[self.dep_var.name][i]
                ws += w
                in_left = False
                if var_split.type == Variable_Types.numeric:
                    in_left = self.sample.data.values[var_split.name][i] <= value_split
                elif var_split.type == Variable_Types.categorical:
                    for v in value_split:
                        if self.sample.data.values[var_split.name][i] == v:
                            in_left = True
                            break
                if in_left:
                    s_left += w * y
                    s2_left += w * (y * y)
                    w_left += w
                    n_left += 1
                    if method == Methods.mae:
                        if self.sample.weights != '1':
                            yw_left.append((self.sample.data.values[self.dep_var.name][i],
                                self.sample.data.values[self.sample.weights][i]))
                        else:
                            yw_left.append((self.sample.data.values[self.dep_var.name][i],
                                            1))
                else:
                    s_right += w * y
                    s2_right += w * (y * y)
                    w_right += w
                    n_right += 1
                    if method == Methods.mae:
                        if self.sample.weights != '1':
                            yw_right.append((self.sample.data.values[self.dep_var.name][i],
                                     self.sample.data.values[self.sample.weights][i]))
                        else:
                            yw_right.append((self.sample.data.values[self.dep_var.name][i],
                                             1))
            #endregion
            if n_left > 0 and n_right > 0:
                mean_left = s_left/w_left
                mean_right = s_right/w_right
                #region mse, poisson, mae
                mse, poisson, mae = np.nan, np.nan, np.nan
                if method == Methods.mse:
                    mse_left = (s2_left - s_left*s_left/w_left) / w_left   # Mean Squared Error
                    mse_right = (s2_right - s_right*s_right/w_right) / w_right # Mean Squared Error
                    mse = w_left/ws * mse_left + w_right/ws * mse_right
                elif method == Methods.poisson:
                    s_poisson_left, s_poisson_right = 0, 0
                    for i in self.sample.index:
                        if self.sample.weights == '1':
                            w = 1
                        else:
                            w = self.sample.data.values[self.sample.weights][i]
                        y = self.sample.data.values[self.dep_var.name][i]
                        in_left = False
                        if var_split.type == Variable_Types.numeric:
                            in_left = self.sample.data.values[var_split.name][i] <= value_split
                        elif var_split.type == Variable_Types.categorical:
                            for v in value_split:
                                if self.sample.data.values[var_split.name][i] == v:
                                    in_left = True
                                    break
                        if in_left:
                            is_poisson = False
                            if mean_left != 0:
                                if y/mean_left>0:
                                    is_poisson = True
                                    s_poisson_left += w * (y*math.log(y/mean_left) - y + mean_left)
                        else:
                            is_poisson = False
                            if mean_right != 0:
                                if y/mean_right>0:
                                    is_poisson = True
                                    s_poisson_right += w * (y*math.log(y/mean_right) - y - mean_right)
                    if is_poisson:
                        poisson_left = s_poisson_left/w_left        # Half Poisson Deviance
                        poisson_right = s_poisson_right/w_right
                    else:
                        poisson_left = np.nan
                        poisson_right = np.nan
                    poisson = w_left/ws * poisson_left + w_right/ws * poisson_right
                elif method == Methods.mae:
                    yw_left.sort(key=lambda row: row[0])
                    yw_right.sort(key=lambda row: row[0])
                    wt_left = sum([row[1] for row in yw_left])
                    wt_right = sum([row[1] for row in yw_right])
                    ws = 0
                    for y, w in yw_left:
                        ws += w
                        median_left = y
                        if ws/wt_left >= 0.5:
                            break
                    ws = 0
                    for y, w in yw_left:
                        ws += w
                        median_right = y
                        if ws/wt_right >= 0.5:
                            break
                    mae_left = sum([w * abs(y-median_left)
                                   for y, w in yw_left])/w_left
                    mae_right = sum([w * abs(y-median_right)
                                     for y, w in yw_right])/w_right
                    mae = w_left/ws * mae_left + w_right/ws * mae_right
                #endregion
                #region t and p-value
                if (n_left-1)*w_left/n_left>0 and (n_right-1)*w_right/n_right>0:
                    var_left = (s2_left - (1/w_left)*s_left*s_left) / ((n_left-1)*w_left/n_left)
                    var_right = (s2_right - (1/w_right)*s_right*s_right)/((n_right-1)*w_right/n_right)
                    if var_left > 0:
                        t = (mean_left - mean_right) / (((var_left) /
                                                        n_left + (var_right)/n_right) ** 0.5)
                        if n_right > 1 and n_left > 1 and var_left > 0:
                            df = (var_left / n_left + var_right/n_right) / (
                                (var_left/n_left)/(n_left-1) + (var_right/n_right)/(n_right-1))
                        else:
                            df = max(n_left, n_right) - 1
                        p_value = (1 - scipy_stats.t.cdf(abs(t), df))*2
                    else:
                        t = np.nan
                        p_value = np.nan
                else:
                    t = np.nan
                    p_value = np.nan
                #endregion
                return {
                    'count_left':n_left, 'count_right': n_right,
                    'mse': mse, 'poisson':poisson, 'mae':mae,
                    't': t, 'p_value': p_value
                    }
        elif self.dep_var.type == Variable_Types.categorical:
            #region frequencies
            n_left, n_right, w_left, w_right = 0, 0, 0, 0
            freq_total, freq_left, freq_right = {}, {}, {}
            weights_total, weights_left, weights_right = {}, {}, {}
            ws = 0
            for i in self.sample.index:
                w = 1 if self.sample.weights == '1' else self.sample.data.values[self.sample.weights][i]
                y = self.sample.data.values[self.dep_var.name][i]
                if y in weights_total.keys():
                    weights_total[y] += w
                    freq_total[y] += 1
                else:
                    weights_total[y] = w
                    freq_total[y] = 1
                in_left = False
                if var_split.type == Variable_Types.numeric:
                    in_left = self.sample.data.values[var_split.name][i] <= value_split
                elif var_split.type == Variable_Types.categorical:
                    for v in value_split:
                        if self.sample.data.values[var_split.name][i] == v:
                            in_left = True
                            break
                if in_left:
                    if y in weights_left.keys():
                        weights_left[y] += w
                        freq_left[y] += 1
                    else:
                        weights_left[y] = w
                        freq_left[y] = 1
                    w_left += w
                    n_left += 1
                else:
                    if y in weights_right.keys():
                        weights_right[y] += w
                        freq_right[y] += 1
                    else:
                        weights_right[y] = w
                        freq_right[y] = 1
                    w_right += w
                    n_right += 1
                ws += w
            #endregion
            #region impurities
            #region total
            gini_total, entropy_total, max_p = 0, 0, 0
            for j in weights_total.keys():
                pj = weights_total[j]/ws
                gini_total += pj*(1-pj)
                entropy_total -= pj*math.log(pj)
                if pj > max_p:
                    max_p = pj
            if max_p != 0:
                mis_total = 1 - max_p
            else:
                mis_total = np.nan
            #endregion
            #region left
            gini_left, entropy_left, max_p = 0, 0, 0
            for j in weights_left.keys():
                pj = weights_left[j]/w_left
                gini_left += pj*(1-pj)
                entropy_left -= pj * math.log(pj)
                if pj > max_p:
                    max_p = pj
            if max_p != 0:
                mis_left = 1 - max_p
            else:
                mis_left = np.nan
            #endregion
            #region right
            gini_right, entropy_right, max_p = 0, 0, 0
            for j in weights_right.keys():
                pj = weights_right[j]/w_right
                gini_right += pj*(1-pj)
                entropy_right -= pj * math.log(pj)
                if pj > max_p:
                    max_p = pj
            if max_p != 0:
                mis_right = 1 - max_p
            else:
                mis_right = np.nan
            #endregion
            gini = gini_total - w_left/ws * gini_left - w_right/ws * gini_right
            info = entropy_total - w_left/ws * entropy_left - w_right/ws * entropy_right
            mis = mis_total - w_left/ws * mis_left - w_right/ws * mis_right
            #endregion
            #region chi2, p_value
            chi2, p_value = np.nan, np.nan
            if n_left>0 and n_right>0:
                chi2 = 0
                for j in freq_total.keys():
                    O_left = freq_left[j] if j in freq_left.keys() else 0
                    O_right = freq_right[j] if j in freq_right.keys() else 0
                    E_left = (O_left + O_right) / (n_left + n_right) * n_left
                    E_right = (O_left + O_right) / (n_left + n_right) * n_right
                    chi2 += (O_left-E_left)**2/E_left + \
                        (O_right-E_right)**2/E_right
                df = len(freq_total.keys())-1
                p_value = 1 - scipy_stats.chi2.cdf(chi2, df)
            #endregion
            return {
                    'count_left':n_left, 'count_right': n_right,
                    'gini': gini, 'entropy':info, 'misclassification':mis,
                    'chi2': chi2, 'p_value': p_value
                    }

    # method = mse, poisson, mae, p-value, or gini, entropy, misclassification, p-value
    def best_value(self, var_split: Variable, min_sample:int=1, min_significant:int=1, method=Methods.mse):
        #region type of dependence variable and method of estimation
        if self.dep_var.type == Variable_Types.numeric and (method != Methods.mse 
                                                            and method != Methods.poisson
                                                            and method != Methods.mae
                                                            and method != Methods.p_value):
            raise ValueError(
                f"for dependent variable {self.dep_var} thare are these methods:{Methods.mse}, {Methods.poisson}, {Methods.mae}")
        if self.dep_var.type == Variable_Types.categorical and (method != Methods.gini
                                                            and method != Methods.entropy
                                                            and method != Methods.misclassification
                                                            and method != Methods.p_value):
            raise ValueError(
                f"for dependent variable {self.dep_var} thare are these methods:{Methods.gini}, {Methods.entropy}, {Methods.misclassification}")
        #endregion
        if len(self.sample.index) >= min_sample:
            start  = True
            is_min_sample = False
            values = var_split.values_set(self.sample)
            for value in values:
                stats_sp = self.stats_split(var_split, value, method)
                if stats_sp != None:
                    if self.dep_var.type == Variable_Types.numeric:
                        mse, poisson, mae = stats_sp['mse'], stats_sp['poisson'], stats_sp['mae']
                        t, p_value = stats_sp['t'], stats_sp['p_value']
                        if stats_sp['count_left'] >= min_sample and stats_sp['count_right'] >= min_sample \
                            and stats_sp['p_value']<=min_significant:
                            if start:
                                is_min_sample = True
                                value_min, mse_min, poisson_min, mae_min= value, mse, poisson, mae
                                t_min, p_value_min = t, p_value
                                start = False
                            else:
                                if method == Methods.mse:
                                    if mse < mse_min:
                                        value_min, mse_min= value, mse
                                        t_min, p_value_min = t, p_value
                                elif method == Methods.poisson:
                                    if poisson < poisson_min:
                                        value_min, poisson_min= value, poisson
                                        t_min, p_value_min = t, p_value
                                elif method == Methods.mae:
                                    if mae < mae_min:
                                        value_min, mae_min= value, mae
                                        t_min, p_value_min = t, p_value
                                elif method == Methods.p_value:
                                    if p_value < p_value_min:
                                        value_min, p_value_min, t_min = value, p_value, t
                    elif self.dep_var.type == Variable_Types.categorical:
                        gini, info, mis = stats_sp['gini'], stats_sp['entropy'], stats_sp['misclassification']
                        chi2, p_value = stats_sp['chi2'], stats_sp['p_value']
                        if stats_sp['count_left'] >= min_sample and stats_sp['count_right'] >= min_sample \
                                and stats_sp['p_value'] <= min_significant:
                            if start:
                                is_min_sample = True
                                value_min, gini_min, entropy_min, mis_min = value, gini, info, mis
                                chi2_min, p_value_min = chi2, p_value
                                start = False
                            else:
                                if method == Methods.gini:
                                    if gini < gini_min:
                                        value_min, gini_min = value, gini
                                        chi2_min, p_value_min = chi2, p_value
                                elif method == Methods.entropy:
                                    if info < entropy_min:
                                        value_min, entropy_min = value, info
                                        chi2_min, p_value_min = chi2, p_value
                                elif method == Methods.misclassification:
                                    if mis < mis_min:
                                        value_min, mis_min = value, mis
                                        chi2_min, p_value_min = chi2, p_value
                                elif method == Methods.p_value:
                                    if p_value < p_value_min:
                                        value_min, p_value_min, chi2_min = value, p_value, chi2
            if is_min_sample:
                if self.dep_var.type == Variable_Types.numeric:
                    if method == Methods.mse:
                        return {'value': value_min, 'method': mse_min, 't': t_min, 'p-value': p_value_min}
                    elif method == Methods.poisson:
                        return {'value': value_min, 'method': poisson_min, 't': t_min, 'p-value': p_value_min}
                    elif method == Methods.mae:
                        return {'value': value_min, 'method': mae_min, 't': t_min, 'p-value': p_value_min}
                    elif method == Methods.p_value:
                        return {'value': value_min, 'method': p_value_min, 't': t_min, 'p-value': p_value_min}
                elif self.dep_var.type == Variable_Types.categorical:
                    if method == Methods.gini:
                        return {'value': value_min, 'method': gini_min, 'chi2': chi2_min, 'p-value': p_value_min}
                    elif method == Methods.entropy:
                        return {'value': value_min, 'method': entropy_min, 'chi2': chi2_min, 'p-value': p_value_min}
                    elif method == Methods.misclassification:
                        return {'value': value_min, 'method': mis_min, 'chi2': chi2_min, 'p-value': p_value_min}
                    elif method == Methods.p_value:
                        return {'value': value_min, 'method': p_value_min, 'chi2': chi2_min, 'p-value': p_value_min}

    def best_split(self, indep_vars:list, min_sample:int=1, min_significant:int=1, method=Methods.mse):
        start, has_min, n = True, False, 0
        for var in indep_vars:
            # print(var)
            if len(var.values(self.sample)) > 1:
                n += 1
                bsv = self.best_value(var, min_sample, min_significant, method)
                if bsv != None:
                    has_min = True
                    if start:
                        var_min = var
                        val_min, cr_min = bsv['value'], bsv['method']
                        if self.dep_var.type == Variable_Types.numeric:
                            t_min = bsv['t']
                        elif self.dep_var.type == Variable_Types.categorical:
                            t_min = bsv['chi2']
                        p_value_min = bsv['p-value']
                        start = False
                    else:
                        if bsv['method'] < cr_min:
                            var_min = var
                            val_min, cr_min = bsv['value'], bsv['method']
                            if self.dep_var.type == Variable_Types.numeric:
                                t_min = bsv['t']
                            elif self.dep_var.type == Variable_Types.categorical:
                                t_min = bsv['chi2']
                            p_value_min = bsv['p-value']
        if has_min:
            if var_min.type == Variable_Types.numeric:
                left_index = [i for i in self.sample.index if self.sample.data.values[var_min.name][i] <= val_min]
                right_index = [i for i in self.sample.index if self.sample.data.values[var_min.name][i] > val_min]
                left_rel, right_rel = '<=', '>'
            else:
                left_index, right_index = [], []
                for i in self.sample.index:
                    in_left = False
                    for v in val_min:
                        if self.sample.data.values[var_min.name][i] == v:
                            in_left = True
                            break
                    if in_left:
                        left_index.append(i)
                    else:
                        right_index.append(i)
                left_rel, right_rel = '==', '!='
            left = Node(Sample(self.sample.data, left_index),
                        self.dep_var,self,var_min,left_rel,val_min, t_min, p_value_min)
            right = Node(Sample(self.sample.data, right_index),
                         self.dep_var, self, var_min, right_rel, val_min, t_min, p_value_min)
            return {'left': left, 'right': right}

    def distribution(self, sorted: bool = True) -> dict:
        # {value:frequency, ...}
        stats = self.dep_var.stats.distribution(self.sample)
        if self.dep_var.type == Variable_Types.numeric:
            if sorted:
                dist = [(v, f['weight']) for v, f in stats]
                dist.sort(key=lambda vf: vf[0])
                return dict(dist)
            else:
                return {v: f['weight'] for v, f in stats}
        elif self.dep_var.type == Variable_Types.categorical:
            if sorted:
                dist = [(v, f['weight']) for v, f in stats]
                dist.sort(key=lambda vf: vf[1])
                return dict(dist)
            else:
                return {v: f['weight'] for v, f in stats}

    def dist_center(self):
        if self.dep_var.type == Variable_Types.numeric:
            return self.dep_var.stats.mean(self.sample)
        elif self.dep_var.type == Variable_Types.categorical:
            return self.dep_var.stats.mode(self.sample)

class Model:
    def __init__(self, dep_var:Union[str,dict,Variable], indep_vars:Union[dict,list,Variable], 
                    min_sample:int=5, min_significant:int=1, method:str=Methods.mse):
        #region dep_var
        if type(dep_var) == str:
            self.dep_var = Variable(dep_var, Variable_Types.numeric)
        elif type(dep_var) == dict:
            self.dep_var = Variable.from_dict(dep_var)
        elif type(dep_var) == Variable:
            self.dep_var = dep_var
        else:
            print('Error: type of dependent Variable is wrong!')
        #endregion
        #region indep_vars
        if type(indep_vars) == list:
            if len(indep_vars) > 0:
                if type(indep_vars[0]) == str:
                    self.indep_vars = [Variable(var, Variable_Types.numeric) for var in indep_vars]
                elif type(indep_vars[0]) == Variable:
                    self.indep_vars = indep_vars
                else:
                    print('Error: type of dependent Variable is wrong!')
            else:
                print('Error: list of independent variables is empty!')
        elif type(indep_vars) == dict:
            self.indep_vars = [Variable(nam, typ) for nam, typ in indep_vars.items()]
        else:
            print('Error: indep_var has to be a list of strings, dictionaries or variables!')
        #endregion
        self.min_sample = min_sample
        self.min_significant = min_significant
        self.method = method

    def __str__(self):
        return f"dep_var: '{self.dep_var}'\n\n" +  \
                f"indep_vars: \n{Variables(self.indep_vars)}\n" + \
                f"min_sample: {self.min_sample}\n" + \
                f"min_significant: {self.min_significant}\n" + \
                f"method: {self.method}\n"

    __node_list = []
    leafs = []
    __summary_str, __full_str = '', ''
    def __nodes(self, node:Node, do_print:bool=True, is_summary:bool=True):
        bs0 = node.best_split(self.indep_vars,self.min_sample, self.min_significant, self.method)
        if bs0 != None:
            left, right = bs0['left'], bs0['right']
            Model.__node_list.append(left)
            print_str = ' ' * 5 * (left.depth()-1) + (left.to_summary_str() if is_summary 
                                                        else left.to_full_str())
            
            Model.__summary_str += ' ' * 5 * \
                    (left.depth()-1) + left.to_summary_str() + '\n'
            Model.__full_str += ' ' * 5 * \
                    (left.depth()-1) + left.to_full_str() + '\n'
            if do_print:
                print(print_str)
            self.__nodes(left, do_print, is_summary)

            Model.__node_list.append(right)
            print_str = ' ' * 5 * (right.depth()-1) + (right.to_summary_str() if is_summary 
                                                        else right.to_full_str())
            Model.__summary_str += ' ' * 5 * \
                    (right.depth()-1) + right.to_summary_str() + '\n'
            Model.__full_str += ' ' * 5 * \
                    (right.depth()-1) + right.to_full_str() + '\n'
            if do_print:
                print(print_str)
            self.__nodes(right, do_print, is_summary)
        else:
            leaf = Model.__node_list[-1]
            
            res = ' '*5 + ' ' * 5 * (leaf.depth()-1)
            if self.dep_var.type == Variable_Types.numeric:
                mean = leaf.dep_var.stats.mean(leaf.sample)
                mse = leaf.dep_var.stats.std(leaf.sample)**2
                res += f'S{len(Model.leafs)}: n = {len(leaf.sample.index)}, {leaf.dep_var.name}: mean = {mean:.4f}, mse = {mse:.4f}'
            elif self.dep_var.type == Variable_Types.categorical:
                stats = leaf.dep_var.stats.distribution(leaf.sample)
                dist = '{'
                for v in stats:
                    dist += f"{v}:{stats[v]['count']}, "
                dist = dist[:-2] + '}'
                res += f'S{len(Model.leafs)}: n = {len(leaf.sample.index)}, {leaf.dep_var.name} = {dist}'
            Model.__summary_str += res + '\n'
            Model.__full_str += res + '\n'

            leaf.__class__ = Leaf
            Model.leafs.append(leaf)
            if do_print:
                print(res)
    
    def estimate(self, sample:Sample, do_print:bool=True, is_summary:bool=True):  
        Model.__node_list = []
        Model.leafs = []
        Model.__summary_str, Model.__full_str = '', ''
        node0 = Node(sample, self.dep_var)
        self.__nodes(node0, do_print, is_summary)
        return Equation(sample, self.dep_var, self.indep_vars, Model.__node_list,Model.leafs, Model.__summary_str, Model.__full_str)
    
class Leaf(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def __str__(self) -> str:
        if Model.leafs != []:
            zeros = '0' * (len(str(len(Model.leafs)))-1)
            return f'S{zeros}{Model.leafs.index(self)}'
        else:
            return super().__str__()
    
    def definition(self)->str:
        branch = self.parents() + [self]
        define = ''
        for nod in branch:
            if nod.split_var != None:
                x = nod.split_var.name
                v = nod.split_value
                define += f' and {nod.split_var.name} {nod.relation} {nod.split_value}'
        if len(define)>5:
            return define[5:]
    
    
class Equation:
    def __init__(self, sample:Sample , dep_var: Union[str, dict, Variable], 
                indep_vars: Union[dict, list, Variables], 
                node_list: list[Node], leafs: list[Node] = [], summary_str: str = '', full_str: str = '') -> None:
        self.sample = sample
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.node_list = node_list
        self.leafs = leafs
        self.summary_str = summary_str
        self.full_str = full_str
    
    def __str__(self):
        return self.summary_str

    forecast_outputs = ['leaf', 'point', 'dist']
    def forecast(self, sample: Sample, name:str='', output:str='point'):
        if name == '':
            name = f'{self.dep_var.name}_foerecast'
            if name in sample.data.variables():
                k = 1
                while f'{name}_{k}' in sample.data.variables():
                    k += 1
                name += f'_{k}'
        
        values = {}
        for i in sample.index:
            for leaf in self.leafs:
                branch = leaf.parents() + [leaf]
                in_sample = True
                for nod in branch:
                    if nod.split_var != None:
                        x = sample.data.values[nod.split_var.name][i]
                        v = nod.split_value
                        if nod.relation == '<=':
                            in_sample = in_sample and (x <= v)
                        elif nod.relation == '>':
                            in_sample = in_sample and (x > v)
                        elif nod.relation == '!=':
                            in_sample = in_sample and (not (x in v))
                        elif nod.relation == '==':
                            in_sample = in_sample and (x in v)
                if in_sample:
                    if output == 'point':
                        values[i] = leaf.dist_center()
                    elif output == 'dist':
                        values[i] = leaf.distribution()
                    elif output == 'leaf':
                        values[i] = leaf
                    break
        return Data(type=sample.data.type, values={name:values})

    anova = ''
    def goodness_of_fit(self, sample:Sample, decimals:int=4, do_print:bool=True):
        yf_data = self.forecast(sample)
        yf_name, k = f'{self.dep_var.name}_foerecast', 1
        while k < len(yf_data.variables()):
            if f'{yf_name}_{k}' in yf_data.variables():
                yf_name = f'{yf_name}_{k}'
                break
            k += 1

        if self.dep_var.type == Variable_Types.numeric:
            res = {}
            #region r2, r2adj
            sse2, sst, sst2, ws, n_data = 0, 0, 0, 0, 0
            sst_f, sst2_f, sse= 0,0,0
            for i in sample.index:
                if i in yf_data.values[yf_name].keys():
                    y = sample.data.values[self.dep_var.name][i]
                    if sample.weights == '1':
                        w = 1
                    else:
                        w = sample.data.values[sample.weights][i]
                    yf = yf_data.values[yf_name][i]
                    sse += w * (y-yf)
                    sse2 += w * (y - yf) ** 2
                    sst += w * y
                    sst2 += w * (y ** 2)
                    sst_f += w * yf
                    sst2_f += w * (yf ** 2)
                    ws += w
                    n_data += 1
                else:
                    pass
                    #print('in index', str(i).center(10), 'forecast has ecountered an error!')
            res['r2'] = 1 - sse2 / (sst2 - sst**2/ws)
            res['r2adj'] = 1-(1-res['r2'])*(n_data-1)/(n_data-len(self.node_list)-1)
            res ['r2_McKelvey & Zavonia'] = (sst2_f-sst_f**2/ws) / (sst2_f-sst_f**2/ws+sse2-sse**2/ws)
            #endregion
            #region anova components
            ss_total = sst2 - sst**2/ws
            ss_within = sse2
            ss_between = ss_total-ss_within

            df_total = n_data-1
            df_between = len(self.node_list)-1
            df_within = df_total - df_between
    

            ms_between = ss_between/df_between
            ms_within = ss_within/df_within
            res['f'] = ms_between/ms_within
            res['p_value'] = 1 - scipy_stats.f.cdf(res['f'], df_between, df_within)
            #endregion
            #region anova table
            len_ss = max(number_of_digits(ss_total, decimals), 
                        number_of_digits(ss_between, decimals),
                        number_of_digits(ss_within, decimals))
            len_df = number_of_digits(df_total, decimals)
            len_ms = max(number_of_digits(ms_between, decimals),
                        number_of_digits(ms_within, decimals))
            if np.isnan(res['p_value']):
                len_f = 16
            else:
                len_f = number_of_digits(res['f'], decimals) + 13 + decimals
            len_line = 48+len_ss+len_df+len_ms+len_f

            prt = '-'*int((len_line-5)/2)+'ANOVA'+'-'*int((len_line-5)/2) + '\n'
            prt += 'H0: The mean of all samples is equal.\n'
            prt += '\n'
            ss_side = int((len_ss-2)/2)
            df_side = int((len_df-2)/2)
            ms_side = int((len_ms-2)/2)
            f_side = int((len_f-1)/2)
            prt += '-'*len_line + '\n'
            prt += ' '*28 + '|  ' + ' '*ss_side+'SS'+' '*(len_ss - ss_side-2)+' '*5+ \
                    ' '*df_side+'df'+' '*(len_df - df_side-2)+' '*5+ \
                    ' '*ms_side+'ms'+' '*(len_ms - ms_side-2)+' '*5+ \
                    ' '*f_side+'F'+' '*(len_f - f_side-1)+' '*2 + '\n'
            prt += '-'*len_line + '\n'
            ss_side = int((len_ss-number_of_digits(ss_between, decimals))/2)
            df_side = int((len_df-number_of_digits(df_between, decimals))/2)
            ms_side = int((len_ms-number_of_digits(ms_between, decimals))/2)
            if np.isnan(res['p_value']):
                f_r, p_value_r = 'nan', 'nan'
            else:
                f_r = int(res['f']*10**decimals)/10**decimals
                p_value_r = int(res['p_value']*10**decimals)/10**decimals
            prt += '  Between groups (Factor)   ' + '|  ' + \
                ' '*ss_side+f'{int(ss_between*10**decimals)/10**decimals}'+ \
                ' '*(len_ss - ss_side - number_of_digits(ss_between, decimals))+ \
                ' '*5+' '*df_side+f'{int(df_between*10**decimals)/10**decimals}'+ \
                ' '*(len_df - df_side-number_of_digits(df_between, decimals))+ \
                ' '*5+' '*ms_side+f'{int(ms_between*10**decimals)/10**decimals}'+ \
                ' '*(len_ms - ms_side-number_of_digits(ms_between, decimals))+ \
                ' '*5+f'{f_r}'+ \
                    f'(p-value:{p_value_r})' + ' '*2 + '\n'
            ss_side = int((len_ss-number_of_digits(ss_within, decimals))/2)
            df_side = int((len_df-number_of_digits(df_within, decimals))/2)
            ms_side = int((len_ms-number_of_digits(ms_within, decimals))/2)
            prt += '  Within groups (Error)     ' + '|  ' + ' '*ss_side+ \
                f'{int(ss_within*10**decimals)/10**decimals}'+ \
                ' '*(len_ss - ss_side-number_of_digits(ss_within, decimals))+ \
                ' '*5+' '*df_side+f'{int(df_within*10**decimals)/10**decimals}'+ \
                ' '*(len_df - df_side-number_of_digits(df_within, decimals))+ \
                ' '*5+' '*ms_side+f'{int(ms_within*10**decimals)/10**decimals}'+ \
                ' '*(len_ms - ms_side-number_of_digits(ms_within, decimals)) + '\n'
            prt += '-'*len_line + '\n'
            ss_side = int((len_ss-number_of_digits(ss_total, decimals))/2)
            df_side = int((len_df-number_of_digits(df_total, decimals))/2)
            prt+= '  Total                     ' + '|  ' + ' '*ss_side+ \
                    f'{int(ss_total*10**decimals)/10**decimals}'+ \
                    ' '*(len_ss - ss_side-number_of_digits(
                    ss_total, decimals))+' '*5+' '*df_side+ \
                    f'{int(df_total*10**decimals)/10**decimals}'+ \
                    ' '*(len_df - df_side-number_of_digits(df_total, decimals)) + '\n'
            prt += '-'*len_line + '\n'
            #endregion
            res['anova'] = prt
            if do_print:
                print('\n***DETERMINANATION COEFFICIENTS***', '\n')
                print('r2:'.ljust(22), res['r2'])
                print('r2adj:'.ljust(22), res['r2adj'])
                print('r2_McKelvey & Zavonia:'.ljust(22), res['r2_McKelvey & Zavonia'])
                print('Note: r2 and r2adj are only used for in-sample forecasts, but Mackelvey & Zavonia r2 can use for out-of-sample forecasts')
                print()
                print(prt)
            return res
        elif self.dep_var.type == Variable_Types.categorical:
            res = {}
            counts = {}
            max_lenght = 0
            for i in sample.index:
                if i in yf_data.values[yf_name].keys():
                    y = sample.data.values[self.dep_var.name][i]
                    if sample.weights == '1':
                        w = 1
                    else:
                        w = sample.data.values[sample.weights][i]
                    yf = yf_data.values[yf_name][i]
                    if yf in counts.keys():
                        if y in counts[yf].keys():
                            counts[yf][y] += 1
                            if len(str(round(counts[yf][y],decimals)))>max_lenght:
                                max_lenght = len(str(round(counts[yf][y],decimals)))
                            if len(str(y)) > max_lenght:
                                max_lenght = len(str(y))
                        else:
                            counts[yf][y] = 1
                    else:
                        counts[yf] = {}
            max_lenght += 5
            max_lenght = max(12, max_lenght)
            ys = list({y for yf in counts for y in counts[yf]})
            ys.sort()


            conf_m_head = ' ' + '-'*(max_lenght*(len(ys)+1)-1) + '\n'
            conf_m_head += '|' + 'predict'.center(max_lenght-2) + '|' + 'actual'.center(max_lenght*len(ys)) + '|\n'
            conf_m_head += '|' + ' '*(max_lenght-2) + '|' + '-'*max_lenght*len(ys) + '\n|' + ' ' * (max_lenght-2) + '|'
            confusion_matrix  = ''

            trues, falses = 0, 0
            precisions, recalls, f1 = {}, {}, {}
            accuracy_table = ' ' + '-' * (4*max_lenght-1) + '\n'
            accuracy_table += '|' + 'class'.center(max_lenght-2) + '|' + 'precision'.center(max_lenght) + \
                            'recall'.center(max_lenght) + 'F1-score'.center(max_lenght) + '|\n'
            accuracy_table += ' ' + '-' * (4*max_lenght-1) + '\n'
            for yf in ys:
                row = '|' + str(yf).center(max_lenght-2) + '|'
                tp, fp, fn = 0, 0, 0
                for y in ys:
                    if not (str(y) in conf_m_head):
                        conf_m_head += str(y).center(max_lenght)
                    if yf in counts.keys():
                        if y in counts[yf].keys():
                            row += str(round(counts[yf][y],decimals)).center(max_lenght)
                            if yf == y:
                                trues += counts[yf][y]
                                tp += counts[yf][y]
                            else:
                                falses += counts[yf][y]
                                fp += counts[yf][y]
                        else:
                            row += str(0).center(max_lenght)
                    else:
                        row += '0'.center(max_lenght)
                    if y in counts.keys():
                        if yf in counts[y].keys():
                            if y != yf:
                                fn += counts[y][yf]
                precisions[yf] = tp/(tp+fp) if tp+fp>0 else '-'
                precision = round(tp/(tp+fp)*100,2) if tp+fp>0 else '-'
                recalls[yf] = tp/(tp+fn) if tp+fn>0 else '-'
                recall = round(tp/(tp+fn)*100, 2) if tp+fn>0 else '-'
                if is_numeric(precisions[yf]) and is_numeric(recalls[yf]):
                    f1[yf] = 2*(precisions[yf]*recalls[yf])/(precisions[yf]+recalls[yf])
                    f1_ = round(2*(precisions[yf]*recalls[yf])/(precisions[yf]+recalls[yf])*100,2)
                else:
                    f1[yf] = '-'
                    f1_ = '-'
                accuracy_table += '|' + str(yf).center(max_lenght-2)  + '|' + \
                                str(precision).center(max_lenght) + \
                                str(recall).center(max_lenght) + \
                                str(f1_).center(max_lenght) + '|\n'
                confusion_matrix += row + '|\n'
            conf_m_head += '|\n ' + '-' * (max_lenght*(len(ys)+1)-1)
            conf_m_footer = ' ' + '-'*(max_lenght*(len(ys)+1)-1)
            res['confusion_matrix'] = conf_m_head + '\n' + \
                confusion_matrix + conf_m_footer
            accuracy_table += ' ' + '-' * (4*max_lenght-1)
            res['accuracy_table'] = accuracy_table
            res['precisions'] = precisions
            res['recalls'] = recalls
            res['f1_scores'] = f1
            res['micro-f1'] = trues / (trues + falses)
            res['macro-f1'] = sum([f1[f] for f in f1 if f1[f] != '-'])/len(f1)
            if do_print:
                print('confusion_matrix')
                print(res['confusion_matrix'])
                print('accuracy_table')
                print(res['accuracy_table'])
                print('micro-f1:', res['micro-f1'])
                print('macro-f1', res['macro-f1'])
            return res

    def first_nodes(self):
        nds = []
        for nod in self.node_list:
            if nod.depth() == 1:
                nds.append(nod)
        return nds

    __nod = None
    def __to_dict(self, nodes:list, decimals:int) -> None:
        nod = None
        if nodes == []:
            mean = self.__nod.dist_center()
            if is_numeric(mean):
                mean = int(mean*10**decimals)/10**decimals
            return f'{self.__nod}\nn = {len(self.__nod.sample.index)}\n{self.__nod.dep_var.name}: {mean}'
        else:
            value_dict = {}
            for node in nodes:
                nod = node
                if node.childs(self.node_list) == []:
                    self.__nod = node
                if is_numeric(node.split_value):
                    value_dict[f'{node.relation}{int(node.split_value*10**decimals)/10**decimals}\n'] = self.__to_dict(node.childs(self.node_list), decimals)
                else:
                    if node.relation == '==':
                        values = node.split_value
                    else:
                        values = [v for v in node.split_var.values(
                            node.sample) if not v in node.split_value]
                    m = 20
                    val = '['
                    part = ''
                    start = True
                    for v in values:
                        if len(part + str(v)) > m:
                            if start:
                                val += part
                                start = False
                            else:
                                pass
                                val += '\n' + part
                            part = 'or ' + str(v)
                        else:
                            part += ' or ' + str(v) if part != '' else str(v)
                    val += '\n' + part + ']' if not start else part + ']'
                    value_dict[f'=={val}\n'] = self.__to_dict(node.childs(self.node_list), decimals)
            if nod != None:
                prt = nod.parent
                mean = self.__nod.dist_center()
                return {f'{prt.childs(self.node_list)[0].split_var.name}\nn = {len(prt.sample.index)}\n{prt.dep_var.name}: {int(mean*10**decimals)/10**decimals}':value_dict}

    def plot(self, size:int=10, decimals:int=2):
        dic = self.__to_dict(self.first_nodes(), decimals)
        Plot.createPlot(dic, size)

    #region old save
    # def save(self, path_name:str = 'results') -> None:
    #     results = {}
    #     results['sample'] = self.sample
    #     if self.sample.name != None and path_name == 'results':
    #         path_name = self.sample.name
    #     results['dep_var'] = self.dep_var
    #     results['indep_vars'] = self.indep_vars
    #     results['node_list'] = self.node_list
    #     results['leafs'] = self.leafs
    #     results['summary_str'] = self.summary_str
    #     results['full_str'] = self.full_str
    #     if os.path.exists(path_name):
    #         os.remove(path_name)
    #     with open(path_name,'wb') as f:
    #         pickle.dump(results, f)
    #     print('Results were saved successfully')

    # @classmethod
    # def load(cls, path_name:str = 'results'):
    #     with open(path_name, 'rb') as f:
    #         results = pickle.load(f)
    #     return cls(results['sample'], results['dep_var'],
    #             results['indep_vars'], results['node_list'],
    #             results['leafs'], results['summary_str'],
    #             results['full_str'])
    #endregion

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print('Results were saved successfully')

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'rb') as f:
            eq = pickle.load(f)
        return eq

class Plot:
    # These codes get from: https://www.programmersought.com/article/93024638825/ with a little edit.
    """
    boxstyle=
    Class	Name	Attrs
    Circle	circle	pad=0.3
    DArrow	darrow	pad=0.3
    LArrow	larrow	pad=0.3
    RArrow	rarrow	pad=0.3
    Round	round	pad=0.3,rounding_size=None
    Round4	round4	pad=0.3,rounding_size=None
    Roundtooth	roundtooth	pad=0.3,tooth_size=None
    Sawtooth	sawtooth	pad=0.3,tooth_size=None
    Square	square	pad=0.3
    """
    # define nodeType Leaf node, distinguish node, definition of arrow type
    decisionNode = dict(boxstyle="round", fc="0.8")  # sawtooth
    leafNode = dict(boxstyle="square", fc="1", ec="b")  # round4 , lw=2 (width), ec: color w='White' 
    arrow_args = dict(arrowstyle="<-", color='b',lw=0.3,ls='--')

    # Define node function
    @staticmethod
    def plotNode(nodeText, centerPt, parentPt, nodeType, size):
        # , xycoords='axes fraction' , textcoords='axes fraction'
        Plot.createPlot.ax1.annotate(nodeText, xy=parentPt, xytext=centerPt,
                                va='bottom', ha='center', bbox=nodeType, arrowprops=Plot.arrow_args,
                                     fontsize=size)  # size=size,
        # This parameter is a bit scary. did not understand


    # Numleafs function
    @staticmethod
    def getNumLeafs(myTree):
        numLeafs = 0
        firstList = list(myTree.keys())
        firstStr = firstList[0]
        secondDict = myTree[firstStr]  # Read the value of the key value
        for key in secondDict.keys():  # Monitor if there is a dictionary collection
            if type(secondDict[key]).__name__ == 'dict':
                numLeafs += Plot.getNumLeafs(secondDict[key])
            else:
                numLeafs += 1
        return numLeafs

    # depths function
    @staticmethod
    def getTreeDepth(myTree):
        maxDepth = 0
        firstList = list(myTree.keys())
        firstStr = firstList[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                thisDepth = 1+Plot.getTreeDepth(secondDict[key])
            else:
                thisDepth = 1
            if thisDepth > maxDepth:
                maxDepth = thisDepth
        return maxDepth

    # Plots text between child and parent
    @staticmethod
    def plotMidText(cntrPt, parentPt, txtString, size):
        xMid = (parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
        yMid = (parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
        Plot.createPlot.ax1.text(xMid, yMid, txtString, fontsize=size-2)

    # define the main functions, plotTree
    @staticmethod
    def plotTree(myTree, parentPt, nodeTxt, size):  # if the first key tells you what feat was split on
        numLeafs = Plot.getNumLeafs(myTree)  # this determines the x width of this tree
        depth = Plot.getTreeDepth(myTree)
        firstList = list(myTree.keys())
        firstStr = firstList[0]  # the text label for this node should be this
        cntrPt = (Plot.plotTree.xOff + (1 + float(numLeafs)) /(2.0*Plot.plotTree.totalW), 
                Plot.plotTree.yOff)
        Plot.plotMidText(cntrPt, parentPt, nodeTxt, size)
        Plot.plotNode(firstStr, cntrPt, parentPt, Plot.decisionNode, size)
        secondDict = myTree[firstStr]
        Plot.plotTree.yOff = Plot.plotTree.yOff - 1.0/Plot.plotTree.totalD
        for key in secondDict.keys():
            # test to see if the nodes are dictonaires, if not they are leaf nodes
            if type(secondDict[key]).__name__ == 'dict':
                Plot.plotTree(secondDict[key], cntrPt, str(key), size)  # recursion
            else:  # it's a leaf node print the leaf node
                Plot.plotTree.xOff = Plot.plotTree.xOff + 1.0/Plot.plotTree.totalW
                Plot.plotNode(secondDict[key], (Plot.plotTree.xOff,
                        Plot.plotTree.yOff), cntrPt, Plot.leafNode, size)
                Plot.plotMidText((Plot.plotTree.xOff, Plot.plotTree.yOff), cntrPt, str(key), size)
        Plot.plotTree.yOff = Plot.plotTree.yOff + 1.0/Plot.plotTree.totalD
        #if you do get a dictonary you know it's a tree, and the first element will be another dict

    # Perform graphic display
    @staticmethod
    def createPlot(inTree, size=10):
        import matplotlib.pyplot as plt
        fig = plt.figure(1, facecolor='white', figsize=(16,9))
        
        axprops = dict(xticks=[], yticks=[])
        Plot.createPlot.ax1 = fig.add_subplot(frameon=False, **axprops)
        Plot.plotTree.totalW = float(Plot.getNumLeafs(inTree))
        Plot.plotTree.totalD = float(Plot.getTreeDepth(inTree))
        Plot.plotTree.xOff = -0.5/Plot.plotTree.totalW
        Plot.plotTree.yOff = 1
        Plot.plotTree(inTree, (0.5, 1), '', size)

        plt.show()

