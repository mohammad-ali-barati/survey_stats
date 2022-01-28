from __future__ import annotations
from cProfile import label
from turtle import width
import pandas as pd
from _models.functions import *
from _models.data_process import *
import math

class Variable_Types:
    numeric = 'numeric'
    categorical = 'categorical'

class Variable:
    class __Stats:
        def __init__(self, var_name: str, var_type: str) -> None:
            self.var_name = var_name
            self.var_type = var_type

        def mean(self, sample: Sample) -> float:
            if self.var_type == Variable_Types.numeric:
                s, ws = 0, 0
                freq, wgh = {}, {}
                for i in sample.index:
                    if sample.weights == '1':
                        w = 1
                    else:
                        w = sample.data.values[sample.weights][i]
                    x = sample.data.values[self.var_name][i]
                    is_nan = False
                    if is_numeric(x):
                        if np.isnan(x):
                            is_nan = True
                    if is_numeric(w):
                        if np.isnan(w):
                            is_nan = True
                    if not is_nan:
                        s += w * x
                        ws += w
                if ws > 0:
                    mean = s/ws
                    return mean
                else:
                    raise ValueError(f"Error! number of observation is zero.")
            else:
                raise ValueError(
                    f"Error! variable {self.var_name} is not numeric.")

        def std(self, sample: Sample) -> float:
            if self.var_type == Variable_Types.numeric:
                s, s2, ws = 0, 0, 0
                for i in sample.index:
                    if sample.weights == '1':
                        w = 1
                    else:
                        w = sample.data.values[sample.weights][i]
                    x = sample.data.values[self.var_name][i]
                    is_nan = False
                    if is_numeric(x):
                        if np.isnan(x):
                            is_nan = True
                    if is_numeric(w):
                        if np.isnan(w):
                            is_nan = True
                    if not is_nan:
                        s += w * x
                        s2 += w * (x * x)
                        ws += w
                if ws > 0:
                    tss = s2 - (1/ws)*s*s  # Total Sum of Squares
                    var = tss/(ws-1) if ws > 1 else 0
                    return var**0.5
                else:
                    raise ValueError(f"Error! number of observation is zero.")
            else:
                raise ValueError(
                    f"Error! variable {self.var_name} is not numeric.")

        def count(self, sample: Sample) -> float:
            if self.var_type == Variable_Types.numeric:
                n = 0
                for i in sample.index:
                    x = sample.data.values[self.var_name][i]
                    is_nan = False
                    if is_numeric(x):
                        if np.isnan(x):
                            is_nan = True
                    if not is_nan:
                        n += 1
                return n
            else:
                raise ValueError(
                    f"Error! variable {self.var_name} is not numeric.")

        def tss(self, sample: Sample) -> float:
            if self.var_type == Variable_Types.numeric:
                s, s2, ws = 0, 0, 0
                for i in sample.index:
                    if sample.weights == '1':
                        w = 1
                    else:
                        w = sample.data.values[sample.weights][i]
                    x = sample.data.values[self.var_name][i]
                    is_nan = False
                    if is_numeric(x):
                        if np.isnan(x):
                            is_nan = True
                    if is_numeric(w):
                        if np.isnan(w):
                            is_nan = True
                    if not is_nan:
                        s += w * x
                        s2 += w * (x * x)
                        ws += w
                if ws > 0:
                    return s2 - (1/ws)*s*s  # Total Sum of Squares
                else:
                    raise ValueError(f"Error! number of observation is zero.")
            else:
                raise ValueError(
                    f"Error! variable {self.var_name} is not numeric.")

        def distribution(self, sample: Sample) -> float:
            ws = 0
            freq, wgh = {}, {}
            for i in sample.index:
                if sample.weights == '1':
                    w = 1
                else:
                    w = sample.data.values[sample.weights][i]
                x = sample.data.values[self.var_name][i]
                is_nan = False
                if is_numeric(x):
                    if np.isnan(x):
                        is_nan = True
                if is_numeric(w):
                    if np.isnan(w):
                        is_nan = True
                if not is_nan:
                    ws += w
                    if x in freq.keys():
                        freq[x] += 1
                        wgh[x] += w
                    else:
                        freq[x], wgh[x] = 1, w
            if ws > 0:
                dist = {}
                for v, f in freq.items():
                    dist[v] = {'count': f, 'weight': wgh[v]/ws}
                return dist

        def sum(self, sample: Sample) -> float:
            if self.var_type == Variable_Types.numeric:
                s = 0
                for i in sample.index:
                    x = sample.data.values[self.var_name][i]
                    is_nan = False
                    if is_numeric(x):
                        if np.isnan(x):
                            is_nan = True
                    if not is_nan:
                        s += x
                        print(s)
                return s
            else:
                raise ValueError(
                    f"Error! variable {self.var_name} is not numeric.")

        def median(self, sample: Sample) -> float:
            if self.var_type == Variable_Types.numeric:
                dist = self.distribution(sample)
                values = list(dist.keys())
                values.sort()
                comul_weights = 0
                for value in values:
                    comul_weights += dist[value]['weight']
                    if comul_weights >= 0.5:
                        return value
                raise ValueError(
                    f"Error! The total weights are probably less than half..")
            else:
                raise ValueError(
                    f"Error! variable {self.var_name} is not numeric.")

        def mode(self, sample: Sample) -> float:
            dist = self.distribution(sample)
            max_freq, mod = 0, None
            for value in dist:
                if dist[value]['count'] > max_freq:
                    mod = value
                    max_freq = dist[value]['count']
            return mod

        def min(self, sample: Sample, k: int = 1) -> float:
            if type(k) != int:
                raise ValueError(f"Error! type of k ({k}) must be integer.")
            elif k < 1:
                raise ValueError(f"Error! value of k ({k}) must be more than one.")
            vals = [x for x in sample.data.values[self.var_name].values() if not np.isnan(x)]
            if k == 1:
                return min(vals)
            elif k > 1:
                vals.sort()
                return vals[k]

        def max(self, sample: Sample, k: int = 1) -> float:
            if type(k) != int:
                raise ValueError(f"Error! type of k ({k}) must be integer.")
            elif k < 1:
                raise ValueError(f"Error! value of k ({k}) must be more than one.")
            vals = [x for x in sample.data.values[self.var_name].values() if not np.isnan(x)]
            if k == 1:
                return max(vals)
            elif k > 1:
                vals.sort(reverse=True)
                return vals[k]

        def percentile(self, sample: Sample, k: float) -> float:
            dist = self.distribution(sample)
            values = list(dist.keys())
            values.sort()
            comul_weights = 0
            for value in values:
                comul_weights += dist[value]['weight']
                if comul_weights >= 0.5:
                    return value
            raise ValueError(
                f"Error! The total weights are probably less than half..")

        def gini_coef(self, sample: Sample) -> float:
            pass  # TODO

    #types: 'numeric', 'categorical'
    def __init__(self, name:str, type:str='numeric') -> None:
        self.name = name
        self.type = type
        self.stats = Variable.__Stats(name, type)

    def __str__(self) -> str:
        return f'{self.name}:{self.type}'

    @classmethod
    def from_dict(cls, var_dict: dict) -> None:
        for var, typ in var_dict.items():
            return cls(var,typ)
    
    @classmethod
    def from_data(cls, data: Data, name:str) -> None:
        if name in data.values.keys():
            nums, cats = 0, 0
            for i in data.index():
                if type(data.values[name][i]) == float or type(data.values[name][i]) == int:
                    nums += 1
                else:
                    cats += 1
            if nums < cats:
                type_var = Variable_Types.categorical
            else:
                type_var = Variable_Types.numeric
            return cls(name, type_var)

    def values(self, sample: Sample) -> None:
        if self.name in  sample.data.values:
            # values = [v for i,v in sample.data.values[self.name].items() if (i in sample.index) and (not is_numeric(v))]
            values = []
            for i,v in sample.data.values[self.name].items():
                if i in sample.index:
                    if is_numeric(v):
                        if not np.isnan(v):
                            values.append(v)
                    else:
                        values.append(v)
            return list(set(values))

    def values_set(self, sample: Sample, half_of_set:bool=True):
        v_list = self.values(sample)
        v_list.sort()
        if self.type == 'numeric':
            return [i for i in v_list[:-1]]
        elif self.type == 'categorical':
            if v_list != None:
                if half_of_set:
                    n = int(len(subsets(v_list)) / 2)
                    return subsets(v_list)[1:n]
                else:
                    return subsets(v_list)

    def map(self, sample:Sample, old_values:list[list|str], new_values:list, other:str|int=np.nan, name:str=''):
        if name == '':
            name = f"{self.name}_mapped"
        # old_values = [[1,5,6],[7],[8,9,10]]  --->  new_values = ['a', 'b', 'c']
        if len(old_values) == len(new_values):
            if min([type(x)==list for x in old_values]):
                values = {}
                for index in sample.index:
                    for i, vals in enumerate(old_values):
                        if sample.data.values[self.name][index] in vals:
                            values[index] = new_values[i]
                        elif type(sample.data.values[self.name][index])==str:
                            if is_numeric_str(sample.data.values[self.name][index]):
                                if np.isnan(float(sample.data.values[self.name][index])):
                                    values[index] = np.nan
                        elif is_numeric(sample.data.values[self.name][index]):
                            if np.isnan(float(sample.data.values[self.name][index])):
                                values[index] = np.nan
                    if not index in values.keys():
                        values[index] = other
                return Data(sample.data.type, {name:values})
            elif  min([type(x)==str for x in old_values]):
                if [x[0] in ['=', '<', '>', '!'] for x in old_values]:
                    if min([type(x) == int or type(x) == str for x in new_values]):
                        values = {}
                        for index in sample.index:
                            x = float(sample.data.values[self.name][index])
                            for i, cond in enumerate(old_values):
                                if cond[:2] == '!=':
                                    if x != float(cond[2:]):
                                        values[index] = new_values[i]
                                        break
                                    else:
                                        if np.isnan(x):
                                            values[index]= np.nan
                                        else:
                                            values[index] = other
                                elif cond[:2] == '<=':
                                    if x <= float(cond[2:]):
                                        values[index] = new_values[i]
                                        break
                                    else:
                                        if np.isnan(x):
                                            values[index]= np.nan
                                        else:
                                            values[index] = other
                                elif cond[:2] == '>=':
                                    if x >= float(cond[2:]):
                                        values[index] = new_values[i]
                                        break
                                    else:
                                        if np.isnan(x):
                                            values[index]= np.nan
                                        else:
                                            values[index] = other
                                elif cond[0] == '=':
                                    if x == float(cond[1:]):
                                        values[index] = new_values[i]
                                        break
                                    else:
                                        if np.isnan(x):
                                            values[index]= np.nan
                                        else:
                                            values[index] = other
                                elif cond[0] == '<':
                                    if x < float(cond[1:]):
                                        values[index] = new_values[i]
                                        break
                                    else:
                                        if np.isnan(x):
                                            values[index]= np.nan
                                        else:
                                            values[index] = other
                                elif cond[0] == '>':
                                    if x > float(cond[1:]):
                                        values[index] = new_values[i]
                                        break
                                    else:
                                        if np.isnan(x):
                                            values[index]= np.nan
                                        else:
                                            values[index] = other
                                # else:
                                #     values[index] = np.nan
                        return Data(sample.data.type, {name:values})
                    else:
                        raise ValueError(f"Error! '{new_values}' must contain str or int.")
                else:
                    raise ValueError(f"Error! '{old_values}' must contain conditions that begin with one of these characters: '=', '!=', '<', '>', '<=', '>='.")
            else:
                raise ValueError(f"Error! '{old_values}' must include only list or only str, and it can not include both of them or other types.")
        else:
            raise ValueError(f"Error! the number of components of '{old_values}' and '{new_values}'")

class Variables:
    def __init__(self, var_list:list[Variable]=None) -> None:
        self.var_list = var_list

    def __str__(self) -> str:
        res = ''
        for v in self.var_list:
            res += v.__str__() + '\n'
        return res

    @classmethod
    def from_data(cls, data: Data, var_names:list = []) -> None:
        if var_names == []:
            var_names = data.variables()
        return cls([Variable.from_data(data, var) for var in data.values 
                            if (var in var_names) or (var_names == [])])

    @classmethod
    def from_dict(cls, dict:dict) -> None:
        # dict = {'name1':type1, 'name2':type2, ...}
        vars = []
        for nam, typ in dict.items():
            vars.append(Variable(nam, typ))
        return cls(vars)

    def to_variable(self, index:int=0):
        return self.var_list[index]
    
    def summary(self, sample:Sample):
        unknowns = []
        unknowns_w = [8, 4]
        numerics_cols = ['variable', 'count', 'mean', 'std']
        numerics_cols_w = [8, 5, 4, 3]
        numerics = []
        categoricals = []
        categoricals_w = [0, 0, 0, 0]
        for var in self.var_list:
            if not var.name in sample.data.variables():
                unknowns.append([var.name, var.type])
                if unknowns_w[0] < len(var.name):
                    unknowns_w[0] = len(var.name)
                if unknowns_w[1] < len(var.type):
                    unknowns_w[1] = len(var.type)
            elif var.type == Variable_Types.numeric:
                row = [var.name, var.stats.count(sample), var.stats.mean(sample), var.stats.std(sample)]
                numerics.append(row)
                for i, w in enumerate(numerics_cols_w):
                    if is_numeric(row[i]):
                        row_i = f'{row[i]:.2f}'
                    else:
                        row_i = row[i]
                    if w < len(row_i):
                        numerics_cols_w[i] = len(row_i)
            elif var.type == Variable_Types.categorical:
                stat = var.stats.distribution(sample)
                stat = [(x, stat[x]['count'], round(stat[x]['weight'],4)) for x in stat]
                stat.sort(key=lambda row: row[1], reverse=True)
                categoricals.append([var.name, stat])
                if categoricals_w[0] < len(var.name):
                    categoricals_w[0] = len(var.name)
                for val, count, weight in stat:
                    if categoricals_w[1] < len(str(val)):
                        categoricals_w[1] = len(str(val))
                    if categoricals_w[2] < len(str(count)):
                        categoricals_w[2] = len(str(count))
                    if categoricals_w[3] < len(str(weight)):
                        categoricals_w[3] = len(str(weight))

        unknowns_w = [i+5 for i in unknowns_w]
        max_unknown_table = sum(unknowns_w)
        numerics_cols_w = [i+5 for i in numerics_cols_w]
        max_numeric_table = sum(numerics_cols_w)
        categoricals_w = [i+5 for i in categoricals_w]
        max_categoricals_table = max(categoricals_w[0], sum(categoricals_w[1:]))
        max_width = max(max_unknown_table, max_numeric_table, max_categoricals_table)
        res = 'Summary'.center(max_width,'-') + '\n\n'
        if unknowns != []:
            res += 'Unknown Variables:' + '\n\n'
            res += '***Those variables for which there is no data.***' + '\n'
            res += '-' * max_unknown_table + '\n'
            res += 'variable'.center(unknowns_w[0]) + 'type'.center(unknowns_w[1]) + '\n'
            res += '-' * max_unknown_table + '\n'
            for var, type in unknowns:
                res += var.center(unknowns_w[0]) + type.center(unknowns_w[1]) + '\n'
            res += '-' * max_unknown_table + '\n'
            res += '\n\n'
        if numerics != []:
            res += 'Numeric Variables:' + '\n'
            res += '-'* max_numeric_table + '\n'
            row = ''
            for i, col in enumerate(numerics_cols):
                row += col.center(numerics_cols_w[i])
            res += row + '\n'
            res += '-'* max_numeric_table + '\n'
            for num in numerics:
                row = ''
                for i, col in enumerate(num):
                    row += str(col).center(numerics_cols_w[i])
                res += row + '\n'
            res += '-'* max_numeric_table + '\n'
            res += '\n\n'
        if categoricals != []:
            res += 'Categorical Variables:' + '\n\n'
            for var, stat in categoricals:
                res += var.center(max_categoricals_table) + '\n'
                res += '-' * max_categoricals_table + '\n'
                res += 'value'.center(categoricals_w[1]) + 'count'.center(categoricals_w[2])
                res += 'weight'.center(categoricals_w[3]) + '\n'
                res += '-' * max_categoricals_table + '\n'
                for val, count, weight in stat:
                    res += str(val).center(categoricals_w[1])
                    res += str(count).center(categoricals_w[2])
                    res += str(weight).center(categoricals_w[3]) + '\n'
                res += '-' * max_categoricals_table + '\n'
                res += '\n'
        return res


functions = ['log', 'exp', 'lag', 'dif', 'gr', 'sum', 'count', 'mean', 'std', 'min', 'max']
operators = ['+', '-', '*', '/', '^', '=', '!=', '<', '>', '<=', '>=']
operators_levels = [['+', '-'], ['*', '/'], ['^'], ['=','!=','<', '>', '<=', '>=']]
likes = ['linear', 'function', 'operator']
class Formula:
    # formula: age + age ^ 2 - height/weight + log(year) 
    def __init__(self, formula:str) -> None:
        self.formula = formula

    def __str__(self) -> str:
        return self.formula

    #region split
    @staticmethod
    def __remove_braces(text:str)->str:
        if text != '':
            text = text.replace(' ', '')
            opened_braces = 0
            for i, w in enumerate(text):
                if w == '(':
                    opened_braces += 1
                elif w == ')':
                    opened_braces -= 1
                if opened_braces == 0:
                    break
            if text[0] == '(' and i == len(text)-1 and text[-1]==')':
                return text[1:-1]
            else:
                return text

    @staticmethod
    def __split_operators(text:str, operators:list[str])->list[str]:
        # print(text, operators)
        formula = Formula.__remove_braces(text)
        #region remove parenthesis in the start and the end
        if formula[0]=='(' and (not ')' in formula[1:-1]):
            if formula[-1]==')':
                formula = formula[1:-1]
        #endregion
        #region '-'
        if formula[0]=='-' and len(formula)>2:
            formula = '0'+formula
        #endregion
        #region split
        splits = []
        expr, in_braces = '', False
        i = 0
        while i < len(formula):
            w = formula[i]
            if w == '(':
                in_braces = True            # start of expersion
                # print(i, w, expr)
                expr += w
            elif in_braces and w == ')':
                in_braces = False           # end of expersion
                expr += w
            else:
                if not in_braces:           # end of operator
                    if formula[i:i+2] in operators:
                        if expr != '':
                            splits.extend([expr, formula[i:i+2]])
                            # print(i, 2, w, expr)
                            expr = ''
                            i += 1
                    elif w in operators:
                        if expr != '':
                            splits.extend([expr, w])
                            # print(i, 1, w, expr)
                            expr = ''
                    else:
                        expr += w
                else:
                    expr += w
            i += 1
        if expr != '':
            splits.append(expr)
        #endregion
        #region grouping into 3 parts
        if len(splits) == 2:
            raise ValueError(f"Error! maybe the number of operators is not currect.")
        elif len(splits) == 3:
            if not splits[1] in operators:
                raise ValueError(f"Error! operator must be between values.")
        elif len(splits) > 3:
            if (len(splits) + 1) % 2 == 0:
                while len(splits)!=3:
                    if not splits[1] in operators:
                        raise ValueError(f"Error! operators must be between values.")
                    splits = [splits[:3]] + splits[3:]
            else:
                raise ValueError(f"Error! maybe the number of operators is not currect.") 
        #endregion
        return splits

    @staticmethod
    def __split_functions(text:str, functions:list[str], data:Data)->list[str]:
        formula = Formula.__remove_braces(text)
        for f in functions:
            if len(formula) > len(f):
                if formula[:len(f)] == f and formula[len(f)] == '(' and formula[-1] == ')':
                    arguments = formula[len(f)+1:-1]
                    if not arguments in data.variables():
                        split = Formula.__split_functions(arguments, functions, data)
                        if split == []:
                            return [f, *formula[len(f)+1:-1].split(',')]
                        else:
                            return [f, split]
                    else:
                        return [f, arguments]
        return []

    @staticmethod
    def __split(formulas:list, data:Data)->list:
        is_deeping = True
        while is_deeping:
            is_deeping = False
            for i in range(len(formulas)):
                if not formulas[i] in data.variables():
                    for p in operators_levels:
                        if type(formulas[i]) == list:
                            formulas[i] = Formula.__split(formulas[i], data)
                        else:
                            splits = Formula.__split_operators(formulas[i], p)
                            if len(splits) > 1:
                                is_deeping = is_deeping or True
                                formulas[i] = splits
                    if type(formulas[i]) == list:
                        formulas[i] = Formula.__split(formulas[i], data)
                    else:
                        splits = Formula.__split_functions(formulas[i], functions, data)
                        if len(splits) > 1:
                            is_deeping = is_deeping or True
                            formulas[i] = splits
        return formulas

    @staticmethod
    def __is_end_level(splits:list)->bool:
        for i in splits:
            if type(i) == list:
                return False
        return True

    @staticmethod
    def __name_data(left: str, operator: str, right: str):
        var_name = ''
        var_name += f"({left})" if max(
            [operator in left for operator in ['+','-','=','!=','<','>','<=','>=']]) and \
            operator in ['*', '/', '^'] else f"{left}"
        var_name += f"{operator}"
        var_name += f"({right})" if max(
            [operator in right for operator in ['+','-','=','!=','<','>','<=','>=']]) and \
            operator in ['*', '/', '^'] else f"{right}"
        return var_name

    @staticmethod
    def __include_text(variable, data):
        if type(variable) != Data:
            if not is_numeric_str(variable):
                if not variable in data.variables():
                    return True
        return False

    @staticmethod
    def __calculate(splits:list, data:Data, weights:str='1', skip_collinear:bool=False)->Data:
        if Formula.__is_end_level(splits):
            if splits[0] in functions:
                res = {}
                if splits[0] in ['log', 'exp', 'sum', 'count', 'mean', 'std', 'min', 'max']:
                    if len(splits)!=2:
                        raise ValueError(f"Error! function {splits[0]} has a argument, but there are more or less than one arguments.")
                    if not type(splits[1]) == Data:
                        if not is_numeric_str(splits[1]):
                            if not splits[1] in data.variables():
                                raise ValueError(f"Error! variable '{splits[1]} is not in data.' ")
                    
                    if type(splits[1]) == Data:
                        s, s2, n, ws, minimum, maximum = {}, {}, {}, {}, {}, {}
                        for var  in splits[1].variables():
                            s[var], s2[var], n[var], ws[var], minimum[var], maximum[var]= 0,0,0,0,0,0
                            res[f'{splits[0]}({var})']={}
                    elif is_numeric_str(splits[1]):
                        s, s2, n, ws, minimum, maximum = 0,0,0,0,0,0
                        res[f'{splits[0]}({splits[1]})'] = {}
                    else: 
                        s, s2, n, ws, minimum, maximum = {}, {}, {}, {}, {}, {}
                        variable = Variable.from_data(data, splits[1])
                        if variable.type == Variable_Types.categorical:
                            vals = variable.values(Sample(data))[:-1] if skip_collinear else variable.values(Sample(data))
                            for val in vals:
                                s[f'{splits[1]}={val}'], s2[f'{splits[1]}={val}'] = 0,0
                                n[f'{splits[1]}={val}'], ws[f'{splits[1]}={val}'] = 0,0
                                minimum[f'{splits[1]}={val}'], maximum[f'{splits[1]}={val}']= 0,0
                                res[f'{splits[0]}({splits[1]}={val})'] = {}
                        elif variable.type == Variable_Types.numeric:
                            s, s2, n, ws, minimum, maximum = 0,0,0,0,0,0
                            res[f'{splits[0]}({splits[1]})'] = {}

                    if splits[0] in ['sum', 'count', 'mean', 'std', 'min', 'max']:
                        if weights != '1':
                            if not weights in data.variables():
                                raise ValueError(f"Error! variable {weights} as waights, is not in data.")
                        start = True
                        for i in data.index():
                            w = 1 if weights == '1' else data.values[weights][i]
                            if type(splits[1]) == Data:
                                for var  in splits[1].variables():
                                    x = splits[1].values[var][i]
                                    is_nan = False
                                    if is_numeric(x):
                                        if np.isnan(x):
                                            is_nan = True
                                    if not is_nan:
                                        s[var] += w * x
                                        s2[var] += w * x**2
                                        n[var] += 1
                                        ws[var] += w
                                        if start:
                                            minimum[var] = x
                                            maximum[var] = x
                                            start = False
                                        else:
                                            minimum[var] = min(minimum[var], x)
                                            maximum[var] = max(maximum[var], x)
                                if start:
                                    start = False
                            elif is_numeric_str(splits[1]):
                                s += w * float(splits[1])
                                s2 += w * float(splits[1])**2
                                n += 1
                                ws += w
                                if start:
                                    minimum = float(splits[1])
                                    maximum = float(splits[1])
                                    start = False
                                else:
                                    minimum = min(minimum, float(splits[1]))
                                    maximum = max(maximum, float(splits[1]))
                            else:
                                variable = Variable.from_data(
                                    data, splits[1])
                                if variable.type == Variable_Types.categorical:
                                    vals = variable.values(Sample(data))[:-1] if skip_collinear else variable.values(Sample(data))
                                    for val in vals:
                                        x = data.values[splits[1]][i]
                                        if is_numeric(x):
                                            if np.isnan(x):
                                                break
                                        n[f"{splits[1]}={val}"] += 1
                                        ws[f"{splits[1]}={val}"] += w
                                        if x == val:
                                            s[f"{splits[1]}={val}"] += w
                                            s2[f"{splits[1]}={val}"] += w
                                elif variable.type == Variable_Types.numeric:
                                    x = data.values[splits[1]][i]
                                    is_nan = False
                                    if is_numeric(x):
                                        if np.isnan(x):
                                            is_nan = True
                                    if not is_nan:
                                        s += w * x
                                        s2 += w * x**2
                                        n += 1
                                        ws += w
                                        if start:
                                            minimum = x
                                            maximum = x
                                            start = False
                                        else:
                                            minimum = min(minimum, x)
                                            maximum = max(maximum, x)
                            

                    for i in data.index():
                        if splits[0] == 'log':
                            if type(splits[1]) == Data:
                                for var in splits[1].variables():
                                    if splits[1].values[var][i]>0:
                                        res[f'{splits[0]}({var})'][i] = math.log(splits[1].values[var][i])
                                    else:
                                        res[f'{splits[0]}({var})'][i] = np.nan
                            elif is_numeric_str(splits[1]):
                                for var in res.keys():
                                    if float(splits[1])>0:
                                        res[var][i] = math.log(float(splits[1]))
                                    else:
                                        res[var][i] =np.nan
                            else:
                                variable = Variable.from_data(
                                    data, splits[1])
                                if variable.type == Variable_Types.categorical:
                                    for var in res.keys():
                                        if data.values[variable.name][i] == var.split('=')[1][:-1]:
                                            res[var][i] = 0
                                        else:
                                            res[var][i] = np.nan
                                elif variable.type == Variable_Types.numeric:
                                    for var in res.keys():
                                        if data.values[splits[1]][i]>0:
                                            res[var][i] = math.log(data.values[splits[1]][i])
                                        else:
                                            res[var][i] =np.nan
                        elif splits[0] == 'exp':
                            if type(splits[1]) == Data:
                                for var in splits[1].variables():
                                    try:
                                        res[f'{splits[0]}({var})'][i] = math.exp(
                                                splits[1].values[var][i])
                                    except:
                                        res[f'{splits[0]}({var})'][i] =np.nan
                            elif is_numeric_str(splits[1]):
                                for var in res.keys():
                                    try:
                                        res[var][i] = math.exp(float(splits[1]))
                                    except:
                                        res[var][i] = np.nan
                            else:
                                variable = Variable.from_data(
                                    data, splits[1])
                                if variable.type == Variable_Types.categorical:
                                    for var in res.keys():
                                        if data.values[variable.name][i] == var.split('=')[1][:-1]:
                                            res[var][i] = math.exp(1)
                                        else:
                                            res[var][i] = 1
                                elif variable.type == Variable_Types.numeric:
                                    for var in res.keys():
                                        try:
                                            res[var][i] = math.exp(
                                                data.values[splits[1]][i])
                                        except:
                                            res[var][i] = np.nan
                        elif splits[0] in ['sum', 'count', 'mean', 'std', 'min', 'max']:
                            if type(splits[1]) == Data:
                                for var  in splits[1].variables():
                                    x = splits[1].values[var][i]
                                    is_nan = False
                                    if is_numeric(x):
                                        if np.isnan(x):
                                            res[f'{splits[0]}({var})'][i] = np.nan
                                            is_nan = True
                                    if not is_nan:
                                        if splits[0] == 'sum':
                                            res[f'{splits[0]}({var})'][i] = s[var]
                                        elif splits[0] == 'count':
                                            res[f'{splits[0]}({var})'][i] = n[var]
                                        elif splits[0] == 'mean':
                                            if ws[var] == 0:
                                                res[f'{splits[0]}({var})'][i] = np.nan
                                            else:
                                                res[f'{splits[0]}({var})'][i] = s[var]/ws[var]
                                        elif splits[0] == 'std':
                                            if ws[var] == 0:
                                                res[f'{splits[0]}({var})'][i] = np.nan
                                            else:
                                                res[f'{splits[0]}({var})'][i] = ((s2[var]-s[var]**2/ws[var])/ws[var])**0.5
                                        elif splits[0] == 'min':
                                            res[f'{splits[0]}({var})'][i] = minimum[var]
                                        elif splits[0] == 'max':
                                            res[f'{splits[0]}({var})'][i] = maximum[var]
                            elif is_numeric_str(splits[1]):
                                if splits[0] == 'sum':
                                    res[f'{splits[0]}({splits[1]})'][i] = s
                                elif splits[0] == 'count':
                                    res[f'{splits[0]}({splits[1]})'][i] = n
                                elif splits[0] == 'mean':
                                    if ws == 0:
                                        res[f'{splits[0]}({splits[1]})'][i] = np.nan
                                    else:
                                        res[f'{splits[0]}({splits[1]})'][i] = s/ws
                                elif splits[0] == 'std':
                                    if ws == 0:
                                        res[f'{splits[0]}({splits[1]})'][i] = np.nan
                                    else:
                                        res[f'{splits[0]}({splits[1]})'][i] = ((s2-s**2/ws)/ws)**0.5
                                elif splits[0] == 'min':
                                    res[f'{splits[0]}({splits[1]})'][i] = minimum
                                elif splits[0] == 'max':
                                    res[f'{splits[0]}({splits[1]})'][i] = maximum
                            else:
                                variable = Variable.from_data(
                                    data, splits[1])
                                if variable.type == Variable_Types.categorical:
                                    vals = variable.values(Sample(data))[:-1] if skip_collinear else variable.values(Sample(data))
                                    for val in vals:
                                        is_nan = False
                                        x = data.values[splits[1]][i]
                                        if is_numeric(x):
                                            if np.isnan(x):
                                                res[f'{splits[0]}({splits[1]}={val})'][i] = np.nan
                                                is_nan = True
                                        if not is_nan:
                                            if splits[0] == 'sum':
                                                res[f'{splits[0]}({splits[1]}={val})'][i] = s[f'{splits[1]}={val}']
                                            elif splits[0] == 'count':
                                                res[f'{splits[0]}({splits[1]}={val})'][i] = n[f'{splits[1]}={val}']
                                            elif splits[0] == 'mean':
                                                if ws[f'{splits[1]}={val}'] == 0:
                                                    res[f'{splits[0]}({splits[1]}={val})'][i] = np.nan
                                                else:
                                                    res[f'{splits[0]}({splits[1]}={val})'][i] = s[f'{splits[1]}={val}']/ws[f'{splits[1]}={val}']
                                            elif splits[0] == 'std':
                                                if ws[f'{splits[1]}={val}'] == 0:
                                                    res[f'{splits[0]}({splits[1]}={val})'][i] = np.nan
                                                else:
                                                    res[f'{splits[0]}({splits[1]}={val})'][i] = ((s2[f'{splits[1]}={val}']-s[f'{splits[1]}={val}']**2/ws[f'{splits[1]}={val}'])/ws[f'{splits[1]}={val}'])**0.5
                                            elif splits[0] == 'min':
                                                res[f'{splits[0]}({splits[1]}={val})'][i] = minimum[f'{splits[1]}={val}']
                                            elif splits[0] == 'max':
                                                res[f'{splits[0]}({splits[1]}={val})'][i] = maximum[f'{splits[1]}={val}']
                                elif variable.type == Variable_Types.numeric:
                                    is_nan = False
                                    x = data.values[splits[1]][i]
                                    if is_numeric(x):
                                        if np.isnan(x):
                                            res[f'{splits[0]}({splits[1]})'][i] = np.nan
                                            is_nan = True
                                    if not is_nan:
                                        if splits[0] == 'sum':
                                            res[f'{splits[0]}({splits[1]})'][i] = s
                                        elif splits[0] == 'count':
                                            res[f'{splits[0]}({splits[1]})'][i] = n
                                        elif splits[0] == 'mean':
                                            if ws == 0:
                                                res[f'{splits[0]}({splits[1]})'][i] = np.nan
                                            else:
                                                res[f'{splits[0]}({splits[1]})'][i] = s/ws
                                        elif splits[0] == 'std':
                                            if ws == 0:
                                                res[f'{splits[0]}({splits[1]})'][i] = np.nan
                                            else:
                                                res[f'{splits[0]}({splits[1]})'][i] = ((s2-s**2/ws)/ws)**0.5
                                        elif splits[0] == 'min':
                                            res[f'{splits[0]}({splits[1]})'][i] = minimum
                                        elif splits[0] == 'max':
                                            res[f'{splits[0]}({splits[1]})'][i] = maximum
                        
                elif splits[0] in ['lag', 'dif', 'gr']:
                    if data.type != Data_Types.time:
                        raise ValueError(f"Error! function '{splits[0]}'' only uses for data type of 'time'.")
                    if len(splits) != 3:
                        raise ValueError(
                            f"Error! function '{splits[0]}'' has two argument, but there are more or less than two arguments.")
                    if not type(splits[1]) == Data:
                        if not splits[1] in data.variables():
                            raise ValueError(
                                f"Error! variable '{splits[1]}' is not in data.")
                    if type(splits[2])==Data:
                        raise ValueError(
                            f"Error! arguments '{splits[1]}' must be a number as a lag.")
                    elif not splits[2].isdigit():
                        raise ValueError(
                            f"Error! arguments '{splits[1]}' must be a number as a lag.")
                    if not min([is_numeric(i) for i in data.index()]):
                        raise ValueError(
                            f"Error! for time data, index must be numeric.")
                    
                    if not type(splits[1]) == Data:
                        variable = Variable.from_data(
                            data, splits[1])
                        if variable.type == Variable_Types.categorical:
                            values_set = variable.values(Sample(data))[:-1] if skip_collinear else variable.values(Sample(data))
                            for val in values_set:
                                res[f'{splits[0]}({splits[1]}={val},{splits[2]})'] = {}
                        elif variable.type == Variable_Types.numeric:
                            res[f'{splits[0]}({splits[1]},{splits[2]})'] = {}
                    else:
                        for var in splits[1].variables():
                            res[f'{splits[0]}({var},{splits[2]})'] = {}

                    lag = int(splits[2])
                    for i in data.index():
                        if type(splits[1])==Data:
                            for var in splits[1].variables():
                                if splits[0] == 'lag':
                                    if i >= lag:
                                        if i-lag in data.index():
                                            res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                                splits[1].values[var][i-lag]
                                        else:
                                            res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                                np.nan
                                    else:
                                        res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                            np.nan
                                elif splits[0] == 'dif':
                                    if i >= lag:
                                        if i-lag in splits[1].values[var]:
                                            res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                                splits[1].values[var][i] - \
                                                splits[1].values[var][i-lag]
                                        else:
                                            res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                                np.nan
                                    else:
                                        res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                            np.nan
                                elif splits[0] == 'gr':
                                    if i >= lag:
                                        if i-lag in splits[1].values[var]:
                                            res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                                splits[1].values[var][i] / \
                                                splits[1].values[var][i-lag]-1
                                        else:
                                            res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                                np.nan
                                    else:
                                        res[f'{splits[0]}({var},{splits[2]})'][i] = \
                                            np.nan
                        else:
                            variable = Variable.from_data(data, splits[1])
                            if variable.type == Variable_Types.categorical:
                                if i >= lag:
                                    if i-lag in data.index():
                                        for var in res.keys():
                                            val = var.split(
                                                '=')[1].split(',')[0]
                                            if splits[0] == 'lag':
                                                if str(data.values[splits[1]][i-lag]) == val:
                                                    res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = 1
                                                else:
                                                    res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = 0
                                            elif splits[0] == 'dif':
                                                if str(data.values[splits[1]][i]) == val:
                                                    if str(data.values[splits[1]][i-lag]) == val:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = 0
                                                    else:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = 1
                                                else:
                                                    if str(data.values[splits[1]][i-lag]) == val:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = -1
                                                    else:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = 0
                                            elif splits[0] == 'gr':
                                                if str(data.values[splits[1]][i]) == val:
                                                    if str(data.values[splits[1]][i-lag]) == val:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = 0
                                                    else:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = np.nan
                                                else:
                                                    if str(data.values[splits[1]][i-lag]) == val:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = -1
                                                    else:
                                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = np.nan
                                    else:
                                        for var in res.keys():
                                            val = var.split('=')[1].split(',')[0]
                                            res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = \
                                            np.nan
                                else:
                                    for var in res.keys():
                                        val = var.split('=')[1].split(',')[0]
                                        res[f'{splits[0]}({splits[1]}={val},{splits[2]})'][i] = \
                                            np.nan
                            elif variable.type == Variable_Types.numeric:
                                if i >= lag:
                                    if i-lag in data.index():
                                        if splits[0] == 'lag':
                                            res[f'{splits[0]}({splits[1]},{splits[2]})'][i] = \
                                                data.values[splits[1]][i-lag]
                                        elif splits[0] == 'dif':
                                            res[f'{splits[0]}({splits[1]},{splits[2]})'][i] = \
                                                data.values[splits[1]][i] - \
                                                data.values[splits[1]][i-lag]
                                        elif splits[0] == 'gr':
                                            res[f'{splits[0]}({splits[1]},{splits[2]})'][i] = \
                                                data.values[splits[1]][i] / \
                                                data.values[splits[1]][i-lag] - 1 
                                    else:
                                        res[f'{splits[0]}({splits[1]},{splits[2]})'][i] = \
                                            np.nan
                                else:
                                    res[f'{splits[0]}({splits[1]},{splits[2]})'][i] = \
                                        np.nan
                return Data(data.type, res)
            else:
                if len(splits)==3:
                    var_left, operator, var_right = splits
                    # Errors
                    if not operator in operators:
                        raise ValueError(f"Error! {operator} is not in operators (+,-,*,/,^)")
                    if not operator in ['=','!=']:   # if left or right are or include text -> Error!
                        if Formula.__include_text(var_left, data) or Formula.__include_text(var_right, data):
                            raise ValueError(
                                f"Error! {operator} only works on numbers, and cannot work on text values.")

                    # initials
                    res = {}
                    if type(var_left) == Data:
                        if type(var_right) == Data:
                            for v_left in var_left.values:
                                for v_right in var_right.values:
                                    res[Formula.__name_data(v_left, operator, v_right)] = {}
                        elif is_numeric_str(var_right):
                            for v_left in var_left.values:
                                res[Formula.__name_data(v_left, operator, var_right)] = {}
                        else:
                            if var_right in data.variables(): # var_right == a variable in sample
                                variable_right = Variable.from_data(
                                    data, var_right)
                                if variable_right.type == Variable_Types.categorical:
                                    vals_right = variable_right.values(Sample(data))[:-1] if skip_collinear else variable_right.values(Sample(data))
                                    for v_left in var_left.values:
                                        for val_right in vals_right:
                                            res[Formula.__name_data(v_left, operator, f'{var_right}={val_right}')] = {}
                                elif variable_right.type == Variable_Types.numeric:
                                    for v_left in var_left.values:
                                        res[Formula.__name_data(v_left, operator, var_right)] = {}
                            else:  # var_right == text value, operator == '=' or '!='
                                for v_left in var_left:
                                    res[Formula.__name_data(v_left, operator, var_right)] = {}
                    elif is_numeric_str(var_left):
                        if type(var_right) == Data:
                            for v_right in var_right.variables():
                                res[Formula.__name_data(var_left, operator, v_right)] = {}
                        elif is_numeric_str(var_right):
                            res[Formula.__name_data(var_left, operator, var_right)] = {}
                        else:
                            variable_right = Variable.from_data(
                                data, var_right)
                            if variable_right.type == Variable_Types.categorical:
                                vals_right = variable_right.values(Sample(data))[:-1] if skip_collinear else variable_right.values(Sample(data))
                                for val_right in vals_right:
                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')] = {}
                            elif variable_right.type == Variable_Types.numeric:
                                res[Formula.__name_data(var_left, operator, var_right)] = {}
                    else:
                        if var_left in data.variables():     # var_left == variable in sample
                            variable_left = Variable.from_data(data, var_left)
                            if variable_left.type == Variable_Types.categorical:
                                vals_left = variable_left.values(Sample(data))[:-1] if skip_collinear else variable_left.values(Sample(data))
                                if type(var_right) == Data:
                                    for val_left in vals_left:
                                        for v_right in var_right.values:
                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)] = {}
                                elif is_numeric_str(var_right):
                                    for val_left in vals_left:
                                        res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)] = {}
                                else:
                                    if var_right in data.variables():    # var_right == a variable in sample
                                        variable_right = Variable.from_data(
                                            data, var_right)
                                        if variable_right.type == Variable_Types.categorical:
                                            vals_right = variable_right.values(Sample(data))[:-1] if skip_collinear else variable_right.values(Sample(data))
                                            for val_left in vals_left:
                                                for val_right in vals_right:
                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')] = {}
                                        elif variable_right.type == Variable_Types.numeric:
                                            for val_left in vals_left:
                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)] = {}
                                    else:   # var_right == a text value, operator == '=' or '!='
                                        res[Formula.__name_data(var_left, operator, var_right)] = {}
                            elif variable_left.type == Variable_Types.numeric:
                                if type(var_right) == Data:
                                    for v_right in var_right.values:
                                        res[Formula.__name_data(var_left, operator, v_right)] = {}
                                elif is_numeric_str(var_right):
                                    res[Formula.__name_data(var_left, operator, var_right)] = {}
                                else:
                                    variable_right = Variable.from_data(
                                        data, var_right)
                                    if variable_right.type == Variable_Types.categorical:
                                        vals_right = variable_right.values(Sample(data))[:-1] if skip_collinear else variable_right.values(Sample(data))
                                        for val_right in vals_right:
                                            res[Formula.__name_data(var_left, operator, f'({var_right}={val_right})')] = {}
                                    elif variable_right.type == Variable_Types.numeric:
                                        res[Formula.__name_data(var_left, operator, var_right)] = {}
                        else:   # var_left == a text value
                            if type(var_right) == Data:
                                for v_right in var_right.variables():
                                    res[Formula.__name_data(var_left, operator, v_right)] = {}
                            elif not is_numeric_str(var_right):
                                res[Formula.__name_data(var_left, operator, var_right)] = {}
                                
                    ## calculates
                    for i in data.index():
                        if type(var_left) == Data:
                            if type(var_right) == Data:
                                for v_left in var_left.values:
                                    for v_right in var_right.values:
                                        if operator == '+':
                                            res[Formula.__name_data(v_left, operator, v_right)][i] = var_left.values[v_left][i] + \
                                                var_right.values[v_right][i]
                                        elif operator == '-':
                                            res[Formula.__name_data(v_left, operator, v_right)][i] = var_left.values[v_left][i] - \
                                                var_right.values[v_right][i]
                                        elif operator == '*':
                                            res[Formula.__name_data(v_left, operator, v_right)][i] = var_left.values[v_left][i] * \
                                                var_right.values[v_right][i]
                                        elif operator == '/':
                                            if var_right.values[v_right][i] != 0:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = var_left.values[v_left][i] / \
                                                    var_right.values[v_right][i]
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = np.nan
                                        elif operator == '^':
                                            res[Formula.__name_data(v_left, operator, v_right)][i] = var_left.values[v_left][i] ** \
                                                var_right.values[v_right][i]
                                        elif np.isnan(var_left.values[v_left][i]) or np.isnan(var_right.values[v_right][i]):
                                            res[Formula.__name_data(v_left, operator, v_right)][i] = np.nan
                                        elif operator == '=':
                                            if var_left.values[v_left][i] == var_right.values[v_right][i]:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 0
                                        elif operator == '!=':
                                            if var_left.values[v_left][i] != var_right.values[v_right][i]:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 0
                                        elif operator == '<':
                                            if var_left.values[v_left][i] < var_right.values[v_right][i]:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 0
                                        elif operator == '>':
                                            if var_left.values[v_left][i] > var_right.values[v_right][i]:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 0
                                        elif operator == '<=':
                                            if var_left.values[v_left][i] <= var_right.values[v_right][i]:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 0
                                        elif operator == '>=':
                                            if var_left.values[v_left][i] >= var_right.values[v_right][i]:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(v_left, operator, v_right)][i] = 0
                            elif is_numeric_str(var_right):
                                for v_left in var_left.values:
                                    if operator == '+':
                                        res[Formula.__name_data(v_left, operator, var_right)][i] = var_left.values[v_left][i] + float(var_right)
                                    elif operator == '-':
                                        res[Formula.__name_data(v_left, operator, var_right)][i] = var_left.values[v_left][i] - float(var_right)
                                    elif operator == '*':
                                        res[Formula.__name_data(v_left, operator, var_right)][i] = var_left.values[v_left][i] * float(var_right)
                                    elif operator == '/':
                                        if float(var_right) != 0:
                                            res[Formula.__name_data(v_left, operator, var_right)][i] = var_left.values[v_left][i] / float(var_right)
                                        else:
                                            res[Formula.__name_data(v_left, operator, var_right)][i] = np.nan
                                    elif operator == '^':
                                        res[Formula.__name_data(v_left, operator, var_right)][i] = var_left.values[v_left][i] ** float(var_right)
                                    elif np.isnan(var_left.values[v_left][i]):
                                        res[Formula.__name_data(v_left, operator, var_right)][i] = np.nan
                                    elif operator == '=':
                                        if var_left.values[v_left][i] == float(var_right):
                                            res[Formula.__name_data(
                                            v_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 0
                                    elif operator == '!=':
                                        if var_left.values[v_left][i] != float(var_right):
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 0
                                    elif operator == '<':
                                        if var_left.values[v_left][i] < float(var_right):
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 0
                                    elif operator == '>':
                                        if var_left.values[v_left][i] > float(var_right):
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 0
                                    elif operator == '<=':
                                        if var_left.values[v_left][i] <= float(var_right):
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 0
                                    elif operator == '>=':
                                        if var_left.values[v_left][i] >= float(var_right):
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(
                                                v_left, operator, var_right)][i] = 0
                            else:
                                if var_right in data.variables(): # var_right == a variable in sample
                                    variable_right = Variable.from_data(data, var_right)
                                    if variable_right.type == Variable_Types.categorical:
                                        vals_right = variable_right.values(Sample(data))[:-1] if skip_collinear else variable_right.values(Sample(data))
                                        for v_left in var_left.values:
                                            for val_right in vals_right:
                                                if not is_numeric(data.values[var_right][i]):
                                                    if data.values[var_right][i] == val_right:
                                                        if operator == '+':
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                var_left.values[v_left][i] + 1
                                                        elif operator == '-':
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                var_left.values[v_left][i] - 1
                                                        elif operator in ['*', '/', '^']:
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                var_left.values[v_left][i]
                                                        elif np.isnan(var_left.values[v_left][i]):
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = np.nan
                                                        elif operator == '=':
                                                            if var_left.values[v_left][i] == 1:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                0
                                                        elif operator == '!=':
                                                            if var_left.values[v_left][i] != 1:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                0
                                                        elif operator == '<':
                                                            if var_left.values[v_left][i] < 1:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                0
                                                        elif operator == '>':
                                                            if var_left.values[v_left][i] > 1:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                0
                                                        elif operator == '<=':
                                                            if var_left.values[v_left][i] <= 1:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                0
                                                        elif operator == '>=':
                                                            if var_left.values[v_left][i] >= 1:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                0
                                                    else:
                                                        if operator in ['+', '-']:
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                var_left.values[v_left][i]
                                                        elif operator == '*':
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = 0
                                                        elif operator == '/':
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                np.nan
                                                        elif operator == '^':
                                                            res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = 1
                                                        elif operator == '=':
                                                            if var_left.values[v_left][i] == 0:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    0
                                                        elif operator == '!=':
                                                            if var_left.values[v_left][i] != 0:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    0
                                                        elif operator == '<':
                                                            if var_left.values[v_left][i] < 0:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    0
                                                        elif operator == '>':
                                                            if var_left.values[v_left][i] > 0:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    0
                                                        elif operator == '<=':
                                                            if var_left.values[v_left][i] <= 0:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    0
                                                        elif operator == '>=':
                                                            if var_left.values[v_left][i] >= 0:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    1
                                                            else:
                                                                res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = \
                                                                    0
                                                else:
                                                    res[Formula.__name_data(v_left, operator, f'({var_right}={val_right})')][i] = np.nan
                                    elif variable_right.type == Variable_Types.numeric:
                                        for v_left in var_left.variables():
                                            if operator == '+':
                                                res[Formula.__name_data(v_left, operator, var_right)][i] = \
                                                    var_left.values[v_left][i] + \
                                                        data.values[var_right][i]
                                            elif operator == '-':
                                                res[Formula.__name_data(v_left, operator, var_right)][i] = \
                                                    var_left.values[v_left][i] - \
                                                        data.values[var_right][i]
                                            elif operator == '*':
                                                res[Formula.__name_data(v_left, operator, var_right)][i] = \
                                                    var_left.values[v_left][i] * \
                                                        data.values[var_right][i]
                                            elif operator == '/':
                                                if data.values[var_right][i] != 0:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = \
                                                        var_left.values[v_left][i] / \
                                                            data.values[var_right][i]
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = np.nan
                                            elif operator == '^':
                                                res[Formula.__name_data(v_left, operator, var_right)][i] = \
                                                    var_left.values[v_left][i] ** \
                                                        data.values[var_right][i]
                                            elif np.isnan(var_left.values[v_left][i]) or np.isnan(data.values[var_right][i]):
                                                res[Formula.__name_data(v_left, operator, var_right)][i] = np.nan
                                            elif operator == '=':
                                                if var_left.values[v_left][i] == data.values[var_right][i]:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 0
                                            elif operator == '!=':
                                                if var_left.values[v_left][i] != data.values[var_right][i]:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 0
                                            elif operator == '<':
                                                if var_left.values[v_left][i] < data.values[var_right][i]:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 0
                                            elif operator == '>':
                                                if var_left.values[v_left][i] > data.values[var_right][i]:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 0
                                            elif operator == '<=':
                                                if var_left.values[v_left][i] <= data.values[var_right][i]:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 0
                                            elif operator == '>=':
                                                if var_left.values[v_left][i] >= data.values[var_right][i]:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)][i] = 0
                                else: # var_right == a text value, operator == '=' or '!='
                                    for v_left in var_left:
                                        if is_numeric(var_left.values[v_left][i]):
                                            if np.isnan(var_left.values[v_left][i]):
                                                res[Formula.__name_data(v_left, operator, var_right)] = np.nan
                                        else:
                                            if operator == '=':
                                                if var_left.values[v_left][i] == var_right:
                                                    res[Formula.__name_data(v_left, operator, var_right)] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)] = 0
                                            elif operator == '!=':
                                                if var_left.values[v_left][i] != var_right:
                                                    res[Formula.__name_data(v_left, operator, var_right)] = 1
                                                else:
                                                    res[Formula.__name_data(v_left, operator, var_right)] = 0
                        elif is_numeric_str(var_left):
                            if type(var_right) == Data:
                                for v_right in var_right.values:
                                    if operator == '+':
                                        res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                            float(var_left) + var_right.values[v_right][i]
                                    elif operator == '-':
                                        res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                            float(var_left) - \
                                            var_right.values[v_right][i]
                                    elif operator == '*':
                                        res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                            float(var_left) * \
                                            var_right.values[v_right][i]
                                    elif operator == '/':
                                        if var_right.values[v_right][i] != 0:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                                float(var_left) / \
                                                var_right.values[v_right][i]
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = np.nan
                                    elif operator == '^':
                                        res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                            float(var_left) ** \
                                            var_right.values[v_right][i]
                                    elif np.isnan(var_right.values[v_right][i]):
                                        res[Formula.__name_data(var_left, operator, v_right)][i] = np.nan
                                    elif operator == '=':
                                        if float(var_left) == var_right.values[v_right][i]:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                    elif operator == '!=':
                                        if float(var_left) != var_right.values[v_right][i]:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                    elif operator == '<':
                                        if float(var_left) < var_right.values[v_right][i]:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                    elif operator == '>':
                                        if float(var_left) > var_right.values[v_right][i]:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                    elif operator == '<=':
                                        if float(var_left) <= var_right.values[v_right][i]:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                    elif operator == '>=':
                                        if float(var_left) >= var_right.values[v_right][i]:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                            elif is_numeric_str(var_right):
                                if operator == '+':
                                    res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                        float(var_left) + float(var_right)
                                elif operator == '-':
                                    res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                        float(var_left) - float(var_right)
                                elif operator == '*':
                                    res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                        float(var_left) * float(var_right)
                                elif operator == '/':
                                    if float(var_right) != 0:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                            float(var_left) / float(var_right)
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                elif operator == '^':
                                    res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                        float(var_left) ** float(var_right)
                                elif operator == '=':
                                    if float(var_left) == float(var_right):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                elif operator == '!=':
                                    if float(var_left) != float(var_right):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                elif operator == '<':
                                    if float(var_left) < float(var_right):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                elif operator == '>':
                                    if float(var_left) > float(var_right):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                elif operator == '<=':
                                    if float(var_left) <= float(var_right):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                elif operator == '>=':
                                    if float(var_left) >= float(var_right):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                    else:
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                            else:
                                variable_right = Variable.from_data(
                                    data, var_right)
                                if variable_right.type == Variable_Types.categorical:
                                    for val_right in vals_right:
                                        if not is_numeric(data.values[var_right][i]):
                                            if is_numeric(data.values[var_right][i]):
                                                if np.isnan(data.values[var_right][i]):
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = np.nan
                                            elif data.values[var_right][i] == val_right:
                                                if operator == '+':
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                        float(var_left) + 1
                                                elif operator == '-':
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                        float(var_left) - 1
                                                elif operator in ['*', '/', '^']:
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                        float(var_left)
                                                elif operator == '=':
                                                    if float(var_left) == 1:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '!=':
                                                    if float(var_left) != 1:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '<':
                                                    if float(var_left) < 1:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '>':
                                                    if float(var_left) > 1:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '<=':
                                                    if float(var_left) <= 1:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '>':
                                                    if float(var_left) >= 1:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                            else:
                                                if operator in ['+', '-']:
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                        float(var_left)
                                                elif operator == '*':
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '/':
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                        np.nan
                                                elif operator == '^':
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                elif operator == '=':
                                                    if float(var_left) == 0:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '!=':
                                                    if float(var_left) != 0:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '<':
                                                    if float(var_left) < 0:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '>':
                                                    if float(var_left) > 0:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '<=':
                                                    if float(var_left) <= 0:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                elif operator == '>':
                                                    if float(var_left) >= 0:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                    else:
                                                        res[Formula.__name_data(
                                                            var_left, operator, f'{var_right}={val_right}')][i] = 0
                                        else:
                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = np.nan
                                elif variable_right.type == Variable_Types.numeric:
                                    if operator == '+':
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                            float(var_left) + \
                                                data.values[var_right][i]
                                    elif operator == '-':
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                            float(var_left) - \
                                                data.values[var_right][i]
                                    elif operator == '*':
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                            float(var_left) * \
                                                data.values[var_right][i]
                                    elif operator == '/':
                                        if data.values[var_right][i] != 0:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                float(var_left) / \
                                                    data.values[var_right][i]
                                        else:
                                            data.values[var_right][i] = np.nan
                                    elif operator == '^':
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                            float(var_left) ** \
                                                data.values[var_right][i]
                                    elif np.isnan(data.values[var_right][i]):
                                        res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                    elif operator == '=':
                                        if float(var_left) == data.values[var_right][i]:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    elif operator == '!=':
                                        if float(var_left) != data.values[var_right][i]:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    elif operator == '<':
                                        if float(var_left) < data.values[var_right][i]:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    elif operator == '>':
                                        if float(var_left) > data.values[var_right][i]:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    elif operator == '<=':
                                        if float(var_left) <= data.values[var_right][i]:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    elif operator == '>=':
                                        if float(var_left) >= data.values[var_right][i]:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                        else:
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                        else:
                            if var_left in data.variables():  # var_left == a variable in sample
                                variable_left = Variable.from_data(data, var_left)
                                if variable_left.type == Variable_Types.categorical:
                                    if type(var_right) == Data:
                                        for val_left in vals_left:
                                            for v_right in var_right.values:
                                                if not is_numeric(data.values[var_left][i]):
                                                    if data.values[var_left][i] == val_left:
                                                        if operator == '+':
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = \
                                                                var_right.values[v_right][i] + 1
                                                        elif operator == '-':
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = \
                                                                var_right.values[v_right][i] - 1
                                                        elif operator in ['*', '/', '^']:
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = \
                                                                var_right.values[v_right][i]
                                                        elif operator == '=':
                                                            if var_right.values[v_right][i] == 1:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '!=':
                                                            if var_right.values[v_right][i] != 1:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '<':
                                                            if var_right.values[v_right][i] < 1:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '>':
                                                            if var_right.values[v_right][i] > 1:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '<=':
                                                            if var_right.values[v_right][i] <= 1:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '>=':
                                                            if var_right.values[v_right][i] >= 1:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                    else:
                                                        if operator in ['+', '-']:
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = \
                                                                var_right.values[v_right][i]
                                                        elif operator == '*':
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '/':
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = np.nan
                                                        elif operator == '^':
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                        elif operator == '=':
                                                            if var_right.values[v_right][i] == 0:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '!=':
                                                            if var_right.values[v_right][i] != 0:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '<':
                                                            if var_right.values[v_right][i] < 0:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '>':
                                                            if var_right.values[v_right][i] > 0:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '<=':
                                                            if var_right.values[v_right][i] <= 0:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                        elif operator == '>=':
                                                            if var_right.values[v_right][i] >= 0:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 1
                                                            else:
                                                                res[Formula.__name_data(
                                                                    f'({var_left}={val_left})', operator, v_right)][i] = 0
                                                else:
                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, v_right)][i] = np.nan
                                    elif is_numeric_str(var_right):
                                        for val_left in vals_left:
                                            if not is_numeric(data.values[var_left][i]):
                                                if data.values[var_left][i] == val_left:
                                                    if operator == '+':
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                            1 + float(var_right)
                                                    elif operator == '-':
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                            1 - float(var_right)
                                                    elif operator == '*':
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                            1 * float(var_right)
                                                    elif operator == '/':
                                                        if float(var_right) != 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                                1/float(var_right)
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = np.nan
                                                    elif operator == '^':
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                            1
                                                    elif operator == '=':
                                                        if float(var_right) == 1:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '!=':
                                                        if float(var_right) != 1:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '<':
                                                        if float(var_right) < 1:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '>':
                                                        if float(var_right) > 1:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '<=':
                                                        if float(var_right) <= 1:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '>=':
                                                        if float(var_right) >= 1:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                else:
                                                    if operator == '+':
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                            float(var_right)
                                                    elif operator == '-':
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = \
                                                            -float(var_right)
                                                    elif operator in ['*', '/', '^']:
                                                        res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '=':
                                                        if float(var_right) == 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '!=':
                                                        if float(var_right) != 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '<':
                                                        if float(var_right) < 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '>':
                                                        if float(var_right) > 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '<=':
                                                        if float(var_right) <= 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                                    elif operator == '>=':
                                                        if float(var_right) >= 0:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 1
                                                        else:
                                                            res[f"({var_left}={val_left}){operator}{var_right}"][i] = 0
                                            else:
                                                res[f"({var_left}={val_left}){operator}{var_right}"][i] = np.nan
                                    else:
                                        if var_right in data.variables():    # var_right == a variable in sample
                                            variable_right = Variable.from_data(data, var_right)
                                            if variable_right.type == Variable_Types.categorical:
                                                for val_left in vals_left:
                                                    for val_right in vals_right:
                                                        if (not is_numeric(data.values[var_left][i])) and \
                                                            (not is_numeric(data.values[var_right][i])):
                                                            if data.values[var_left][i] == val_left and \
                                                                    data.values[var_right][i] == val_right:
                                                                if operator == '+':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 2
                                                                elif operator == '-':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                                elif operator in ['*', '/', '^']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                                elif operator in ['=', '<=', '>=']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                                elif operator in ['!=', '<', '>']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                            elif data.values[var_left][i] == val_left and \
                                                                    data.values[var_right][i] != val_right:
                                                                if operator in ['+', '-', '^']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                                elif operator == '*':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                                elif operator == '/':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = \
                                                                        np.nan
                                                                elif operator in ['=', '<=', '>=']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                                elif operator in ['!=', '<', '>']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                            elif data.values[var_left][i] != val_left and \
                                                                    data.values[var_right][i] == val_right:
                                                                if operator == '+':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                                elif operator == '-':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = -1
                                                                elif operator == ['*','/','^']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                                elif operator in ['=', '<=', '>=']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                                elif operator in ['!=', '<', '>']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                if operator in ['+','-','*']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                                elif operator == '/':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = np.nan
                                                                elif operator == '^':
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                                elif operator in ['=', '<=', '>=']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 1
                                                                elif operator in ['!=', '<', '>']:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = 0
                                                        else:
                                                            res[Formula.__name_data(f'({var_left}={val_left})', operator, f'{var_right}={val_right}')][i] = np.nan
                                            elif variable_right.type == Variable_Types.numeric:
                                                for val_left in vals_left:
                                                    if not is_numeric(data.values[var_left][i]):
                                                        if data.values[var_left][i] == val_left:
                                                            if operator == '+':
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = \
                                                                    1 + data.values[var_right][i]
                                                            elif operator == '-':
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = \
                                                                    1 - data.values[var_right][i]
                                                            elif operator == '*':
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = \
                                                                    data.values[var_right][i]
                                                            elif operator == '/':
                                                                if data.values[var_right][i] != 0:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = \
                                                                        1 / data.values[var_right][i]
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = np.nan
                                                            elif operator == '^':
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                            elif operator == '=':
                                                                if 1 == data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '!=':
                                                                if 1 != data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '<':
                                                                if 1 < data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '>':
                                                                if 1 > data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '<=':
                                                                if 1 <= data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '>=':
                                                                if 1 >= data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                        else:
                                                            if operator == '+':
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = \
                                                                    data.values[var_right][i]
                                                            elif operator == '-':
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = \
                                                                    -data.values[var_right][i]
                                                            elif operator in ['*', '/', '^']:
                                                                res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '=':
                                                                if 0 == data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '!=':
                                                                if 0 != data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '<':
                                                                if 0 < data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '>':
                                                                if 0 > data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '<=':
                                                                if 0 <= data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                            elif operator == '>=':
                                                                if 0 >= data.values[var_right][i]:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 1
                                                                else:
                                                                    res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = 0
                                                    else:
                                                        res[Formula.__name_data(f'({var_left}={val_left})', operator, var_right)][i] = np.nan
                                        else:   # var_right == a text value, operator == '=', '!='
                                            for val_left in vals_left:
                                                if is_numeric(data.values[var_left][i]):
                                                    if np.isnan(data.values[var_left][i]):
                                                        res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                                else:
                                                    if operator == '=':
                                                        if data.values[var_left][i] == var_right:
                                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                        else:
                                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                                    elif operator == '!=':
                                                        if data.values[var_left][i] != var_right:
                                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                        else:
                                                            res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                elif variable_left.type == Variable_Types.numeric:
                                    if type(var_right) == Data:
                                        for v_right in var_right.values:
                                            if operator == '+':
                                                res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                                    data.values[var_left][i] + var_right.values[v_right][i]
                                            elif operator == '-':
                                                res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                                    data.values[var_left][i] - \
                                                    var_right.values[v_right][i]
                                            elif operator == '*':
                                                res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                                    data.values[var_left][i] * var_right.values[v_right][i]
                                            elif operator == '/':
                                                if var_right.values[v_right][i] != 0:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                                        data.values[var_left][i] / \
                                                        var_right.values[v_right][i]
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = np.nan
                                            elif operator == '^':
                                                res[Formula.__name_data(var_left, operator, v_right)][i] = \
                                                    data.values[var_left][i] ** \
                                                    var_right.values[v_right][i]
                                            elif np.isnan(data.values[var_left][i]) or np.isnan(var_right.values[v_right][i]):
                                                res[Formula.__name_data(var_left, operator, v_right)][i] = np.nan
                                            elif operator == '=':
                                                if data.values[var_left][i] == var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                            elif operator == '!=':
                                                if data.values[var_left][i] != var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                            elif operator == '<':
                                                if data.values[var_left][i] < var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                            elif operator == '<':
                                                if data.values[var_left][i] > var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                            elif operator == '<=':
                                                if data.values[var_left][i] <= var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                            elif operator == '>=':
                                                if data.values[var_left][i] >= var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                    elif is_numeric_str(var_right):
                                        if operator == '+':
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                data.values[var_left][i] + float(var_right)
                                        elif operator == '-':
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                data.values[var_left][i] - float(var_right)
                                        elif operator == '*':
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                data.values[var_left][i] * float(var_right)
                                        elif operator == '/':
                                            if float(var_right) != 0:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                    data.values[var_left][i] / float(var_right)
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                        elif operator == '^':
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                data.values[var_left][i] ** \
                                                float(var_right)
                                        elif np.isnan(data.values[var_left][i]):
                                            res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                        elif operator == '=':
                                            if data.values[var_left][i] == float(var_right):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                        elif operator == '!=':
                                            if data.values[var_left][i] != float(var_right):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                        elif operator == '<':
                                            if data.values[var_left][i] < float(var_right):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                        elif operator == '>':
                                            if data.values[var_left][i] > float(var_right):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                        elif operator == '<=':
                                            if data.values[var_left][i] <= float(var_right):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                        elif operator == '>=':
                                            if data.values[var_left][i] >= float(var_right):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    else:
                                        variable_right = Variable.from_data(
                                            data, var_right)
                                        if variable_right.type == Variable_Types.categorical:
                                            for val_right in vals_right:
                                                if not is_numeric(data.values[var_right][i]):
                                                    if data.values[var_right][i] == val_right:
                                                        if operator == '+':
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                                data.values[var_left][i] + 1
                                                        elif operator == '-':
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                                data.values[var_left][i] - 1
                                                        elif operator in ['*', '/', '^']:
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                                data.values[var_left][i]
                                                        elif operator == '=':
                                                            if data.values[var_left][i] == 1:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '!=':
                                                            if data.values[var_left][i] != 1:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '<':
                                                            if data.values[var_left][i] < 1:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '>':
                                                            if data.values[var_left][i] > 1:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '<=':
                                                            if data.values[var_left][i] <= 1:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '>=':
                                                            if data.values[var_left][i] >= 1:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                    else:
                                                        if operator in ['+', '-']:
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = \
                                                                data.values[var_left][i]
                                                        elif operator == '*':
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '/':
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = np.nan
                                                        elif operator == '^':
                                                            res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                        elif operator == '=':
                                                            if data.values[var_left][i] == 0:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '!=':
                                                            if data.values[var_left][i] != 0:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '<':
                                                            if data.values[var_left][i] < 0:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '>':
                                                            if data.values[var_left][i] > 0:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '<=':
                                                            if data.values[var_left][i] <= 0:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                        elif operator == '>=':
                                                            if data.values[var_left][i] >= 0:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 1
                                                            else:
                                                                res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = 0
                                                else:
                                                    res[Formula.__name_data(var_left, operator, f'{var_right}={val_right}')][i] = np.nan
                                        elif variable_right.type == Variable_Types.numeric:
                                            if operator == '+':
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                    data.values[var_left][i] + \
                                                        data.values[var_right][i]
                                            elif operator == '-':
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                    data.values[var_left][i] - \
                                                    data.values[var_right][i]
                                            elif operator == '*':
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                    data.values[var_left][i] * \
                                                        data.values[var_right][i]
                                            elif operator == '/':
                                                if data.values[var_right][i] != 0:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                        data.values[var_left][i] / \
                                                            data.values[var_right][i]
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                            elif operator == '^':
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = \
                                                    data.values[var_left][i] ** \
                                                    data.values[var_right][i]
                                            elif np.isnan(data.values[var_left][i]) or np.isnan(data.values[var_right][i]):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                            elif operator == '=':
                                                if data.values[var_left][i] == data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                            elif operator == '!=':
                                                if data.values[var_left][i] != data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                            elif operator == '<':
                                                if data.values[var_left][i] < data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                            elif operator == '>':
                                                if data.values[var_left][i] > data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                            elif operator == '<=':
                                                if data.values[var_left][i] <= data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                            elif operator == '>=':
                                                if data.values[var_left][i] >= data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                            else:   # var_left == a text value, operator == '=', '!='
                                if type(var_right) == Data:
                                    for v_right in var_right.variables():
                                        if is_numeric(var_right.values[v_right][i]):
                                            if np.isnan(var_right.values[v_right][i]):
                                                res[Formula.__name_data(var_left, operator, v_right)][i] = np.nan
                                        else:
                                            if operator == '=':
                                                if var_left == var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                            elif operator == '!=':
                                                if var_left != var_right.values[v_right][i]:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, v_right)][i] = 0
                                if not is_numeric_str(var_right):
                                    if var_right in data.variables():
                                        if is_numeric(data.values[var_right][i]):
                                            if np.isnan(data.values[var_right][i]):
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = np.nan
                                        else:
                                            if operator == '=':
                                                if var_left == data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                            elif operator == '!=':
                                                if var_left != data.values[var_right][i]:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                                else:
                                                    res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                    else:
                                        if operator == '=':
                                            if var_left == var_right:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                                        elif operator == '!=':
                                            if var_left != var_right:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 1
                                            else:
                                                res[Formula.__name_data(var_left, operator, var_right)][i] = 0
                    return Data(data.type, res)
                else:
                    raise ValueError(
                        f"Error! {splits} cannot be decomposed into operators and functions.")
        else:
            for i, split in enumerate(splits):
                if type(split) == list:
                    splits[i] = Formula.__calculate(split, data, weights, skip_collinear)
            return Formula.__calculate(splits, data, weights, skip_collinear)
    #endregion

    def calculate(self, data:Data, weights:str='1', skip_collinear:bool=False)->Data:
        splits = Formula.__split([self.formula], data)[0]
        if type(splits) != list:
            if splits in data.variables():
                variable = Variable.from_data(data, splits)
                if variable.type == Variable_Types.numeric:
                    data = data.select_variables([variable.name])
                    data = data.select_index(data.index())
                    return data
                elif variable.type == Variable_Types.categorical:
                    values = variable.values(Sample(data))[:-1] if skip_collinear else variable.values(Sample(data))
                    vars = {}
                    for value in values:
                        vars[f'{variable.name}={value}'] = {}
                    for i in data.index():
                        val_i = data.values[variable.name][i]
                        for value in values:
                            if is_numeric(val_i):
                                if np.isnan(val_i):
                                    vars[f'{variable.name}={value}'][i] = np.nan
                            elif val_i == value:
                                vars[f'{variable.name}={value}'][i] = 1
                            else:
                                vars[f'{variable.name}={value}'][i] = 0
                    return Data(data.type, vars)
            else:
                if is_numeric_str(splits):
                    values = {}
                    for i in data.index():
                        values[i] = float(splits)
                    return Data(data.type, {splits:values})
                elif splits[0] == '-' and splits[1:] in data.variables():
                    variable = Variable.from_data(data, splits[1:])
                    if variable.type == Variable_Types.numeric:
                        values = {}
                        for i in data.index():
                            values[i] = -data.values[splits[1:]]
                        return Data(data.type, {splits: values})
                    elif variable.type == Variable_Types.categorical:
                        values = variable.values(Sample(data))[:-1] if skip_collinear else variable.values(Sample(data))
                        vars = {}
                        for value in values:
                            vars[f'-{variable.name}={value}'] = {}
                        for i in data.index():
                            val_i = data.values[variable.name][i]
                            for value in values:
                                if is_numeric(val_i):
                                    if np.isnan(val_i):
                                        vars[f'-{variable.name}={value}'][i] = np.nan
                                elif val_i == value:
                                    vars[f'-{variable.name}={value}'][i] = -1
                                else:
                                    vars[f'-{variable.name}={value}'][i] = 0
                        return Data(data.type, vars)
                else:
                    raise ValueError(f"Error! variable '{splits}' is not in data.")
        return Formula.__calculate(splits, data, weights, skip_collinear)
        
    def split(self)->Formulas:
        formulas = []
        self_formula = self.formula.replace(' ', '')
        in_braces = False
        formula, i = '', 0
        while i < len(self_formula):
            w = self_formula[i]
            if w == '(':
                in_braces = True
            elif in_braces and w == ')':
                in_braces = False
            elif w == '-' and not in_braces:
                if i>1:
                    self_formula = self_formula[:i] + '+' + self_formula[i:]
                    break
                    i += 1
            i += 1
        formulas_plus = self_formula.split('+')
        for formula in formulas_plus:
            formulas.append(formula)
        return Formulas(formulas)

    def filter(self, value:str|int|float, data:Data)->Data:
        calculate = self.calculate(data)
        calc_var = calculate.variables()[0]
        values = {}
        for var in data.variables():
            values[var] = {}
        for i in data.index():
            if calculate.values[calc_var][i] == value:
                for var in data.variables():
                    values[var][i] = data.values[var][i]
        return Data(data.type, values)

class Formulas:
    def __init__(self, formulas:list[str]) -> None:
        self.formulas = formulas
    
    def calculate_all(self, data:Data, weights:str='1', skip_collinear:bool=False)->Data:
        res = Data(data.type, {})
        for formula in self.formulas:
            res.add_data(Formula(formula).calculate(data, weights, skip_collinear))
        return res


class Table:
    def __init__(self, based_variable:str|dict|Variable, formulas:list[str]|Formulas, sample:Sample) -> None:
        if type(based_variable) == str:
            self.based_variable = Variable.from_data(sample.data, based_variable)
        elif type(based_variable) == dict:
            self.based_variable = Variable.from_dict(based_variable)
        else:
            self.based_variable = based_variable
        # furmulas must begin with 'sum' 'count 'mean' 'std' 'min' or 'max'. 
        # otherwise add all of them to start of all formulas.
        if type(formulas) == list:
            for formula in formulas:
                has_func = False
                for func in ['sum', 'count', 'mean', 'std', 'min', 'max']:
                    if func == formula[:len(func)]:
                        has_func = True
                        break
                if not has_func:
                    formulas.remove(formula)
                    for func in ['sum', 'count', 'mean', 'std', 'min', 'max']:
                        formulas.append(f"{func}({formula})")
            self.formulas = Formulas(formulas)
        else:
            self.formulas = formulas
        self.sample = sample

    def to_data(self, weights='1', skip_collinear:bool=False)->Data:
        data_calc = self.sample.data.select_variables([self.based_variable.name])
        for formula in self.formulas.formulas:
            for func in ['sum', 'count', 'mean', 'std', 'min', 'max']:
                if func == formula[:len(func)]:
                    data_calc.add_data(Formula(formula[len(func)+1:-1]).calculate(self.sample.data, weights, skip_collinear))
        data_values = {}
        values = self.based_variable.values(self.sample)
        if self.based_variable.type == 'categorical':
            for value in values:
                data = Formula(self.based_variable.name + '=' + value).filter(1, data_calc)
                calculations = self.formulas.calculate_all(data, weights, skip_collinear)
                for var in calculations.variables():
                    if not var in data_values.keys(): 
                        data_values[var] = {}
                    try:
                        data_values[var][value] = Variable(var).stats.mean(Sample(calculations))
                    except:
                        data_values[var][value] = np.nan
        elif self.based_variable.type == 'numeric':
            values.sort()
            for value in values:
                data = Formula(self.based_variable.name).filter(value, data_calc)
                calculations = self.formulas.calculate_all(data, weights, skip_collinear)
                for var in calculations.variables():
                    if not var in data_values.keys(): 
                        data_values[var] = {}
                    try:
                        data_values[var][value] = Variable(var).stats.mean(Sample(calculations))
                    except:
                        data_values[var][value] = np.nan
        return Data(self.sample.data.type, data_values)

    def __str__(self) -> str:
        return self.to_data()

    def plot(self, weights='1', skip_collinear:bool=False)->None:
        data = self.to_data(weights, skip_collinear)
        import matplotlib.pyplot as plt
        n = len(data.variables())
        bar_width = 1/(n+1)

        x = [i+1 for i in range(len(data))]
        x_ticks = data.index()

        for j, var in enumerate(data.variables()):
            y = [val for _, val in data.values[var].items()]
            plt.bar([i+j*bar_width for i in x],y, label=var, width=bar_width)

        plt.xticks([i+(n-1)*bar_width/2 for i in x], x_ticks)
        plt.xlabel(self.based_variable.name)
        plt.legend()
        plt.show()


