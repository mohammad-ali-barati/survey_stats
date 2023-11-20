from __future__ import annotations
from survey_stats.functions import *
from survey_stats.data_process import *
import math, time
from copy import copy

class Model:
    '''
    Model is a conceptual structure of data.\n
    A variable is a concept that has two general types: 'numeric' and 'categorical'.\n
    A model is a set of several interrelated variables on a data.\n
    on a same data, we can define several models.\n
    for example:\n
    - personality model can include psychological test and behaviors.\n
    - consumtion model can include pesonality test and behaviors.\n
    Observations include indexs that the values of all model variables are non-nan.\n
    We can increase the number of observations by reducing the variables. In this case, the quality of the model decreases but its statistical accuracy increases.\n
    Variables can also be divided into 'dependent' and 'independent' categories according to central and influential concepts.\n
    Causal relationships are either defined based on a theory and then tested with data, or they are inferred from the data.\n
    Also, the variables can be divided into more homogeneous groups based on their correlation.\n
    In basic model, we only infer using pairwise correlation coefficients.\n
    More complex inferences will be placed in specialized statistical modules.\n
    '''
    def __init__(self, data: Data, dep_vars:list=[], indep_vars:list=[], name:str=None) -> None:
        self.data = data
        if dep_vars != []:
            self.dep_vars = [v for v in dep_vars if v in data.variables()]
        else:
            self.dep_vars = dep_vars
        if indep_vars!=[]:
            self.indep_vars = [v for v in indep_vars if v in data.variables()]
        else:
            self.indep_vars = data.variables()
        self.name = name
        self.all_vars = self.dep_vars + self.indep_vars

    def __str__(self) -> str:
        res = ''
        for dep in self.dep_vars:
            res += f'{dep} = f('
            for i, indep in enumerate(self.indep_vars):
                res += f',{indep}' if i>0 else f'{indep}'
            res += ')\n'
        return res

    def observation(self)->Sample:
        pass

    def increase_obs(self):
        pass

    def correls(self, weighted:bool=False, print_progress:bool=True, indent:int=0, subindent:int=5):
        return Sample(self.data).stats.correl(self.all_vars, weighted=weighted, print_progress=print_progress, indent=indent, subindent=subindent)

    def r2(self, dep_var:str, indep_vars:list[str])->float:
        '''
        for a dep_var 'y' and indep_vars 'x':\n
        - ri = correlation of y and xi\n
        - rij = correlation of xi and xj\n
        matrix:
        - ryx = [r1 ... rm]T (m*1)\n
        - rxx = [r11 ...r1m/.../rm1...rmm] (m*m)\n
        r2 = ryxT . rxx-1 . ryx\n
        if rxx = I then r2 = r1^2 + ... + rm^2\n
        source: https://stats.stackexchange.com/questions/314926/can-you-calculate-r2-from-correlation-coefficents-in-multiple-linear-regressi\n
        https://en.wikipedia.org/wiki/Coefficient_of_multiple_correlation\n
        when the number of observations is lower than the number of independent variables, estimation of multiple regression is imposible, But r2 enables us to estimate the multiple determination coefficient based on partial correlation coefficients and identify the best independent variables based on the effect on r2.\n
        ***** problem larg variables -> larg matrix -> time-intensive calculations ******
        '''

        pass

    def grouping(self, correls:dict, initials_number:int, min_correl:float,
                 max_groups:int,
                 excel_file:str='',
                           print_progress:bool=False, indent:int=0, subindent:int=5)->list[str]:
        '''
        grouping variables to k groups base on the absolute value of the pairwise correlation coeficients.\n
        1- The distance between two variables is equal to 1 minus their correlation coefficient.\n
        2- The distance a variable from each group is equal to the average distance from each member of each group.\n
        3- First Identifying k variables that have the greatest distance from each other.\n
        4- Then each variable is added to the closest group.\n
        .\n
        Note: The order of entering the variables cannot affect the result due to the method of finding the initial values which are the farthest points from each other.\n
        '''
        if print_progress:
            print(' '*indent+f'grouping variables to {initials_number} groups')
        sorted_correls = sorted([(v1,v2,correls[v1][v2])
                                    for v1 in correls
                                        for v2 in correls[v1]],
                                key=lambda x:abs(x[2]))
        def distance(var:str, other:str|list[str])->float:
            if isinstance(other, str):
                return 1-abs(correls[var][other])
            else:
                return 1-sum([abs(correls[var][v]) for v in other])/len(other)
        
        def furthests(total_vars:list[str], p:float=2, min_i:int=200):
            def max_distance(vars:list[str], group:list[str])->str:
                    '''The furthest point from a group'''
                    max_dist = 0
                    for var in vars:
                        if (dist:=distance(var, group))>max_dist:
                            max_var, max_dist = var, dist
                    return max_var
            def furthests(group:list[str], vars:list[str])->list[str]:
                ''''initials_number' number of furthest points'''
                j = len(group)
                while j<initials_number:
                    group.append(v3:=max_distance(vars, group))
                    j += 1
                    vars.remove(v3)
                d, n= 0, 0
                for i, v1 in enumerate(group[:-1]):
                    for v2 in group[i+1:]:
                        d += distance(v1, [v2])
                        n += 1
                return group, d/n
            max_d = 0
            i, start, prelog, n = 0, time.perf_counter(), 0, len(sorted_correls)
            for v1, v2, _ in sorted_correls:
                vars = total_vars.copy()
                vars.remove(v1)
                vars.remove(v2)
                initials, d = furthests([v1 , v2], vars)
                if d>max_d:
                    max_initials, max_d, i_max = initials, d, i+1
                i += 1
                if i/i_max>p and i>min_i:
                    break
                if print_progress:
                    prelog = progress(start, i-1, max(p*i_max, min_i), prelog, f'i_max: {i_max}', indent=indent+subindent)
            return max_initials

        if print_progress:
            print(' '*indent+'finding initial points (farthest points)')
        initials = furthests(self.all_vars)

        def nearest(var:str, groups:list[list[str]])->str:
            min_d = float('inf')
            for g in groups:
                if (d:=distance(var, g))<min_d:
                    min_d, min_g = d, g
            return min_d, min_g

        def grouping(vars:list[str], initials:list[str]):
            groups = [[v] for v in initials]
            others = set()
            for var in vars.copy():
                if not var in initials:
                    min_d, min_g = nearest(var, groups)
                    if min_d<=1-min_correl:
                        min_g.append(var)
                    else:
                        others.add(var)
            while True:
                groupeds = 0
                for var in others:
                    min_d, min_g = nearest(var, groups)
                    if min_d<=1-min_correl:
                        min_g.append(var)
                        others.remove(var)
                        groupeds += 1
                if groupeds == 0:
                    break
            return groups, others
        
        if print_progress:
            print(' '*indent+'grouping')
        groups, others = grouping(self.all_vars, initials)
        def group_center(group:list[str])->str:
            d = []
            for var in group:
                d.append((var, distance(var, group)))
            d.sort(key=lambda x:x[1])
            return d[0][0]
        
        while len(groups)<=max_groups:
            center_others = group_center(others)
            new_groups, others = grouping(others, [center_others])
            if len(new_groups)==0:
                break
            groups.extend(new_groups)

        def similarity(g1:list[str], g2:list[str]):
            return len([v for v in g1 if v in g2])*2/(len(g1) + len(g2))
        def to_excel(groups:list[list[str]]):
            for i, g in enumerate(groups):
                res = Data(values={'correl_to_center':{}})
                for v in g:
                    res.values['correl_to_center'][v] = correls[group_center(g)][v]
                res.to_excel(excel_file, sheet=f'group_{i+1}')
            sims = Data(values = {f'g_{i}':{} for i in range(len(groups))})
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    sims.values[f'g_{i}'][f'g_{j}'] = similarity(g1, g2)
            sims.to_excel(excel_file, sheet='similarity',
                          print_progress=print_progress,
                          indent=indent+subindent,
                          subindent=subindent)
        if excel_file != '':
            to_excel(groups)
        
        return groups

class Formula:
    '''
    Formula is a expersion of mathematic operators and functions that can calculate on a data.\n
    for example:
    - formula: age + age**2 - exp(height/weight) + log(year)\n
    operators: all operators on python.\n
    - '+', '-', '*', '/', '//', '**', '%', '==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'not', 'is', 'is not', 'in', 'not in'.\n
    functions: all functions on 'math' madule.\n
    - 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 
    'ceil', 'comb', 'copysign', 'cos', 'cosh', 'degrees', 'dist', 
    'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor',
     'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose',
     'isfinite', 'isinf', 'isnan', 'isqrt', 'lcm', 'ldexp', 'lgamma',
     'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'nextafter',
     'perm', 'pi', 'pow', 'prod', 'radians', 'remainder', 'sin',
     'sinh', 'sqrt', 'tan', 'tanh', 'tau', 'trunc', 'ulp'
    '''
    def __init__(self, formula:str, name:str='') -> None:
        self.formula = formula
        if name=='':
            name = formula
        self.name = name

    def __str__(self) -> str:
        if self.name != '':
            return f'{self.name} = {self.formula}'
        else:
            return self.formula

    def split(self)->Formulas:
        '''
        spliting formula base of seprator '+'.\n
        for example:\n
        - 'log(x+exp(y))+z+sin(y)' -> Formulas(['log(x+exp(y))', 'z', 'sin(y)']).\n
        '''
        formulas = []
        self_formula = self.formula.strip()
        in_braces = False
        formula, i = '', 0
        while i < len(self_formula):
            w = self_formula[i]
            if w == '(':
                in_braces = True
                formula += w
            elif in_braces and w == ')':
                in_braces = False
                formula += w
            elif in_braces:
                formula += w
            elif not(w == '+' or in_braces):
                formula += w
            elif w == '+' and not in_braces:
                formulas.append(formula.strip())
                formula = ''
            i += 1
        formulas.append(formula.strip())
        return Formulas(formulas)

    def filter(self, value:any, data:Data)->Data:
        '''
        select indexs that that formula equal to value.\n
        for example:\n
        - 'formula' is "(sex=='male')and(age<30)" and 'value' is True.\n
        - 'formula' is 'education' (a variable on data) and 'value' is 'pdh' (a value of 'education').\n
        '''
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

    def calculate(self, data:Data, print_progress:bool=False, indent:int=0, subindent:int=5):
        if print_progress:
            print(f'calculating {self.name} for data of {data.index()[0]}...{data.index()[-1]}')
            start, prelog, n = time.perf_counter(), 0, len(data.index())
        res = {self.name:{}}
        vars = data.variables()
        vars.sort(key=lambda v: len(v), reverse=True)
        for j, i in enumerate(data.index()):
            f = copy(self.formula)
            for var in vars:
                if var in f and not f"'{var}'" in f and not f'"{var}"' in f:
                    if is_nan(val:=data.values[var][i]):
                        res[self.name][i] = math.nan
                        break
                    if isinstance(val, str):
                        f = f.replace(var, f"'{val}'")
                    else:
                        f = f.replace(var, str(val))
            else:
                for func in [x for x in dir(math) if not '__' in x]:
                    if func in f:
                        f.replace(func, f'math.{func}')
                try:
                    res[self.name][i] = eval(f)
                except:
                    res[self.name][i] = math.nan
            if print_progress:
                prelog = progress(start, j, n, prelog, i, indent=indent+subindent)
        return Data(values = res)

class Formulas:
    '''A list of Formula.'''
    def __init__(self, formulas:list[str], names:list[str]=[]) -> None:
        self.formulas = formulas
        self.names = names if names!=[] else formulas
    
    def calculate_all(self, data:Data,
                    print_progress:bool=False, indent:int=0, subindent:int=5)->Data:
        res = Data(data.type, values={})
        if print_progress:
            print(' '*indent+'calculating furmulas:')
            start_time, prelog, n = time.perf_counter(), 0, len(self.formulas)
        for i, formula in enumerate(self.formulas):
            try:
                res.add_data(Formula(formula, self.names[i]).calculate(data))
            except Exception as e:
                if print_progress:
                    print(f'Error in {formula}. {e}')
            if print_progress:
                prelog = progress(start_time, i, n, prelog, indent=indent+subindent)
        return res

    def __str__(self) -> str:
        return '\n'.join(f'{i+1}- {f}' for i, f in enumerate(self.formulas))
