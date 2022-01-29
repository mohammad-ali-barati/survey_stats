from __future__ import annotations
import random
from typing import Union
import numpy as np

from functions import *

class Data_Types:
    cross = 'cross'
    time = 'time'
    panel = 'panel'

class Data:
    """
    type: 'cross', 'time', 'panel'
    """
    def __init__(self, type:str='cross', values:dict[dict]={}) -> None:
        if type == Data_Types.cross or type == Data_Types.time \
                or type == Data_Types.panel:
            self.type = type
        else:
            self.type = None
        self.values = values
    
    def __str__(self) -> str:
        #region columns width
        width = 0
        vars = ['index']
        if len(self.variables())>5:
            vars.extend(self.variables()[:3])
            vars.append('...')
            vars.extend(self.variables()[-2:])
        else:
            vars.extend(self.variables())
        if len(self.index()) > 15:
            inds = self.index()[:5]
            inds.append('⁝')
            inds.extend(self.index()[-3:])
        else:
            inds = self.index()
        for i in inds:
            for var in vars:
                if i != '⁝':
                    if var != 'index' and var != '...' :
                        val = str(self.values[var][i])
                    else:
                        val = var
                else:
                    val = '⁝'
                if is_numeric(val):
                    val = str(round(val,4))
                if width<len(str(var)) or width<len(val):
                    width = max(len(str(var)), len(val))
        width = min(width, 20)
        #endregion
        #region title
        title = 'type: ' + self.type + '\n'
        title += ' ' + '-'*(len(vars)*(width+1)-1) + '\n'
        title += ''
        for var in vars:
            title += '|' + str(var)[:20].center(width)
        title += '|\n'
        title += ' ' + '-'*(len(vars)*(width+1)-1) + '\n'
        #endregion
        #region rows
        rows = title
        for i in inds:
            for var in vars:
                if i != '⁝':
                    if var == 'index':
                        rows += '|' + str(i)[:20].center(width)
                    elif var == '...':
                        rows += '|' + '...'.center(width)
                    else:
                        x = self.values[var][i]
                        if is_numeric(x):
                            x = str(round(x,4))
                        rows += '|' + str(x)[:20].center(width)
                else:
                    if var == '...':
                        rows += '|' +str('˙·.')[:20]
                    else:
                        rows += '|' + str(i)[:20].center(width)
            rows += '|\n'
        rows +=  ' ' + '-'*(len(vars)*(width+1)-1) + '\n'
        rows += f'{len(self.index())} × {len(self.variables())}'
        #endregion
        return rows

    def variables(self) -> None:
        return [var for var in self.values]

    def index(self, without_checking:bool=True) -> None:
        if without_checking:
            for v in self.values.keys():
                return [i for i in self.values[v].keys()]
        else:
            start, ind = True, []
            for v in self.values.keys():
                if start:
                    ind = [i for i in self.values[v].keys()]
                    start = False
                else:
                    ind2 = [i for i in self.values[v].keys()]
                    if ind != ind2:
                        raise ValueError("Error! index aren't same for all variables.")
            return ind
    
    def set_index(self, var:str, drop_var:bool=True) -> None:
        new_dict = {}
        if var in self.values.keys():
            for v in self.values.keys():
                if not (v == var and drop_var):
                    new_dict[v] = {self.values[var][i]:self.values[v][i] for i in self.values[var].keys()}
            self.values = new_dict
        else:
            raise ValueError(f"Error! {var} is not in variables of data.")

    def set_names(self, new_names:list[str]=[], old_names:list[str]=[]):
        new_dict,i = {},0
        for v in self.values.keys():
            if (v in old_names) or old_names==[]:
                new_dict[new_names[i]] = self.values[v]
                i+=1
            else:
                new_dict[v] = self.values[v]
        self.values = new_dict

    def select_variables(self, vars:list[str]=[]) -> Data:
        if type(vars) != list:
            raise ValueError(f"Error! {vars} is not a list of variables.")
        if vars!=[]:
            new_dict = {}
            for var in vars:
                if var in self.values.keys():
                    new_dict[var] = self.values[var]
            return Data(self.type,new_dict)

    def select_index(self,index:list)->Data:
        if type(index) != list:
            raise ValueError(f"Error! {index} is not a list of index.")
        vars = self.variables()
        res_dict = {}
        for var in vars:
            values = {}
            for i in index:
                if i in self.index():
                    values[i] = self.values[var][i]
            res_dict[var] = values
        return Data(self.type, res_dict)

    def drop(self, var_names:list[str]):
        if type(var_names) != list:
            raise ValueError(f"Error! {var_names} is not a list of variables.")
        for var in var_names:
            if var in self.values.keys():
                self.values.pop(var)

    def add_a_dummy(self, condition:list[list[tuple]], add_to_data:bool=False)->None:
        # condition = [('sex','=','female'), ('age','<',20)]
        # names =          sex=female_and_age<20
        dummy_values = {}
        for i in self.index():
            satisfied, is_nan = True, False
            for var, sign,val in condition:
                try:
                    nan = np.isnan(self.values[var][i])
                except:
                    nan = False
                if not nan:
                    satisfied = satisfied and check_condition(self.values[var][i], sign, val)
                else:
                    is_nan = True
                    break
            if satisfied:
                dummy_values[i] = 1
            elif not is_nan:
                dummy_values[i] = 0
            else:
                dummy_values[i] = np.nan
        start = True
        for var, sign, val in condition:
            if start:
                dummy_name = var + sign + str(val)
                start = False
            else:
                dummy_name += '_and_' + var + sign + str(val)
        res = {}
        res[dummy_name] = dummy_values
        if add_to_data:
            self.values.update(res)
        return Data(self.type, res)
            
    def add_dummies(self, conditions:list[dict], add_to_data:bool=False)->None:
        # conditions = [[('sex','=','female'),('age','<',20)],[()],[()],...,[()]]
        #               |___________a condition_____________| |__| |__|     |__|
        # names =                 sex=female_age<20
        values = {}
        for cond in conditions:
            values.update(self.add_a_dummy(cond, add_to_data).values)
        return Data(self.type, values)

    def dropna(self, vars:list[str]=[])->None:
        for i in self.index():
            is_nan = False
            vars = vars if vars != [] else self.variables()
            for var in vars:
                if var in self.variables():
                    is_nan = False
                    if is_numeric(self.values[var][i]):
                        is_nan = np.isnan(self.values[var][i])
                    if is_nan:
                        is_nan = True
                        break
            if is_nan:
                for var in self.values.keys():
                    try:
                        self.values[var].pop(i)
                    except:
                        pass

    def to_numpy(self, vars:list[str]=[])->None:
        self.dropna(vars)
        lst = []
        for i in self.index():
            in_lst = []
            for var in self.values.keys():
                if (var in vars) or (vars==[]):
                    if is_numeric(self.values[var][i]):
                        in_lst.append(self.values[var][i])
            lst.append(in_lst)
        return np.array(lst)

    def add_data(self,new_data:Data=None)->Data:
        if self.index() == None:
            indexes = [v for v in new_data.index()]
        else:
            indexes = [*self.index(),*[v for v in new_data.index() if not v in self.index()]]
        for var in new_data.variables():
            if var in self.variables():
                self.values[var].update(new_data.values[var])
            else:
                self.values[var] = {}
                for i in indexes:
                    if not i in new_data.index():
                        self.values[var][i] = np.nan
                    else:
                        self.values[var][i] = new_data.values[var][i]
    
    def transpose(self)->Data:
        values_t = {}
        for var, ival in self.values.items():
            for i, val in ival.items():
                if i in values_t.keys():
                    values_t[i][var] = val
                else:
                    values_t[i] = {var:val}
        return Data(self.type, values_t)

    @classmethod
    def read_csv(cls, path_file:str, data_type:str='cross', na:any='', do_faster_instead_of_low_space:bool=True)->Data:
        if do_faster_instead_of_low_space:
            with open(path_file, 'r') as f:
                lines = f.readlines()
            data, variables = {}, lines[0][:-1].split(',')      # end of line is \n
            for var in variables:
                data[var] = {}
            for index, line in enumerate(lines[1:]):
                row = line[:-1].split(',')      # end of line is \n
                for col in range(len(row)):
                    if row[col] == na:      # NA
                        data[variables[col]][index] = np.nan
                    else:
                        if row[col].isdigit():
                            data[variables[col]][index] = int(row[col])
                        elif is_numeric_str(row[col]):
                            data[variables[col]][index] = float(row[col])
                        else:
                            data[variables[col]][index] = row[col]
        else:
            import csv
            with open(path_file, 'r') as f:
                reader = csv.reader(f)
                start, data = True, {}
                while True:
                    try:
                        if start:
                            variables = next(reader)
                            for var in variables:
                                data[var] = {}
                            start = False
                            index = 0
                        else:
                            values = next(reader)
                            for col in range(len(values)):
                                data[variables[col]][index] = values[col]
                        index += 1
                    except:
                        break
        return cls(data_type, data)

    def to_csv(self, path_file:str, na:str=''):
        with open(path_file, 'a') as f:
            title = 'index'
            for var in self.variables():
                title += ',' + var
            f.write(title + '\n')
            for i in self.index():
                line = str(i)
                for var in self.variables():
                    is_nan = False
                    if is_numeric(self.values[var][i]):
                        if np.isnan(self.values[var][i]):
                            is_nan = True
                    if is_nan:
                        line += ',' + na
                    else:
                        line += ',' + str(self.values[var][i])
                f.write(line + '\n')

    def __len__(self):
        return len(self.index())

class Sample:
    def __init__(self, data: Data, index:list=[], name:str=None, weights:str='1') -> None:
        self.data = data
        if index == []:
            self.index = data.index()
        else:
            self.index = index
        if not set(self.index).issubset(set(data.index())):
            print('sample index is not subset of data index')
            raise
        self.name = name
        self.weights = weights

    def get_data(self) -> Data:
        res = {}
        for var in self.data.variables():
            res[var] = {}
            for i in self.index:
                if i in self.data.index():
                    res[var][i] = self.data.values[var][i]
                else:
                    raise ValueError(f"index {i} isn't in data index")
        return Data(self.data.type, res)

    split_methods = ['random', 'start', 'end']

    def split(self, ratio: float, names: list, method: str = 'random') -> list[Sample]:
        if method == 'random':
            ws = sum([w for i, w in self.data.values[self.weights].items() if i in self.index])
            weights = [w/ws for i, w in self.data.values[self.weights].items() if i in self.index]
            S1 = np.random.choice(self.index, int(ratio*len(self.index)), p=weights, replace=False)

            S2 = [i for i in self.index if not i in S1]
        elif method == 'start':
            n = int(ratio * len(self.index))
            S1, S2 = self.index[:n], self.index[n:]
        elif method == 'end':
            n = int(ratio * len(self.index))
            S1, S2 = self.index[:n], self.index[n:]
        return Sample(self.data, S1, names[0], self.weights), Sample(self.data, S2, names[1], self.weights)

    def get_weights(self, vars_conditions:list[list], totals:list[Union[int,float]])-> None:
        new_data=self.data.add_dummies(vars_conditions)
        vars_num = new_data.to_numpy()
        totals_num = np.array(totals)
        I = np.ones(len(self.data.index()))
        D = np.identity(len(self.data.index()))
        w_values_list = list(D @ I + D @ vars_num @ np.linalg.inv(vars_num.T @
                           D @ vars_num) @ np.transpose(totals_num.T - I.T @ D.T @ vars_num))
        w_values = dict(zip(self.data.index(), w_values_list))
        res = {}
        res['weights'] = w_values
        self.data.values.update(res)
        self.weights = 'weights'

    def __len__(self):
        return len(self.index)
    
    def __str__(self) -> str:
        return str(self.get_data())