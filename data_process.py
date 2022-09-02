from __future__ import annotations
from contextlib import suppress
import os, jdatetime
import time
from tkinter.ttk import Separator
from attr import validate
import numpy as np
import urllib, csv
from urllib.request import urlopen
import requests


from survey_stats.functions import *

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
        return self.to_str()

    def to_str(self, variables:list=[], full_variables:bool=False,
                     index:list=[], full_index:bool=False,
                     decimals:int=4, formated:bool=True)->str:
        #region vars
        if variables!=[]:
            vars = ['index'] + variables
        elif full_variables or len(self.variables())<7:
            vars = ['index'] + self.variables()
        else: 
            vars = ['index'] + self.variables()[:3] + ['...'] + self.variables()[-2:]
        #endregion
        #region inds
        if index!=[]:
            inds = index
        elif full_index or len(self.index())<15:
            inds = self.index()
        else:
            inds = self.index()[:10] + ['⁝'] + self.index()[-5:]
        #endregion
        #region widths[var]
        widths = {}
        for var in vars:
            for i in inds:
                if i!='⁝' and var!='...' and var!='index':
                    if var in self.variables() and i in self.index():
                        v = self.values[var][i]
                        if formated:
                            v = to_formated_str(v,decimals)
                        else:
                            v = str(v)
                        if not var in widths.keys():
                            widths[var] = max(len(v), len(str(var)))
                        else:
                            widths[var] = max(widths[var], len(v))
                    elif not var in self.variables():
                        raise ValueError(f"Error! '{var}' is not in variables.")
                    elif not i in self.index():
                        raise ValueError(f"Error! '{i}' is not in index.")
                elif var=='index':
                    if not var in widths.keys():
                        widths[var] = len(str(i))
                    else:
                        widths[var] = max(widths[var], len(str(i)))
                elif var=='...':
                    if not var in widths.keys():
                        widths[var] = 3
                    else:
                        widths[var] = max(widths[var], 3)
            widths[var] += 4
        #endregion
        rows = f'type: {self.type}\n'
        total_width = sum([widths[var] for var in vars])
        #region title
        rows += '\n ' + '-' * (total_width + len(vars)-1) + '\n|'
        for var in vars:
            rows += str(var).center(widths[var]) + '|'
        #endregion
        rows += '\n ' + '-' * (total_width + len(vars)-1)
        #region values
        for i in inds:
            for var in vars:
                if var == 'index':
                    rows += '\n|' + str(i).center(widths[var]) + '|'
                elif var == '...':
                    rows += '...'.center(widths['...']) + '|'
                elif i in self.index():
                    v = self.values[var][i]
                    if formated:
                        v = to_formated_str(v,decimals)
                    else:
                        v = str(v)
                    rows += v.center(widths[var]) + '|'
                elif i == '⁝':
                    rows += '⁝'.center(widths[var]) + '|'
        #endregion
        rows += '\n ' + '-' * (total_width + len(vars)-1)
        rows += f'\n{len(self.index())} × {len(self.variables())}'
        return rows

    def variables(self) -> None:
        return [var for var in self.values]

    def index(self, without_checking:bool=True) -> None:
        if without_checking:
            for v in self.values.keys():
                return list(self.values[v].keys())
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
    
    def set_index(self, var:str|list, drop_var:bool=True) -> None:
        new_dict = {}
        if type(var)==str:
            if var in self.values.keys():
                for v in self.values.keys():
                    if not (v == var and drop_var):
                        new_dict[v] = {self.values[var][i]:self.values[v][i] for i in self.values[var].keys()}
                self.values = new_dict
            else:
                raise ValueError(f"Error! {var} is not in variables of data.")
        elif type(var)==list:
            if len(var) == len(self.index()):
                for v in self.values.keys():
                    new_dict[v] = {var[i]:val for i, val in enumerate(self.values[v].values())}
                self.values = new_dict
            else:
                raise ValueError(f"Error! len of var (={len(var)}) is not equal to len of data index (={len(self.index())}).")

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
                    try:
                        if is_numeric(self.values[var][i]):
                            is_nan = np.isnan(self.values[var][i])
                        if is_nan:
                            is_nan = True
                            break
                    except:
                        pass
            if is_nan:
                for var in self.values.keys():
                    try:
                        self.values[var].pop(i)
                    except:
                        pass

    def value_to_nan(self, value:str|int|float, variables:list[str]=[])->None:
        if variables == []:
            variables = self.variables()
        for var in variables:
            for i in self.index():
                if self.values[var][i] == value:
                    self.values[var][i] = np.nan

    def to_numpy(self, vars:list[str]=[])->None:
        # self.dropna(vars)
        lst = []
        for i in self.index():
            in_lst = []
            for var in self.values.keys():
                if (var in vars) or (vars==[]):
                    if is_numeric(self.values[var][i]):
                        in_lst.append(self.values[var][i])
                    else:
                        raise ValueError(f"Error! value of {self.values[var][i]} is not numeric.")
            lst.append(in_lst)
        return np.array(lst)

    def add_data(self, new_data:Data=None)->Data:
        if self.index() == None:
            new_index = new_data.index()
            indexes = new_index
        else:
            old_index = self.index()
            indexes = set(old_index)
            if new_data.values != {}:
                new_index = set(new_data.index())-indexes
                indexes.update(new_index)
                new_index = list(new_index)
            else:
                new_index = []
            indexes = list(indexes)
            indexes.sort()
        old_vars = self.variables()
        new_vars = new_data.variables()
        vars = old_vars + [var for var in new_vars if not var in old_vars]
        for var in vars:
            if not var in old_vars:
                self.values[var] = dict(zip(indexes,[np.nan]*len(indexes)))
            if var in new_vars:
                new_values = {i:v for i,v in new_data.values[var].items() if not is_nan(v)}
                self.values[var].update(new_values)
            elif new_index != []:
                new_vals = dict(zip(new_index,[np.nan]*len(new_index)))
                self.values[var].update(new_vals)
    
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
    def read_csv(cls, path_file:str, data_type:str='cross', na:any='', index:str='index')->Data:
        if path_file[:4].lower()=='http':
            response = urlopen(path_file)
            lines = [l.decode('utf-8') for l in response.readlines()]
            cr = csv.reader(lines)
            values, vars, is_first = {}, [], True
            for i, row in enumerate(cr):
                if is_first:
                    vars = row
                    for var in vars:
                        values[var] = {}
                    is_first = False
                else:
                    for j, val in enumerate(row):
                        if val == na:
                            values[vars[j]][i] = np.nan
                        elif is_numeric_str(val):
                            values[vars[j]][i] = to_float(val)
                        else:
                            values[vars[j]][i] = val.strip()
        else:
            with open(path_file,'r', encoding='utf-8') as f:
                lines = f.readlines()
            n = len(lines)
            values, vars = {}, []
            for j, var in enumerate(lines[0].split(',')):
                var = var.replace('ï»؟','').replace('\n','')
                vars.append(var)
                values[var] = {}
            for i in range(1,n):
                for j, val in enumerate(lines[i].split(',')):
                    val = val.replace('ï»؟','').replace('\n','')
                    if val == na:
                        values[vars[j]][i] = np.nan
                    elif is_numeric_str(val):
                        values[vars[j]][i] = to_float(val)
                    else:
                        values[vars[j]][i] = val
        data = cls(data_type, values)
        if index in vars:
            data.set_index(index)
        elif 'index' in vars:
            data.set_index('index')
        return data

    def to_csv(self, path_file:str, na:str='', replace:bool = True, skip_index:bool=False):
        if os.path.exists(path_file):
            if replace:
                os.remove(path_file)
            else:
                old_name = path_file.split('\\')[-1]
                new_name = input(f"there is a file with same name '{old_name}', please, enter a new name without the path: ")+'.csv'
                path_file = path_file.replace(old_name,new_name)

        with open(path_file, 'w', encoding='utf-8') as f:
            if skip_index:
                title = ''
            else:
                title = 'index'
            start = True
            for var in self.variables():
                if not(start and skip_index):
                    title += ','
                start = False
                title += str(var)

            f.write(title + '\n')
            for i in self.index():
                if skip_index:
                    line = ''
                else:
                    line = str(i)
                start = True
                for var in self.variables():
                    if not(start and skip_index):
                        line += ','
                    start = False
                    is_nan = False
                    if is_numeric(self.values[var][i]):
                        is_nan = np.isnan(self.values[var][i])
                    if is_nan:
                        line += na
                    else:
                        line += str(self.values[var][i])
                f.write(line + '\n')

    def __len__(self):
        return len(self.index())

    def add_trend(self):
        j = 0
        self.values['trend'] = {}
        for i in self.index():
            self.values['trend'][i] = j
            j += 1

    fillna_methods = ['last', 'mean', 'growth']
    def fillna(self, variables:str|list[str]=[] , method:str|int|float='last', replace:bool=False):
        if type(variables) == str:
            variable = variables
            last_value, next_value= np.nan, np.nan
            n, index, j = len(self), self.index(), 0
            if not replace:
                self.values[variable + '_'] = {}
            while j < n:
                v = self.values[variable][index[j]]
                try:
                    is_nan = False
                    if is_numeric(v):
                        if np.isnan(v):
                            is_nan = True
                    if is_nan:
                        if method == 'last':
                            if not replace:
                                self.values[variable + '_'][index[j]] = last_value
                            else:
                                self.values[variable][index[j]] = last_value
                        elif method == 'mean':
                            k = 1
                            next_value = np.nan
                            while j+k < n:
                                v_ = self.values[variable][index[j+k]]
                                if not is_numeric(v):
                                    raise ValueError(f"Error! value of '{v_}' in variable '{variable}' is not numeric.")
                                try:
                                    if not np.isnan(v_):
                                        next_value = v_
                                        break
                                except:
                                    raise ValueError(f"Error! value of '{v}' in variable '{variable}'.")
                                k+=1
                            k_ = 0
                            if not is_numeric(last_value):
                                raise ValueError(f"Error! value of '{last_value}' in variable '{variable}' is not numeric.")
                            if not is_numeric(next_value):
                                raise ValueError(f"Error! value of '{next_value}' in variable '{variable}' is not numeric.")
                            while k_ < k:
                                if not (np.isnan(last_value) or np.isnan(next_value)):
                                    try:
                                        if not replace:
                                            self.values[variable + '_'][index[j+k_]] = ((k_+1)*last_value+(k-k_)*next_value)/(k+1)
                                        else:
                                            self.values[variable][index[j+k_]] = ((k_+1)*last_value+(k-k_)*next_value)/(k+1)
                                    except:
                                        raise ValueError(f"Error! value of '{v}' in variable '{variable}'.")
                                else:
                                    if not replace:
                                        self.values[variable + '_'][index[j+k_]] = np.nan
                                    else:
                                        self.values[variable][index[j+k_]] = np.nan
                                k_+=1 
                            if not replace:
                                self.values[variable + '_'][index[j+k]] = next_value
                            else:
                                self.values[variable][index[j+k]] = next_value
                            last_value = next_value
                            j += k
                        elif method == 'growth':
                            k = 1
                            next_value = np.nan
                            while j+k < n:
                                v_ = self.values[variable][index[j+k]]
                                if not is_numeric(v):
                                    raise ValueError(f"Error! value of '{v_}' in variable '{variable}' is not numeric.")
                                try:
                                    if not np.isnan(v_):
                                        next_value = v_
                                        break
                                except:
                                    raise ValueError(f"Error! value of '{v}' in variable '{variable}'.")
                                k+=1
                            k_ = 0
                            if not is_numeric(last_value):
                                raise ValueError(f"Error! value of '{last_value}' in variable '{variable}' is not numeric.")
                            if not is_numeric(next_value):
                                raise ValueError(f"Error! value of '{next_value}' in variable '{variable}' is not numeric.")
                            while k_ < k:
                                if not (np.isnan(last_value) or np.isnan(next_value)):
                                    try:
                                        if not replace:
                                            self.values[variable + '_'][index[j+k_]] = last_value * (next_value/last_value)**((k_+1)/(k+1))
                                        else:
                                            self.values[variable][index[j+k_]] = last_value * (next_value/last_value)**((k_+1)/(k+1))
                                    except:
                                        raise ValueError(f"Error! value of '{v}' in variable '{variable}'.")
                                else:
                                    if not replace:
                                        self.values[variable + '_'][index[j+k_]] = np.nan
                                    else:
                                        self.values[variable][index[j+k_]] = np.nan
                                k_+=1 
                            if not replace:
                                self.values[variable + '_'][index[j+k]] = next_value
                            else:
                                self.values[variable][index[j+k]] = next_value
                            last_value = next_value
                            j += k
                        else:
                            if not replace:
                                self.values[variable + '_'][index[j]] = method
                            else:
                                self.values[variable][index[j]] = method
                    else:
                        if not replace:
                            self.values[variable + '_'][index[j]] = v
                        else:
                            self.values[variable][index[j]] = v
                        last_value = v
                    j += 1
                except:
                    raise ValueError(f"Error! value of '{v}' in variable '{variable}', index = {index[j]}.")
        elif type(variables)==list:
            if variables == []:
                variables = self.variables()
            for var in variables:
                self.fillna(var, method, replace)

    def add_trend(self):
        j = 0
        self.values['trend'] = {}
        for i in self.index():
            self.values['trend'][i] = j
            j += 1

    def sort(self, key:str='', ascending:bool=True):
        if key=='' or key=='index':
            index, vars = self.index(), self.variables()
            index.sort(reverse=not ascending)
            data = Data(self.type, {vars[0]:dict(zip(index, ['']*len(index)))})
            data.add_data(self)
            self.values = data.values
        else:
            if not key in self.variables():
                raise ValueError(f"Error! key ('{key}') is not in variables.")
            vars = self.variables()
            index = [(i, self.values[key][i]) for i in self.index() 
                            if not np.isnan(self.values[key][i])]
            index.extend([(i, self.values[key][i]) for i in self.index() 
                            if np.isnan(self.values[key][i])])
            index.sort(reverse=not ascending,key=lambda row: row[1])
            index = [i  for i,_ in index]
            data = Data(self.type, {vars[0]:dict(zip(index, ['']*len(index)))})
            data.add_data(self)
            self.values = data.values

    @classmethod
    def read_text(cls, path_file:str, data_type:str='cross', na:any='', index:str='index', seprator:str=',')->Data:
        with open(path_file,'r', encoding='utf-8') as f:
            lines = f.readlines()
        n = len(lines)
        values, vars = {}, []
        for j, var in enumerate(lines[0].split(seprator)):
            var = var.replace('ï»؟','').replace('\n','')
            vars.append(var)
            values[var] = {}
        for i in range(1,n):
            for j, val in enumerate(lines[i].split(seprator)):
                val = val.replace('ï»؟','').replace('\n','')
                if val == na:
                    values[vars[j]][i] = np.nan
                elif is_numeric_str(val):
                    values[vars[j]][i] = to_float(val)
                else:
                    values[vars[j]][i] = val.strip()
        data = cls(data_type, values)
        if index in vars:
            data.set_index(index)
        elif 'index' in vars:
            data.set_index('index')
        return data

    def to_text(self, path_file:str, na:str='', replace:bool = True, skip_index:bool=False, seprator:str=','):
        if os.path.exists(path_file):
            if replace:
                os.remove(path_file)
            else:
                old_name = path_file.split('\\')[-1]
                new_name = input(f"there is a file with same name '{old_name}', please, enter a new name without the path: ")+'.csv'
                path_file = path_file.replace(old_name,new_name)

        with open(path_file, 'a', encoding='utf-8') as f:
            if skip_index:
                title = ''
            else:
                title = 'index'
            start = True
            for var in self.variables():
                if not(start and skip_index):
                    title += seprator
                start = False
                title += str(var)
            f.write(title + '\n')
            for i in self.index():
                if skip_index:
                    line = ''
                else:
                    line = str(i)
                start = True
                for var in self.variables():
                    if not(start and skip_index):
                        line += seprator
                    start = False
                    is_nan = False
                    if is_numeric(self.values[var][i]):
                        is_nan = np.isnan(self.values[var][i])
                    if is_nan:
                        line += na
                    else:
                        line += str(self.values[var][i])
                f.write(line + '\n')

    def add_index(self, index:list):
        for var in self.variables():
            for i in index:
                if not i in self.values[var].keys():
                    self.values[var][i] = np.nan

    def add_a_variable(self, name:str, values:list):
        if len(self.index())==len(values):
            self.values[name] = dict(zip(self.index(), values))
        else:
            raise ValueError(f"Error! lenght of values ({len(values)}) must be equal to lenght of index ({len(self.index())}).")

    def to_timeseries(self):
        data = TimeSeries('time', self.values)
        data.complete_dates()
        return data

    def line_plot(self, vars:list[str]):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        plt.xticks(rotation='vertical')
        plt.margins(0.1)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        legs = []
        for var in vars:
            ax.plot(self.index(), list(self.values[var].values()))
            legs.append(var)
        ax.legend(legs)
        interval = int(len(self)/20) if len(self)>40 else 2 if len(self)>40 else 1
        ax.set_xticks(ax.get_xticks()[::interval])
        plt.show()

    @classmethod
    def read_xls(cls, path_file: str, data_type:str='cross', na:any='', index:str='index') -> Data:
        if 'http://' in path_file or 'https://' in path_file:
            try:
                content = requests.get(path_file).text
            except:
                import urllib3
                urllib3.disable_warnings()
                content = requests.get(path_file, allow_redirects=True, verify=False).text
        else:
            with open(path_file, encoding='utf8') as f:
                content = f.read()
        rows = xls_read(content)
        titles = rows[0]
        values, i = {}, 0
        for row in rows[1:]:
            cols = row
            i += 1
            val = {}
            for j, col in enumerate(cols):
                vali = col
                if is_numeric_str(vali):
                    vali = to_float(vali)
                elif vali == na:
                    vali = np.nan
                val[titles[j]] = vali
            values[i] = val
        res = cls(data_type, values).transpose()
        if index in titles:
            res.set_index(index)
        elif 'index' in titles:
            res.set_index('index')
        return res

    def to_xls(self,  path_file:str, na:str='', replace:bool = True, skip_index:bool=False):
        #heading
        data = '''
        <html>
            <head>
                <meta charset="utf-8" />
            </head>



            <table>
        '''
        #titles
        data += '''
            <thead>
                <tr>
                    <th>index</th>'''
        for var in self.variables():
            data += f'''
                    <th>{var}</th>'''
        data += '''
                </tr>
            </thead>
        '''
        #values
        data += '''
            <tbody>
        '''
        for row in self.index():
            data += '''
                <tr>
            '''
            data += f'''
                    <th>{row}</th>'''
            for var in self.variables():
                vali, isnan = self.values[var][row], False
                if is_numeric(vali):
                    isnan = np.isnan(vali)
                if isnan:
                    data += f'''
                        <th>{na}</th>'''
                else:
                    data += f'''
                        <th>{vali}</th>'''
            data += '''
                </tr>
            '''
        data += '''
            </tbody>
        '''
        #ending
        data += '''
            </table>
        </html>
        '''
        if os.path.exists(path_file):
            if replace:
                os.remove(path_file)
            else:
                raise ValueError(f"Error! '{path_file.split('/')[-1]}' exists.")
        with open(path_file, 'w', encoding='utf8') as f:
            f.write(data)

    @classmethod
    def read_excel(cls, path_file:str, sheet:str='', data_type:str='cross', na:any='', index:str='index',
                    first_row:int=0, first_col:int=0) -> Data:
        import xlrd
        wb = xlrd.open_workbook(path_file)
        if sheet != '':
            ws = wb.sheet_by_name(sheet)
        else:
            ws = wb.sheet_by_index(0)
        i = first_row
        values = {}

        j = 1
        var_names = {}
        for col in range(first_col, ws.ncols):
            var_name = ws.cell(first_row, col).value
            if var_name == '':
                var_name = f'var{j}'
                j += 1
            var_names[col] = var_name
            values[var_name] = {}
        for row in range(first_row+1,ws.nrows):
            for col in range(first_col, ws.ncols):
                val = ws.cell(row, col).value
                if val == '':
                    val = np.nan
                values[var_names[col]][row-first_row] = val
        data = Data(data_type, values)
        if index in data.variables():
            data.set_index(index)
        return data

class Sample:
    def __init__(self, data: Data, index:list=[], name:str=None, weights:str='1') -> None:
        self.data = data
        if index == []:
            self.index = data.index()
        else:
            self.index = index
        if not set(self.index).issubset(set(data.index())):
            raise ValueError('sample index is not subset of data index')
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
            if self.weights == '1':
                S1 = np.random.choice(self.index, int(ratio*len(self.index)), replace=False)
            else:
                ws = sum([w for i, w in self.data.values[self.weights].items() if i in self.index])
                weights = [w/ws for i, w in self.data.values[self.weights].items() if i in self.index]
                S1 = np.random.choice(self.index, int(ratio*len(self.index)), p=weights, replace=False)

            S2 = list(set(self.index)-set(S1))
        elif method == 'start':
            n = int(ratio * len(self.index))
            S1, S2 = self.index[:n], self.index[n:]
        elif method == 'end':
            n = int((1-ratio) * len(self.index))
            S1, S2 = self.index[:n], self.index[n:]
        return Sample(self.data, S1, names[0], self.weights), Sample(self.data, S2, names[1], self.weights)

    def get_weights(self, path_file_csv:str)-> None:
        # vars_conditions:list[list], totals:list[Union[int,float]]
        groups = Data.read_csv(path_file_csv)
        groups_n = len(groups.index())
        set_index, set_totals = False, False
        for var in groups.variables():
            strs, nums = 0, 0
            for i in groups.index():
                if is_numeric(groups.values[var][i]):
                    nums += 1
                else:
                    strs += 1
            if strs == groups_n:
                groups.set_index(var)
                set_index = True
            elif nums == groups_n:
                totals = list(groups.values[var].values())
                set_totals = True
            else:
                raise ValueError(f"Error! {var} includes numbers and strings, data must be include a string variable as group name and a numeric variable as population of group.")
        if set_index and set_totals:
            vars_conditions = []
            for g in groups.index():
                r = []
                for v in g.split('*'):
                    if len(v.split('>='))>1:
                        i = v.split('>=')
                        sep = '>='
                    elif len(v.split('<='))>1:
                        i = v.split('<=')
                        sep = '<='
                    elif len(v.split('<'))>1:
                        i = v.split('<')
                        sep = '<'
                    elif len(v.split('>'))>1:
                        i = v.split('>')
                        sep = '>'
                    elif len(v.split('='))>1:
                        i = v.split('=')
                        sep = '='
                    if is_numeric_str(i[1]):
                        i[1] = float(i[1])
                    r.append((i[0],sep,i[1]))
                vars_conditions.append(r)

            
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
        else:
            raise ValueError(f"Error! data must be include a string variable as group name and a numeric variable as population of group.")

    def __len__(self):
        return len(self.index)
    
    def __str__(self) -> str:
        return str(self.get_data())

class TimeSeries(Data):
    types = ['daily', 'weekly', 'monthly', 'seasonal', 'annual']
    @staticmethod
    def str_to_date(date:str)->jdatetime.date:
        seprator = '-' if len(date.split('-')) != 1 else '/' if len(date.split('/')) != 1 else ''

        if seprator == '':
            raise ValueError(f"Error! seprator are not standard: '-' or '/'.")
        if len(date.split(seprator))!=3:
            raise ValueError(f"Error! date must have three parts: year, month and day.")
        return jdatetime.date(*[int(p) for p in date.split(seprator)])

    @staticmethod
    def type_of_dates(dates:list[str]):
        if dates == []:
            return 'daily'
        elif type(dates[0]) == int:
            return 'annual'
        else:
            seprator = '-' if len(dates[0].split('-')) != 1 else '/' if len(dates[0].split('/')) != 1 else ''
            if seprator == '':
                raise ValueError(f"Error! seprator are not standard: '-' or '/'.")
            splits = dates[0].split(seprator)
            if len(splits) == 2:
                try:
                    parts2 = [int(date.split(seprator)[1]) for date in dates]
                    if max(parts2) == 4:
                        return 'seasonal'
                    # elif max(parts2) == 12:
                    else:
                        return 'monthly'
                    # else:
                    #     raise ValueError(f"Error! max second part of dates, '{max(parts2)}' is not 4 (seasonal) or 12 (monthly).")
                except:
                    raise ValueError(f"Error! dates have two parts (they are monthly or seasonal), but probably one date is different from other dates., it has one or more than two parts.")
            elif len(splits) == 3:
                try:
                    sorted_dates = []
                    for date in sorted(dates):
                        seprator = '-' if len(date.split('-')) != 1 else '/' if len(date.split('/')) != 1 else ''
                        if seprator == '':
                            raise ValueError(f"Error! date {date} seprator are not standard: '-' or '/'.")
                        sorted_dates.append(jdatetime.date(*[int(p) for p in date.split(seprator)]))
                    difs = set([(sorted_dates[i]-sorted_dates[i-1]).days for i in range(1,len(sorted_dates))])
                    weekly = True
                    for dif in difs:
                        if dif % 7 != 0:
                            weekly = False
                            break
                    if weekly:
                        return 'weekly'
                    else:
                        return 'daily'
                except:
                    raise ValueError(f"Error! dates have three parts (they are daily or weekly), but probably one date is different from other dates, it has one or two than two parts.")
            else:
                raise ValueError(f"Error! the number of parts '{dates[0]}' must be 2 (seasonal or monthly) or 3 (daily or weekly).")

    def __init__(self, type:str='time', values: dict[dict] = {}) -> None:
        self.type = type
        self.dates = list(values[list(values.keys())[0]].keys())
        self.date_type = TimeSeries.type_of_dates(self.dates)
        self.values = values
        super().__init__('time', self.values)

    def complete_dates(self):
        if self.dates != []:
            dates = []
            for date in self.dates:
                seprator = '/' if '/' in date else '-' if '-' in date else ''
                if seprator == '':
                    raise ValueError(f"Error! seprator must be '-' or '/'.")
                sp = [f'0{int(x)}' if int(x) < 10 else f'{int(x)}' 
                        for x in date.split(seprator)]
                date = '-'.join(sp)
                dates.append(date)
            dates.sort()
            start_dates, end_dates = dates[0], dates[-1]
            new_values, dates = {}, []
            if self.date_type in ['daily', 'weekly']:
                sp = '-' if len(start_dates.split('-')) == 3 else '/' if len(start_dates.split('/')) == 3 else ''
                if sp == '':
                    raise ValueError(f'Error! seprator only must be - or /.')
                delta = 1 if self.date_type == 'daily' else 7

                start_dates, end_dates = [TimeSeries.str_to_date(date) for date in [start_dates, end_dates]]
                date = start_dates
                dates = []
                while (end_dates-date).days>=0:
                    new_date = date.strftime('%Y-%m-%d')
                    m = f'{date.month}' if date.month > 9 else f'0{date.month}'
                    d = f'{date.day}' if date.day > 9 else f'0{date.day}'
                    old_dates = [
                                f'{date.year}-{date.month}-{date.day}',
                                 f'{date.year}-{m}-{date.day}', 
                                 f'{date.year}-{date.month}-{d}', 
                                 f'{date.year}-{m}-{d}', 
                                f'{date.year}/{date.month}/{date.day}',
                                 f'{date.year}/{m}/{date.day}', 
                                 f'{date.year}/{date.month}/{d}', 
                                 f'{date.year}/{m}/{d}'
                                 ]
                    dates.append(new_date)
                    
                    for var in self.values.keys():
                        if not var in new_values.keys():
                            new_values[var] = {}
                        new_values[var][new_date] = np.nan
                        for old_date in old_dates:
                            if old_date in self.values[var].keys():
                                new_values[var][new_date] = self.values[var][old_date]
                    date = date + jdatetime.timedelta(delta)
            elif self.date_type in ['monthly', 'seasonal']:
                sp = '-' if len(start_dates.split('-')
                                ) == 2 else '/' if len(start_dates.split('/')) == 2 else ''
                if sp == '':
                    raise ValueError(f'Error! seprator only must be - or /.')
                n = 13 if self.date_type=='monthly' else 5
                ystart, mstart = [int(p) for p in start_dates.split(sp)]
                yend, mend = [int(p) for p in end_dates.split(sp)]
                for y in range(ystart, yend+1):
                    st = mstart if y == ystart else 1
                    en = mend+1 if y==yend else n
                    for m in range(st, en):
                        mstr = f'{m}' if m>9 else f'0{m}'
                        new_date = f'{y}-{mstr}'
                        old_dates = [f'{y}-{mstr}', f'{y}-{m}']
                        dates.append(new_date)
                        for var in self.values.keys():
                            if not var in new_values.keys():
                                new_values[var] = {}
                            new_values[var][new_date] = np.nan
                            for date in old_dates:
                                if date in self.values[var].keys():
                                    new_values[var][new_date] = self.values[var][date]
            elif self.date_type != 'annual':
                if not (is_numeric(start_dates) or is_numeric(end_dates)):
                    raise ValueError(f"Error! annual dates must be numeric.")
                for y in range(start_dates, end_dates+1):
                    dates.append(y)
                    for var in self.values.keys():
                        if not var in new_values.keys():
                            new_values[var] = {}
                        if y in self.values[var].keys():
                            new_values[var][y] = self.values[var][y]
                        else:
                            new_values[var][y] = np.nan
            else:
                raise ValueError(
                    f"Error! date_type '{self.date_type}' most be daily, weekly, monthly, seasonal, or annual")
                
            self.dates = dates
            self.values = new_values

    def reset_date_type(self):
        self.dates = self.index()
        self.date_type = TimeSeries.type_of_dates(self.dates)

    def set_index(self, var: str | list, drop_var: bool = True) -> None:
        new_dict = {}
        if type(var)==str:
            if var in self.values.keys():
                date_types = TimeSeries.type_of_dates(list(self.values[var].values()))
                for v in self.values.keys():
                    values = {}
                    if not (v == var and drop_var):
                        for i in self.values[var].keys():
                            date, value = self.values[var][i], self.values[v][i]
                            date = date_to_standard(date, date_types)
                            values.update({date:value})
                        new_dict.update({v:values})
                self.values = new_dict
            else:
                raise ValueError(f"Error! {var} is not in variables of data.")
        elif type(var)==list:
            if len(var) == len(self.index()):
                for v in self.values.keys():
                    values = {}
                    for i, value in enumerate(self.values[v].values()):
                        date = date_to_standard(var[i], date_types)
                        values.update({date:value})
                    new_dict.update({v:values})
                self.values = new_dict
            else:
                raise ValueError(f"Error! len of var (={len(var)}) is not equal to len of data index (={len(self.index())}).")
        self.reset_date_type()
        self.complete_dates()

    def to_monthly(self, method:str, farvardin_adj:bool=False)->TimeSeries:
        if self.date_type == 'daily':
            #region sums and counts for months
            sums, counts = {}, {}
            for var in self.variables():
                sums[var], counts[var] = {}, {}
                for day in self.dates:
                    v = self.values[var][day]

                    sep = '-' if '-' in day else '/' if '/' in day else ''
                    if sep == '':
                        raise ValueError(
                            f"Error! {day} in variable {var} isn't standard. seprator must be '-' or '/'.")
                    if len(day.split(sep)) != 3:
                        raise ValueError(f"Error! {day} in variable {var} isn't standard. day must has 3 part.")
                    y, m, d = [int(x) for x in day.split(sep)]
                    month = f'{y}-{m}' if m>9 else f'{y}-0{m}'

                    if not is_nan(v, is_number=True):
                        if month in sums[var].keys():
                            sums[var][month] += v
                            counts[var][month] += 1
                        else:
                            sums[var][month] = v
                            counts[var][month] = 1
            #endregion
            #region average
            if method == 'average':
                start = True
                for var in self.variables():
                    for month in sums[var].keys():
                        if start:
                            month_start, month_end, start = month, month, False
                        else:
                            month_start, month_end = min(month_start, month), max(month_end, month)
                        sums[var][month] /= counts[var][month]
            #endregion
            #region complete keys
            def next_month(month:str)->str:
                sep = '-' if '-' in month else '/' if '/' in month else ''
                if sep == '':
                    raise ValueError(
                        f"Error! {month} isn't standard. seprator must be '-' or '/'.")
                if len(month.split(sep)) != 2:
                        raise ValueError(f"Error! {month} isn't standard. month must has 2 part.")
                y, m = [int(x) for x in month.split(sep)]
                return f'{y+1}-01' if m==12 else f'{y}-{m+1}' if m+1>9 else f'{y}-0{m+1}'
            
            month = month_start
            while month <= month_end:
                for var in self.variables():
                    if not month in sums[var].keys():
                        sums[var][month] = np.nan
                month = next_month(month)
            #endregion
            
            res = TimeSeries('time', sums)
            res.sort()

            #region edit last month
            for var in self.variables():
                for month in res.dates[::-1]:
                    v = res.values[var][month]
                    if not is_nan(v, is_number=True):
                        month_end = month
                        break
                sep = '-' if '-' in month else '/'
                y, m = [int(x) for x in month_end.split(sep)]
                if m == 12:
                    no_days = (jdatetime.date(y+1, 1, 1) - jdatetime.date(y, m, 1)).days
                else:
                    no_days = 31 if m<7 else 30
                s0 = 0
                lag = 1
                while s0 == 0:
                    s, n, s0, n0 = 0, 0, 0, 0
                    for d in range(1, no_days+1):
                        m_str = f'{m}' if m>9 else f'0{m}'
                        d_str = f'{d}' if d>9 else f'0{d}'
                        day = f'{y}-{m_str}-{d_str}'
                        if m>lag:
                            m_str0 = f'{m-lag}' if m-lag>9 else f'0{m-lag}'
                            day0 = f'{y}-{m_str0}-{d_str}'
                            month_last = f'{y}-{m_str0}'
                        else:
                            o = y*12+m-lag
                            y_lag, m_lag = (o-1)//12, o - ((o-1)//12)*12
                            m_str0 = f'{m_lag}' if m_lag > 9 else f'0{m_lag}'
                            day0 = f'{y_lag}-{m_str0}-{d_str}'
                            month_last = f'{y_lag}-{m_str}'
                        d0 = d
                        while not day0 in self.dates and d0>0:
                            d0 -= 1
                            d_str = f'{d0}' if d0 > 9 else f'0{d0}'
                            day0 = day0[:8] + d_str
                        if day in self.dates:
                            v = self.values[var][day]
                            v0 = self.values[var][day0]
                            if not is_nan(v, is_number=True):
                                s += v
                                n += 1
                            if not is_nan(v0, is_number=True):
                                s0 += v0
                                n0 += 1
                        else:
                            break
                    lag += 1
                if method=='average':
                    res.values[var][month_end] = (s*n0/(n*s0)) * res.values[var][month_last]
                else:
                    res.values[var][month_end] = (s/s0) * res.values[var][month_last]

            #endregion

            return res
        elif self.date_type == 'weekly':
            values = {}
            for var in self.variables():
                values[var] = {}
                last_month = ''
                for week in self.dates:
                    seprator = '-' if len(week.split('-')) != 1 else '/' if len(week.split('/')) != 1 else ''
                    if seprator == '':
                        raise ValueError(f"Error! seprator are not standard: '-' or '/'.")
                    if len(week.split(seprator))==3:
                        y,m,d = [int(x) for x in week.split(seprator)]
                        month = f'{y}-{m}' if m > 9 else f'{y}-0{m}'
                        if month != last_month:
                            summation, last_value, count = 0, np.nan, 0
                        last_month = month
                        value = self.values[var][week]
                        is_nan_value = False
                        if is_numeric(value):
                            if np.isnan(value):
                                is_nan_value = True
                        if not is_nan_value:
                            summation += value
                            count += 1
                            last_value = value
                        else:
                            is_nan_last_value = False
                            if is_numeric(last_value):
                                if np.isnan(last_value):
                                    is_nan_last_value = True
                            if not is_nan_last_value:
                                summation += last_value
                                count += 1
                        if count>0:
                            values[var][month] = summation / count if method=='average' else summation
                        elif not month in values[var].keys():
                            values[var][month] = np.nan
                    else:
                        raise ValueError(f"Error! '{week}' is not a standard format of daily date like yyyy-mm-dd.")
            res = TimeSeries('time', values)
            res.reset_date_type()
            res.complete_dates()
            return res
        elif self.date_type == 'monthly':
            return self
        elif self.date_type == 'seasonal':
            if method == 'curve':
                #region Type Error
                if self.date_type != 'seasonal':
                    raise ValueError(f"Error! 'cureve' method only work on 'seanonal' data types.")
                #endregion
                values = {}
                for var in self.variables():
                    #region start and end values
                    start, nans, dates, month_dates = True, 0, [], []
                    for date in self.dates:
                        if not np.isnan(self.values[var][date]):
                            dates.append(date)
                            start = False
                        elif not np.isnan(self.values[var][date]) and not start:
                            if nans > 0:
                                raise ValueError(f"Error! there is a 'nan' bettween values of '{var}'.")
                            n += 1
                            dates.append(date)
                        elif np.isnan(self.values[var][date]) and not start:
                            nans += 1
                    #endregion

                    rows = len(dates) * 3
                    coefs_arr, const_vec = np.zeros((rows, rows)), np.zeros((rows, 1))

                    #region values
                    for i in range(rows):
                        year,season = [int(x) for x in dates[i//3].split('-')]
                        month = (season-1)*3 + i%3 + 1
                        date = f'{year}-0{month}' if month < 10 else f'{year}-{month}'
                        month_dates.append(date)
                        if i == 0:
                            coefs_arr[i][0] = 1             # j=
                            coefs_arr[i][1] = 1             # j=1
                            coefs_arr[i][2] = 1             # j=2
                            coefs_arr[i][3] = 0             # j=3
                            for j in range(4, rows):        # j>3
                                coefs_arr[i][j] = 0
                            const_vec[i][0] = self.values[var][dates[0]]
                        elif i == 1:
                            coefs_arr[i][0] = 2             # j=0
                            coefs_arr[i][1] = -3            # j=1
                            coefs_arr[i][2] = 0             # j=2
                            coefs_arr[i][3] = 1             # j=3
                            for j in range(4, rows):        # j>3
                                coefs_arr[i][j] = 0
                            const_vec[i][0] = 0
                        elif i == 2:
                            coefs_arr[i][0] = 1             # j=0
                            coefs_arr[i][1] = 0             # j=1
                            coefs_arr[i][2] = -3            # j=2
                            coefs_arr[i][3] = 2             # j=3
                            for j in range(4, rows):        # j>3
                                coefs_arr[i][j] = 0
                            const_vec[i][0] = 0
                        elif 2<i<rows-3:
                            for j in range(i-1):            # j<i-1
                                coefs_arr[i][j] = 0
                            if i % 3 == 0:
                                coefs_arr[i][i-1] = 1        # j=i-1
                                coefs_arr[i][i] = -2         # j=i
                                coefs_arr[i][i+1] = 1        # j=i+1
                                coefs_arr[i][i+2] = 0        # j=i+2
                                coefs_arr[i][i+3] = 0        # j=i+3
                                const_vec[i][0] = 0
                            elif i % 3 == 1:
                                coefs_arr[i][i-2] = 0        # j=i-2
                                coefs_arr[i][i-1] = 0        # j=i-1
                                coefs_arr[i][i] = 1          # j=i
                                coefs_arr[i][i+1] = -2       # j=i+1
                                coefs_arr[i][i+2] = 1        # j=i+2
                                const_vec[i][0] = 0
                            elif i % 3 == 2:
                                coefs_arr[i][i-3] = 0        # j=i-3
                                coefs_arr[i][i-2] = 1        # j=i-2
                                coefs_arr[i][i-1] = 1        # j=i-1
                                coefs_arr[i][i] = 1          # j=i
                                coefs_arr[i][i+1] = 0        # j=i+1
                                const_vec[i][0] = self.values[var][dates[i//3]]
                            for j in range(i+4,rows):       # j>i+3
                                coefs_arr[i][j] = 0
                        elif i == rows-3:
                            coefs_arr[i][rows-4] = 2
                            coefs_arr[i][rows-3] = -3
                            coefs_arr[i][rows-2] = 0
                            coefs_arr[i][rows-1] = 1
                            for j in range(rows-4):
                                coefs_arr[i][j] = 0
                            const_vec[i][0] = 0
                        elif i == rows-2:
                            coefs_arr[i][rows-4] = 1
                            coefs_arr[i][rows-3] = 0
                            coefs_arr[i][rows-2] = -3
                            coefs_arr[i][rows-1] = 2
                            for j in range(rows-4):
                                coefs_arr[i][j] = 0
                            const_vec[i][0] = 0
                        elif i == rows-1:
                            coefs_arr[i][rows-4] = 0
                            coefs_arr[i][rows-3] = 1
                            coefs_arr[i][rows-2] = 1
                            coefs_arr[i][rows-1] = 1
                            for j in range(rows-4):
                                coefs_arr[i][j] = 0
                            const_vec[i][0] = self.values[var][dates[-1]]
                    #endregion
                    values_arr = np.matmul(np.linalg.inv(coefs_arr), const_vec)
                    values_lst = list([float(x[0]) for x in values_arr])
                    values_dict = dict(zip(month_dates, values_lst))
                    #region adjusted farvardins
                    if farvardin_adj:
                        for date in dates:
                            year, season = [int(x) for x in date.split('-')]
                            if season == 1:
                                # first adjusted
                                m01 = values_dict[f'{year}-01']*0.6
                                m02 = values_dict[f'{year}-01']*0.4+values_dict[f'{year}-02']*.8
                                m03 = values_dict[f'{year}-03']*1.2
                                total = values_dict[f'{year}-01'] + \
                                    values_dict[f'{year}-02'] + \
                                    values_dict[f'{year}-03']
                                # final adjusted
                                values_dict[f'{year}-01'] = m01 * total / (m01+m02+m03)
                                values_dict[f'{year}-02'] = m02 * total / (m01+m02+m03)
                                values_dict[f'{year}-03'] = m03 * total / (m01+m02+m03)
                    #endregion
                    values[var] = values_dict
            else:
                values = {}
                start_year, start_season = [int(x) for x in self.dates[0].split('-')]
                end_year, end_season = [int(x) for x in self.dates[-1].split('-')]
                start_month, end_month = (start_season-1)*3+1, (end_season-1)*3+3
                for year in range(start_year, end_year+1):
                    st = start_month if year == start_year else 1
                    en = end_month+1 if year == end_year else 13
                    for month in range(st,en):
                        date = f'{year}-{month}' if month >9 else f'{year}-0{month}'
                        for var in self.variables():
                            if not var in values.keys(): 
                                values[var] = {}
                            if method=='sum':
                                values[var][date] = self.values[var][f'{year}-{(month-1)//3+1}']/3
                            elif method=='average':
                                values[var][date] = self.values[var][f'{year}-{(month-1)//3+1}']
            res = TimeSeries('time', values)
            res.reset_date_type()
            res.complete_dates()
            return res
        elif self.date_type == 'annual':
            values = {}
            for year in self.dates:
                for month in range(1,13):
                    date = f'{year}-{month}' if month >9 else f'{year}-0{month}'
                    for var in self.variables():
                        if not var in values.keys(): 
                            values[var] = {}
                        if method=='sum':
                            values[var][date] = self.values[var][year]/12
                        elif method=='average':
                            values[var][date] = self.values[var][year]
            res = TimeSeries('time', values)
            res.reset_date_type()
            res.complete_dates()
            return res

    def to_daily(self)->TimeSeries:
        if self.date_type == 'daily':
            return self
        elif self.date_type == 'weekly':
            pass    #TODO
        elif self.date_type == 'monthly':
            y,m = [int(x) for x in self.index()[0].split('-')]
            start = jdatetime.date(y,m, 1)
            y,m = [int(x) for x in self.index()[-1].split('-')]
            days_of_end_month = (jdatetime.date(y,m+1,1)-
                                    jdatetime.date(y,m,1)).days
            end = jdatetime.date(y,m, days_of_end_month)
            new_index, date = [], start
            while (end-date).days >= 0:
                date = date + jdatetime.timedelta(1)
                new_index.append(date)
            values = {}
            for var in self.variables():
                values[var] = {}
            for var in self.variables():
                for i in new_index:
                    index_new = i.strftime('%Y-%m-%d')
                    index_old = f'{i.year}-{i.month}' if i.month>9 else f'{i.year}-0{i.month}'
                    is_in_data = False
                    for j in self.index():
                        if i.year == int(j.split('-')[0]) and \
                            i.month == int(j.split('-')[1]):
                            is_in_data = True
                            break
                    if is_in_data:
                        values[var][index_new] = self.values[var][index_old]
            res = TimeSeries('time', values)
            res.reset_date_type()
            res.complete_dates()
            return res
        elif self.date_type == 'seasonal':
            pass    #TODO
        elif self.date_type == 'annual':
            pass    #TODO

    def to_weekly(self, method:str='average')->TimeSeries:
        if self.date_type == 'daily':
            values = {}
            for var in self.variables():
                values[var] = {}
                w, start = 0, True
                for date in self.dates:
                    sep = '-'
                    y,m,d = [int(x) for x in date.split(sep)]
                    jdate = jdatetime.date(y, m, d)
                    weekday = jdate.weekday()
                    v = self.values[var][date]
                    if weekday == 0:
                        w += 1
                        if not is_nan(v, is_number=True):
                            s, n = v, 1
                        else:
                            s, n = 0, 0
                        start = False
                    elif not start:
                        if not is_nan(v, is_number=True):
                            s += v
                            n += 1
                        if weekday == 6:
                            if n == 0:
                                values[var][date] = np.nan
                            else:
                                values[var][date] = s/n if method=='average' else s
                endweek = (jdate + jdatetime.timedelta(6-weekday)).strftime('%Y-%m-%d')
                if n == 0:
                    values[var][endweek] = np.nan
                else:
                    values[var][endweek] = s/n if method == 'average' else s
            return TimeSeries('time', values)

    @classmethod
    def read_csv(cls, path_file:str, data_type:str='cross', na:any='', index:str='index')->TimeSeries:
        data = Data.read_csv(path_file, data_type, na, index)
        return data.to_timeseries()

    @classmethod
    def read_text(cls, path_file:str, data_type:str='cross', na:any='', index:str='index')->TimeSeries:
        data = Data.read_text(path_file, data_type, na, index)
        return data.to_timeseries()

    def select_variables(self, vars: list[str] = []) -> TimeSeries:
        res = super().select_variables(vars)
        return TimeSeries('time', res.values)
    
    def select_index(self, index: list) -> TimeSeries:
        res = super().select_index(index)
        return TimeSeries('time', res.values)

    def add_data(self, new_data: Data = None) -> TimeSeries:
        super().add_data(new_data)
        self.sort()
        self.dates = self.index()
        return self

    @staticmethod
    def dates_of_a_period(start_date:str, end_date:str, date_type:str='daily')->list[str]:
        dates = []
        if date_type == 'daily':
            if '-' in start_date and '-' in end_date:
                separator = '-'
                start = jdatetime.date(*[int(p) for p in start_date.split('-')])
                end = jdatetime.date(*[int(p) for p in end_date.split('-')])
            elif '/' in start_date and '/' in end_date:
                separator = '/'
                start = jdatetime.date(*[int(p) for p in start_date.split('/')])
                end = jdatetime.date(*[int(p) for p in end_date.split('/')])
            else:
                raise ValueError(f"Error! seprator must be '-' or '/'.")
            date = start
            while (end-date).days>=0:
                dates.append(date.strftime('%Y-%m-%d').replace('-',separator))
                date += jdatetime.timedelta(1)
        elif date_type == 'weekly':
            if '-' in start_date and '-' in end_date:
                separator = '-'
                start = jdatetime.date(*[int(p) for p in start_date.split('-')])
                end = jdatetime.date(*[int(p) for p in end_date.split('-')])
            elif '/' in start_date and '/' in end_date:
                separator = '/'
                start = jdatetime.date(*[int(p) for p in start_date.split('/')])
                end = jdatetime.date(*[int(p) for p in end_date.split('/')])
            else:
                raise ValueError(f"Error! seprator must be '-' or '/'.")
            date = start
            while (end-date).days>=0:
                dates.append(date.strftime('%Y-%m-%d').replace('-',separator))
                date += jdatetime.timedelta(7)
        elif date_type == 'monthly':
            if '-' in start_date and '-' in end_date:
                separator = '-'
                y_, m_ = [int(p) for p in start_date.split('-')]
                _y, _m = [int(p) for p in end_date.split('-')]
            elif '/' in start_date and '/' in end_date:
                separator = '/'
                y_, m_ = [int(p) for p in start_date.split('/')]
                _y, _m = [int(p) for p in end_date.split('/')]
            else:
                raise ValueError(f"Error! seprator must be '-' or '/'.")
            for y in range(y_, _y+1):
                st = m_ if y==y_ else 1
                en = _m+1 if y==_y else 13
                for m in range(st, en):
                    date = f'{y}{separator}0{m}' if m<10 else f'{y}{separator}{m}'
                    dates.append(date)
        elif date_type == 'seasonal':
            if '-' in start_date and '-' in end_date:
                separator = '-'
                y_, s_ = [int(p) for p in start_date.split('-')]
                _y, _s = [int(p) for p in end_date.split('-')]
            elif '/' in start_date and '/' in end_date:
                separator = '/'
                y_, s_ = [int(p) for p in start_date.split('/')]
                _y, _s = [int(p) for p in end_date.split('/')]
            else:
                raise ValueError(f"Error! seprator must be '-' or '/'.")
            for y in range(y_, _y+1):
                st = s_ if y==y_ else 1
                en = _s+1 if y==_y else 5
                for s in range(st, en):
                    dates.append(f'{y}{separator}{s}')
        elif date_type == 'annual':
            try:
                for date in range(start_date, end_date+1):
                    dates.append(date)
            except:
                raise ValueError(f"Error! {start_date} or {end_date} are not integer.")
        return dates

    def dropna(self, vars: list[str] = []) -> None:
        super().dropna(vars)
        self.reset_date_type()

    @classmethod
    def read_xls(cls, path_file:str, data_type:str='cross', na:any='', index:str='index')->TimeSeries:
        data = Data.read_xls(path_file, data_type, na, index)
        data = data.to_timeseries()
        return data

    @classmethod
    def read_excel(cls, path_file:str, sheet:str='', data_type:str='cross', na:any='', index:str='index',
                    first_row:int=0, first_col:int=0)->TimeSeries:
        data = Data.read_excel(path_file, sheet, data_type, na, index, first_row, first_col)
        data = data.to_timeseries()
        return data
        
    def sort(self, key:str='', ascending:bool=True):
        super().sort(key, ascending)
        self.dates.sort()

    def add_index(self, index: list):
        super().add_index(index)
        self.dates = self.index()
        self.complete_dates()

    def to_growth(self, vars:list[str]=[], lag:int=1, is_average:bool=False, 
                    complete_dates:bool=False, print_progress:bool=True)->TimeSeries:
        if vars==[]:
            vars = self.variables()
        else:
            vars = [var for var in vars if var in self.variables()]
        if complete_dates:
            self.complete_dates()
        if print_progress:
            start_time = time.perf_counter()
        values, v = {}, 0
        for var in vars:
            v += 1
            values[var] = {}
            for i in range(len(self.dates)):
                if not is_average:
                    if not (is_nan(self.values[var][self.dates[i-lag]], True) or 
                            is_nan(self.values[var][self.dates[i]], True)):
                        if self.values[var][self.dates[i-lag]] != 0:
                            if self.dates[i] == '1401-05-28' and var == 'gold-irr':
                                print((self.values[var][self.dates[i]] /
                                       self.values[var][self.dates[i-lag]])-1)
                            values[var][self.dates[i]] = (self.values[var][self.dates[i]]/
                                                self.values[var][self.dates[i-lag]])-1
                        else:
                            values[var][self.dates[i]] = np.nan
                    else:
                        values[var][self.dates[i]] = np.nan
                else:
                    s, s_lag = 0, 0
                    n, n_lag = 0, 0
                    for l in range(lag):
                        if is_nan(self.values[var][self.dates[i-l]]):
                            s += self.values[var][self.dates[i-l]]
                            n += 1
                        if is_nan(self.values[var][self.dates[i-lag-l]]):
                            s_lag += self.values[var][self.dates[i-lag-l]]
                            n_lag += 1
                    if n!=0 and n_lag!=0:
                        values[var][self.dates[i]] = (s/n)/(s_lag/n_lag)-1
                    else:
                        values[var][self.dates[i]] = np.nan
                if print_progress:
                    left_time = time.perf_counter() - start_time
                    total_time = left_time * (len(vars)*len(self.dates))/((v-1)*len(self.dates)+i+1)
                    remain_time = total_time-left_time
                    left_time = seconds_to_days_hms(left_time)
                    remain_time = seconds_to_days_hms(remain_time)
                    print(f"{v} of {len(vars)} variables ({v/len(vars)*100:.2f}%) and {i+1} of {len(self.dates)} dates of {var} ({(i+1)/len(self.dates)*100:.2f}%). left: {left_time}, remain: {remain_time}.", end='\r')
        if print_progress:
            print('All growths are calculated', ' '*80)
        data = TimeSeries(values=values)
        data.complete_dates()
        return data



