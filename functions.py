from __future__ import annotations
from typing import Union
import numpy as np

def number_of_digits(n:int, decimals:float=0)->int:
    try:
        return len(str(int(n*10**decimals)/10**decimals))
    except:
        return 5

def subsets(a_set:set)->set:
    if len(a_set) == 0:
        return [[]]
    else:
        s = subsets(a_set[:-1])
        return s + [i + [a_set[-1]] for i in s]

def to_number(x:str, skip_chars:bool=True)->float:
    j, has_seprator = '', False
    for i in x:
        try:
            if i=='.' and not has_seprator:
                j += i
                has_seprator = True
            elif not i=='.':
                j += str(int(i))
            else:
                if not skip_chars:
                    return np.nan
        except:
            if not skip_chars:
                return np.nan
    if j=='':
        return np.nan
    else:
        return float(j)

def remove_number(x:str)->str:
    j, has_seprator = '', False
    for i in x:
        if not i.isnumeric() and (i!='.' or has_seprator):
            j+=i
        elif not has_seprator and i=='.':
            has_seprator = True
    return j

def split_str_number(x:str)->tuple[str, float]:
    j_str, j_int, has_seprator = '', '', False
    for i in x:
        if i == '.' and not has_seprator:
            j_int += i
            has_seprator = True
        elif not i == '.' and i.isnumeric():
            j_int += str(int(i))
        elif  i=='.' or (i!='.' or  has_seprator):
            j_str += i
    if j_int == '':
        j_int = np.nan
    else:
        j_int = float(j_int)
    return (j_str, j_int)

def check_condition(value:Union[float, str], sign:str, criteria:Union[float,str])->bool:
    err = f"Error! '{sign}' is not use for comparing '{criteria}' and '{value}'."
    if sign == '>=':
        if type(criteria)==float or type(criteria)==int:
            res = value >= criteria
        else:
            raise ValueError(err)
    elif sign == '<=':
        if type(criteria)==float or type(criteria)==int:
            res = value <= criteria
        else:
            raise ValueError(err)
    elif sign == '!=':
        res = value != criteria
    elif sign == '>':
        if type(criteria)==float or type(criteria)==int:
            res = value > criteria
        else:
            raise ValueError(err)
    elif sign == '<':
        if type(criteria)==float or type(criteria)==int:
            res = value < criteria
        else:
            raise ValueError(err)
    elif sign == '=':
        res = value == criteria
    else:
        err = f"Error! '{sign}' must be one of the symbols " + \
                            "'=','!=','>','>=','<','<=', or '!='."
        raise ValueError(err)
    return res

def seconds_to_days_hms(seconds:float, decimals:int=4)->str:
    decs = seconds - int(seconds)
    days = seconds // (24*60*60)
    hours = seconds % (24*60*60) // (60*60)
    minutes = seconds % (24*60*60) % (60*60) // 60
    secs = seconds % (24*60*60) % (60*60) // 60 + decs
    return f'{days:.0f} days {hours:.0f}:{minutes:.0f}:{int(secs*10**decimals)/10**decimals}'

def is_numeric(x:Union[str, int, float, bool])->bool:
    if type(x)==int or type(x)==float:
        return True
    return False

def is_numeric_str(text:str)->bool:
    text.replace(' ', '')
    if len(text.split('e'))==2:
        if is_numeric_str(text.split('e')[0]) and is_numeric_str(text.split('e')[1]):
            return True
    elif text[0] == '-':
        text = text[1:]
    try:
        return min([i.isdigit() for i in text.split('.') if i!=''])
    except:
        return False

def to_float(text:str)->float:
    text.replace(' ', '')
    if len(text.split('e'))==2:
        if is_numeric_str(text.split('e')[0]) and is_numeric_str(text.split('e')[1]):
            return float(text.split('e')[0]) * 10** float(text.split('e')[1])
    if is_numeric_str(text):
        return float(text)
    else:
        return np.nan

def days_of_month(year:int, month:int)->int:
    if 0<month <7:
        return 31
    elif month<12:
        return 30
    elif month==12 and ((1244<int(year)<=1342 and int(year)%33 in [1,5,9,13,17,21,26,30])
                    or (1342<int(year)<=1472 and int(year)%33 in [1,5,9,13,17,22,26,30])):
        return 30
    elif month==12:
        return 29
    else:
        raise ValueError(f"Error month of '{month}' must between 1 to 12.")

def date_to_standard(date:str|int, date_type:str='daily')->str:
    if date_type in ['daily', 'weekly', 'monthly', 'seasonal']:
        if not('/' in date or '-' in date):
            raise ValueError(f"Error! '{date}' must has '/' or '-' as seprator.")
    if date_type in ['daily', 'weekly']:
        splits = date.split('/') if '/' in date else date.split('-')
        if len(splits)!=3:
            raise ValueError(f"Error! the number of splits must be 3.")
        y, m, d = splits
        max_days = days_of_month(int(y), int(m))
        if int(d)<1 or int(d)>max_days:
            raise ValueError(f"Error days of in {y}-{m} must between 1 to {max_days}.")
        m = '0'+m if len(m)==1 else m
        d = '0'+d if len(d)==1 else d
        return f'{y}-{m}-{d}'
    elif date_type == 'monthly':
        splits = date.split('/') if '/' in date else date.split('-')
        if len(splits)!=2:
            raise ValueError(f"Error! the number of splits must be 2.")
        y, m = splits
        m = '0'+m if len(m)==1 else m
        return f'{y}-{m}'
    elif date_type == 'seasonal':
        splits = date.split('/') if '/' in date else date.split('-')
        if len(splits)!=2:
            raise ValueError(f"Error! the number of splits must be 2.")
        y, s = splits
        return f'{y}-{s}'
    elif date_type == 'yearly':
        try:
            return int(date)
        except:
            raise ValueError(f"Error! years must be 'int'.")



