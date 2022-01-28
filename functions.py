from typing import Union
import numpy as np

def number_of_digits(n:int, decimals:float=0)->int:
    return len(str(int(n*10**decimals)/10**decimals))

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
    return f'{days} days {hours}:{minutes}:{int(secs*10**decimals)/10**decimals}'

def is_numeric(x:Union[str, int, float, bool])->bool:
    if type(x)==int or type(x)==float:
        return True
    return False

def is_numeric_str(text:str)->bool:
    text.replace(' ', '')
    if text[0] == '-':
        text = text[1:]
    return min([i.isdigit() for i in text.split('.') if i!=''])
