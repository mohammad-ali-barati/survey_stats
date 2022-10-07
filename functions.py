from __future__ import annotations
from typing import Union
import numpy as np
import jdatetime, datetime

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

def seconds_to_days_hms(seconds:float)->str:
    # decs = seconds - int(seconds)
    days = int(seconds // (24*60*60))
    seconds_remind = seconds-days*24*60*60
    hours = int(seconds_remind // (60*60))
    seconds_remind = seconds_remind - hours*60*60
    minutes = int(seconds_remind // 60)
    seconds_remind = seconds_remind - minutes*60 
    res = ''
    if days != 0:
        res += f'{days}days '
    if hours != 0:
        res += f'{hours}h '
    if minutes != 0:
        res += f'{minutes}m '
    if seconds_remind != 0:
        res += f'{seconds_remind:.0f}s'
    return res

def is_numeric(x:Union[str, int, float, bool])->bool:
    if type(x)==int or type(x)==float:
        return True
    return False

def is_numeric_str(text:str)->bool:
    text.replace(' ', '')
    if text == '':
        return False
    elif len(text.split('e'))==2:
        if is_numeric_str(text.split('e')[0]) and is_numeric_str(text.split('e')[1]):
            return True
    elif len(text.split('E'))==2:
        if is_numeric_str(text.split('E')[0]) and is_numeric_str(text.split('E')[1]):
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

def xls_read(content:str)->list[list]:
    content = content.replace('<td>', '<th>').replace('</td>', '</th>')
    if not ('<thead>' in content and '</thead>'):
        raise ValueError(f"Error! content don't include 'thead' tag.")
    if not ('<tbody>' in content and '</tbody>'):
        raise ValueError(f"Error! content don't include 'tbody' tag.")
    res = []
    row = []
    headers = content.split('<thead>')[1].split('</thead>')[0].split('<th>')
    for header in headers[1:]:
        row.append(header.split('</th>')[0].strip())
    res.append(row)

    rows = content.split('<tbody>')[1].split('</tbody>')[0].split('<tr>')
    for row in rows[1:]:
        cols = row.split('<th>')
        row = []
        for col in cols[1:]:
            row.append(col.split('</th>')[0].strip())
        res.append(row)
    return res

def to_formated_str(value:str | int | float, decimals:int=4)->str:
    if type(value) == str:
        return value
    elif np.isnan(value) or value.imag != 0:
        return 'nan'
    else:
        if 'e' in str(value):
            parts = str(value).split('e')
            if len(parts)>2:
                raise ValueError(f"Error! {value} cannot split by 'e'.")
            v, p = parts
            if '.' in v:
                parts = v.split('.')
                if len(parts)>2:
                    raise ValueError(f"Error! {value} cannot split by '.'.")
                i,d = parts
            else:
                i,d = v,'0'
            if p[0]!='-':
                value = f'{i}{d[:int(p[1:])]}.{d[int(p[1:]):]}'
            else:
                if len(i)<=-int(p):
                    i = '0' * (-int(p)-len(i)+1) + i
                value = f'{i[:int(p)]}.{i[int(p):]}{d}'

        splits = str(value).split('.')
        if len(splits) == 2:
            num, dec = splits
        elif len(splits) == 1:
            num, dec = splits, 0
        g, r = int(len(num)/3), len(num)%3
        num_str = num[:r]
        for d in range(g):
            num_str += ',' + num[r+d*3:r+d*3+3]
        if num_str[0]==',':
            num_str = num_str[1:]
        if dec != 0:
            if len(dec)>decimals:
                if int(dec[decimals]) >= 5:
                    if int(dec[decimals-1])+1<10:
                        num_str += '.' + dec[:decimals-1] + str(int(dec[decimals-1])+1)
                    else:
                        num_str += '.' + dec[:decimals-2] + \
                            str(int(dec[decimals-2])+1) + '0'
                else:
                    num_str += '.' + dec[:decimals]
            else:
                num_str += '.' + dec[:decimals]
        return num_str[0] if type(num_str)==list else num_str

def number_to_jdate(num:int, zero_date:str='1278-10-09')->str:
    if '-' in zero_date and len(zero_date.split('-')) == 3:
        y, m, d = zero_date.split('-')
        sep = '-'
    elif '/' in zero_date and len(zero_date.split('/')) == 3:
        y, m, d = zero_date.split('/')
        sep = '/'
    else:
        raise ValueError(f"Error! seprator of date must be - or /")
    date = jdatetime.datetime(int(y), int(m), int(d)) + jdatetime.timedelta(num)
    return date.strftime(f'%Y{sep}%m{sep}%d')

def number_to_date(num: int, zero_date: str = '1900-01-01')->str:
    if '-' in zero_date and len(zero_date.split('-')) == 3:
        y, m, d = zero_date.split('-')
        sep = '-'
    elif '/' in zero_date and len(zero_date.split('/')) == 3:
        y, m, d = zero_date.split('/')
        sep = '/'
    else:
        raise ValueError(f"Error! seprator of date must be - or /")
    date = datetime.datetime(int(y), int(m), int(d)) + \
        datetime.timedelta(num-2)
    return date.strftime(f'%Y{sep}%m{sep}%d')

def is_nan(value:any, is_number:bool=False)->bool:
    nan = False
    if is_numeric(value):
        if np.isnan(value):
            nan = True
    elif is_number:
        nan = True
    return nan

def str_to_jdate(date_str:str)->jdatetime.date:
    seprator = '-' if '-' in date_str else '/' if '/' in date_str else ''
    if seprator == '':
        raise ValueError(f"Error! seprator must be '-' or '/'. {date_str} is not standard")
    splits = date_str.split(seprator)
    if len(splits) != 3:
        raise ValueError(
            f"Error! part of date must be 3 (year, month, day). {date_str} is not standard")
    else:
        try:
            y, m, d = [int(x) for x in splits]
        except:
            raise ValueError(
                f"Error! part of date must be integer (year, month, day). {date_str} is not standard")
        return jdatetime.date(y, m, d)

def to_days_later(date_str:str, days:int=1)->str:
    s = '-' if '-' in date_str else '/' if '/' in date_str else ''
    jdate = str_to_jdate(date_str)
    next_date = jdate + jdatetime.timedelta(days)
    return next_date.strftime(f'%Y{s}%m{s}%d')

def to_days_ago(date_str: str, days: int = 1) -> str:
    s = '-' if '-' in date_str else '/' if '/' in date_str else ''
    jdate = str_to_jdate(date_str)
    next_date = jdate - jdatetime.timedelta(days)
    return next_date.strftime(f'%Y{s}%m{s}%d')

def to_weekend(date_str: str) -> str:
    s = '-' if '-' in date_str else '/' if '/' in date_str else ''
    jdate = str_to_jdate(date_str)
    days = 6-jdate.weekday()
    next_date = jdate + jdatetime.timedelta(days)
    return next_date.strftime(f'%Y{s}%m{s}%d')

def today()->str:
    return jdatetime.date.today().strftime(f'%Y-%m-%d')

