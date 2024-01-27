from __future__ import annotations
from typing import Union
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import jdatetime, datetime, time, math
import random

def number_of_digits(n:int, decimals:float=0)->int:
    try:
        return len(str(int(n*10**decimals)/10**decimals))
    except:
        return 5

def subsets(a_set:list, deep:int=-1, randomly:bool=False)->list:
    '''
    deep is a level base on len of subsets.\n
    randomly == True to select randomly subsets per level as
    '''
    subs = []
    if deep==-1:
        if len(a_set) == 0:
            return [[]]
        else:
            s = subsets(a_set[:-1])
            return s + [i + [a_set[-1]] for i in s]
    elif not randomly:
        if a_set == []:
            return []
        if deep == 0:
            return a_set
        else:
            subs = [a_set[1:]]
            for i in range(1, len(a_set)-1):
                subs.append(a_set[:i]+a_set[i+1:])
            subs.append(a_set[:-1])
            if deep > 1:
                for sub in subs[1:]:
                    subs.extend(subsets(sub, deep-1, randomly)[1:])
            subs = [a_set] + subs
    else:
        if a_set == []:
            return []
        if deep == 0:
            return a_set
        else:
            subs = [a_set[1:]]
            for i in range(1, len(a_set)-1):
                subs.append(a_set[:i]+a_set[i+1:])
            subs.append(a_set[:-1])
            i = 1
            while  deep >= 2:
                for sub in subs[i:]:
                    subsubs = subsets(sub, 1, True)
                    subs.append(subsubs[random.randint(1,len(subsubs)-1)])
                    i += 1
                deep -= 1
            subs = [a_set] + subs
    subs.sort(key=lambda x:len(x), reverse=True)
    return subs

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
    if isinstance(x, int) or isinstance(x, float):
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
        if len(text.split('.'))==2:
            return min([i.isdigit() for i in text.split('.') if i!=''])
        else:
            return False
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
    elif date_type == 'annual':
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
    try:
        if value != 0 and (abs(value)<10**(-decimals) or abs(value)>=10**9):
            res = f'{value:,.{decimals}e}'
        else:
            try:
                decimals = min(decimals, len(str(value).split('.')[1]))
            except:
                decimals = 0
            res = f'{value:,.{decimals}f}'
        return res
    except:
        return str(value)

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
    if is_numeric(value):
        if np.isnan(value):
            return True
    elif is_number:
        return True
    return False

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

def progress(start_time:int, iteration:int, iterations:int, len_previous_log:int,
             state:str='', cut_state:bool=True, indent:int = 5):
    elapsed = time.perf_counter() - start_time
    remaining = elapsed * (iterations/(iteration+1) - 1)
    elapsed = seconds_to_days_hms(elapsed)
    remaining = seconds_to_days_hms(remaining)
    log = ' '*indent 
    if state != '':
        if not isinstance(state, str):
            state = str(state)
        if cut_state:
            state = state[:15].ljust(15)
        log += f'{state}-'
    log += f'{iteration+1} of {iterations} ({(iteration+1)/iterations*100:.2f}%). elapsed: {elapsed}. remaining: {remaining}.'
    log += ' '*(max(len_previous_log, len(log))-(len_previous_log:=len(log)))
    if iteration == iterations-1:
        print(log)
    else:
        print(log, end='\r')
    return len_previous_log

def clear_memory():
    for var in dir():
        if not var.startswith('__'):
            del globals()[var]

def match_str(a:str, b:str)->bool:
    if isinstance(a, str) and isinstance(b, str):
        a_s, b_s = a.split('*'), b.split('*')
        if len(a_s)==1:
            if len(b_s)==1:
                return a==b
            else:
                if b_s[0]=='':
                    if b_s[1] in a:
                        if len(b_s[1:])>1:
                            return match_str(a[a.find(b_s[1]):], '*'.join(b_s[1:]))
                        else:
                            return a[a.find(b_s[1]):] == b_s[1]
                    else:
                        return False
                elif b_s[-1]=='':
                    if b_s[-2][::-1] in a[::-1]:
                        if len(b_s[:-1])>1:
                            return match_str(a[:len(a)-a[::-1].find(b_s[-2][::-1])], '*'.join(b_s[:-1]))
                        else:
                            return a[:len(a)-a[::-1].find(b_s[-2][::-1])] ==  b_s[0]
                    else:
                        return False
                else:
                    if a[:len(b_s[0])]!=b_s[0] or a[-len(b_s[-1]):]!=b_s[-1]:
                        return False
                    n = len(b_s[0])
                    for w in b_s[1:-1]:
                        if a[n:].find(w) == -1:
                            return False
                    return True
        else:
            if len(b_s)==1:
                return match_str(b, a)
            else:
                if len(a_s)<len(b_s):
                    return match_str(''.join(a_s), '*'.join(b_s))
                elif len(a_s)>len(b_s):
                    return match_str('*'.join(a_s), ''.join(b_s))
                else:
                    return min([a_s[i] in b_s[i] if len(a_s[i])<=len(b_s[i]) else b_s[i] in a_s[i] for i in range(len(a_s))])
    else:
        return a==b

def match_in(a:str, b:list)->bool:
    for x in b:
        if match_str(a, x):
            return x
    return False

def list_to_str(a_list:list, min_lenght:int=50, indent:int=10):
    res = []
    for j, element in enumerate(a_list):
        if j==0:
            line = ' '*indent+f'{element}'
        elif len(line)<min_lenght:
            line += f', {element}'
        else:
            res.append(line)
            line = ' '*indent+f'{element}'
    if res ==[]:
        res.append(line)
    return res

def quantile(x:list, n:int=10)->list:
    a = sorted([y for y in x if not is_nan(y)])
    thre = [a[int(d*len(a)/n)] for d in range(1,n)]
    res = []
    for y in x:
        if is_nan(y):
            res.append(math.nan)
        else:
            k = 1
            if y<thre[0]:
                res.append(k)
            else:
                for j,d in enumerate(thre):
                    if j>0:
                        k += 1
                        if thre[j-1]<=y<d:
                            res.append(k)
                            break
                else:
                    if d<=y:
                        res.append(k+1)
    return res
