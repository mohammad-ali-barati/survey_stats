from __future__ import annotations
import jdatetime

class Date:
    types = ['daily', 'weekly', 'monthly', 'seasonal', 'annual']
    def __init__(self, date:str|int, date_type:str='monthly') -> None:
        if type(date)==str:
            date = date.replace('/', '-').replace(' ','')
            if type in ['daily', 'weekly', 'monthly']:
                date = '-'.join([date.split('-')[0]]+[f'0{x}' if int(x)<10 else x for x in date.split('-')[1:]])
        self.date = date
        if not date_type in Date.types:
            raise ValueError(f"Error! '{date_type}' is not standard. ({Date.types})")
        self.type = date_type
        self.date_type = date_type

    def __str__(self) -> str:
        return self.date
    
    def next(self, n:int)->Date:
        if self.type == 'daily':
            return Date(date = (jdatetime.date(*[int(x)
                                     for x in self.date.split('-')])
                    +jdatetime.timedelta(n)).strftime('%Y-%m-%d'),
                    date_type=self.type)
        elif self.type == 'weekly':
            return Date(date = (jdatetime.date(*[int(x)
                                     for x in self.date.split('-')])
                    +jdatetime.timedelta(n*7)).strftime('%Y-%m-%d'),
                    date_type=self.type)
        elif self.type == 'monthly':
            y, m = [int(x) for x in self.date.split('-')]
            if (months:=12*y+m+n) % 12 > 0:
                y, m = months//12, months%12
            else:
                y, m = months//12-1, 12,
            return Date(date = f'{y}-0{m}',date_type=self.type) if m<10 else \
                    Date(date = f'{y}-{m}',date_type=self.type)
        elif self.type == 'seasonal':
            y, s = [int(x) for x in self.date.split('-')]
            if (seasons:=4*y+s+n) % 4 > 0:
                y, s = seasons//4, seasons%4
            else:
                y, s = seasons//4-1, 4
            return Date(date = f'{y}-{s}',date_type=self.type)
        elif self.type=='annual':
            return Date(date = self.date + n,date_type=self.type)

    def before(self, n:int)->Date:
        return self.next(-n)

    def __sub__(self, other:Date)->int:
        if self.type == 'daily':
            return (jdatetime.date(*[int(x)
                                     for x in self.date.split('-')]) -
                    jdatetime.date(*[int(x)
                                     for x in other.date.split('-')])).days
        elif self.type == 'weekly':
            return (jdatetime.date(*[int(x)
                                     for x in self.date.split('-')]) -
                    jdatetime.date(*[int(x)
                                     for x in other.date.split('-')])).days/7
        elif self.type == 'monthly':
            y, m = [int(x) for x in self.date.split('-')]
            y1, m1 = [int(x) for x in other.date.split('-')]
            return 12*y+m-12*y1-m1
        elif self.type == 'seasonal':
            y, s = [int(x) for x in self.date.split('-')]
            y1, s1 = [int(x) for x in other.date.split('-')]
            return 4*y+s-4*y1-s1
        elif self.type=='annual':
            return self.date - other.date
