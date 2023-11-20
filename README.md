# survey_stats Package

survey_stats is a simple, light and usefull package for data processing and statistics for emprical studies that tries to use research logic instead of data logic.
You can use [Github](https://github.com/mohammad-ali-barati/survey_stats/)

[User Guide](https://raw.githubusercontent.com/mohammad-ali-barati/survey_stats/main/User%20Guide.docx)

This package is under development, but until now it covers the most important tools needed for research, such as weighting non-random samples, preparing pivot tables in the sample, defining new variables using specific formulas or rules, filtering data, creating dummy variables, even reading and writing files such as csv, text, excel, access, ..., or saving and loading data by Pickle package, which makes the speed of reading and writing data much faster than files. In the latest version, it is possible to read and write large files without adding them all to RAM.

In this package, in addition to the tools will be developed separately, it is possible that the modules of the two famous packages, 'scikit-learn' and 'statsmodel', will be available with the logic of the current package. Until now, the exclusive tools of this package include ols and tree-regression. However, logistic and multinomial logistic regressions are also available from the above two packages.

# Install

``pip install survey-stats``

# Data Structure

```python3
from survey_stats.data_process import Data_Types, Data 
values = {
	'name': {1: 'Cyrus', 2: 'Mandana', 3: 'Atossa'},
	'age': {1: 32, 2: 65, 3: 40},
	'sex': {1:'male', 2: 'female', 3: 'female'}
}
data = Data(Data_Types.cross, values)
# or
data = Data('cross', values)
```

# Pandas

Since the Pandas package is familiar to data analysts, it is necessary to explain that the data structure in this package is very close to the data structure in Pandas and they can be easily converted to each other. However, the current package has been tried to use research logic instead of data logic, which is more understandable and simple for researchers.

- pandas.DataFrame --> survey_stats.Data

```python3
from survey-stats.data_process import Data
# df = a DataFrame of Pandas
data = Data(values=df.to_dict())
```

- survey_stats.Data --> pandas.DataFrame

```python3
import pandas as pd
# data = an object of Data
df = pd.DataFrame(data.values)
```

# Modules Structure

survey-stats

|______data_process

-------|____Data

-------|____TimeSeries(Data)

-------|____Sample

-------|____DataBase

|______basic_model

-------|____Model

-------|____Formula

-------|____Formulas

|______date

-------|____Date

|______functions

|______linear_regressions

-------|____ols

------------|____Model

------------|____Equation

|______classification

-------|____tree_based_regression

------------|____Model

------------|____Equation

|______statsmodels

-------|____logit

------------|____Model

------------|____Equation

-------|____multinominal_logit

------------|____Model

------------|____Equation

|______sklearn

-------|____ols

------------|____Model

------------|____Equation

-------|____logit

------------|____Model

------------|____Equation

-------|____multinominal_logit

------------|____Model

------------|____Equation

# data_process

## Data

some methods on data:

* dtype
* set_dtype
* to_str
* variables
* items
* index
* fix_index
* set_index
* set_names
* select_variables
* select_index
* drop
* drop_index
* add_a_dummy
* add_dummies
* dropna
* drop_all_na
* value_to_nan
* to_numpy
* add_data
* transpose
* count
* add_trend
* fillna
* fill
* sort
* add_a_variable
* to_timeseries
* line_plot
* add_index
* add_a_group_variable
* load and dump
* read_text, to_text, and add_to_text
* read_csv, to_csv, and add_to_csv
* read_xls and to_xls
* read_excel and to_excel
* read_access and to_access

```python3
print(data)
variables_name = data.variables()
index = data.index()
data.set_index('id', drop_var=False)
data.set_names(['w1','w2'], ['weights1', 'weights2'])
new_data = data.select_variables(['w1','w2'])
new_data = data.select_index(range(50,100))
data.drop(['year'])
dummy = data.add_a_dummy([['height', '>', 160], ['height', '<=', 180]])
dummy = data.add_dummies([
	[('height', '>', 160), ('height', '<=', 180)],
	[('weight', '>', 60), ('height', '<=', 80)]
	])
data.dropna(['height', 'weight'])
num = data.to_numpy()
data.add_data(data_new)
data_t = data.transpose()
data.to_csv('data2.csv')
```

## Sample

sample is sub-set of a data.

some method on Sample:

* get_data
* split
* get_weights
* group
* Stats: weight, sum, average, var, std, distribution, median, mode, correl, min, max, percentile, gini

```python3
from survey_stats.data_process import Data, Sample
s = Sample(data, [0,5,6,10])
data_s = s.get_data()
train_sample, test_sample = main_sample.split(0.7,['train', 'test'], 'start')
# weighting
cond = [
	[('sex','=', 'female'),('age','<=',30)],
	[('sex','=', 'female'),('age','>',30)],
	[('sex','=', 'male'),('age','<=',30)],
	[('sex','=', 'male'),('age','>',30)]
	]
totals = [
	50,
	150,
	45,
	160
	]

sample = Sample(data, data.index())
sample.get_weights(cond, totals)
print(sample.data)
```

## TimeSeries

timeseries is a special type of Data that index is 'jalali' date.

methods:

* type_of_dates
* complete_dates
* reset_date_type
* to_monthly
* to_daily
* to_weekly
* to_annual
* to_growth
* to_moving_average
* to_lead
* to_lag

## DataBase

database a dict of some Data: {'name':Data, ...}

methods:

* dump
* load
* table_list
* variable_list
* query

# basic_model

## Formula

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

methods:

* split
* filter
* calculate

```python3
from survey_stats.basic_model import Formula
f1 = Formula('p=a')
f2 = Formula('3*x**2+p*x+x')
f3 = Formula('log(year)')
# calculate
data_new = f1.calculate(data)
# split
splits = f2.split() #-> ['3*x**2', 'p*x', 'x'] as Formulas
data_new = f2.split().calculate_all(data)
#filter
f = Formula('year>1397')
after1397 = f.filter(1,data)
```

## Formulas

A list of Formula.

methods:

* calculate_all

# linear_regressions.ols

Linear regression consists of a equation, which are numerically independent variables and are combined linearly with each other.
Categorical variables are converted to dummy variables and then used as a numerical variable in the model. We use the Formula and Formulas class to construct these variables.
Simple regression is a linear regression with a numerically dependent variable that is estimated by the least squares method.
In logistic regression, the dependent variable is a binary variable, or a numerical variable consisting of zeros and ones, or a categorical variable with only two values.

## Model

methods:

* estimate
* estimate_skip_collinear
* estimate_most_significant

## Equation

methods:

* anova
* table
* dump
* load
* wald_test
* forecast

```python3
from survey_stats.linear_regressions import simple_regression, logistic_regression
model1 = simple_regression.Model('y', '1 + x1 + x2 + x2**2')
model2 = logistic_regression.Model('y', '1 + x1 + x2 + x2**2')
# samples of s_trian, s_test have already been defined.
eq1 = model1.estimate(s_train)
print(eq1)
data_f = eq1.forecast(s_test)
print(eq1.goodness_o_fit(s_test)
eq1.save('test')
# or instead of estimating a model, you can load a previously estimated model, and use it to predict.
eq2 = Equation.load('test')
eq2.goodness_of_fit(s_test)
```

# classification.tree_based_regression

Decision tree is one of the modelling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Decision trees are among the most popular machine learning algorithms given their intelligibility and simplicity.
Of course, since we also have regressions with discrete variables, such as logistic regressions, so in this package we have included both regression trees and classification trees in the tree_based_regression module.
Currently, the sklearn package is used for the regression tree and the classification tree. But this package does not work by categorical variables, and this is very restrictive for survey researches because many of the variables in this researches are categorical. It also does not take into account the weight of the observations in its calculations. Therefore, this package has been significantly developed compared to it.

## Model

methods:

* estimate

## Equation

methods:

* forecast
* goodness_of_fit
* first_nodes
* plot
* dump
* load

```python3
from survey_stats import tree_based_regression
model = tree_based_regression.Model(dep_var, indep_vars, min_sample=25, method=method)
eq = model.estimate(s_total, True, False)
print(eq)
print(eq.full_str)
# sample of s_test have already been defined.
forecast_test_leaf = eq.forecast(s_test, name='sample', output ='leaf')
forecast_test_dist = eq.forecast(s_test, name='sample', output ='dist')
forecast_test_point = eq.forecast(s_test, name='sample', output ='point')
# sample of s_total have already been defined.
eq.goodness_of_fit(s_total)
eq.save('total')
# or instead of estimating a model, you can load a previously estimated model, and use it to predict.
eq2 = Equation.load('total')
eq2.goodness_of_fit(s_total)
```

# sklearn.logit

## Model

methods:

* estimate
* estimate_skip_collinear

## Equation

methods:

* dump
* load
* forecast

# sklearn.multinominal_logit

## Model

methods:

* estimate
* estimate_skip_collinear

## Equation

methods:

* dump
* load
* forecast

# sklearn.ols

## Model

methods:

* estimate
* estimate_skip_collinear
* estimate_most_significant

## Equation

methods:

* anova
* table
* dump
* load
* wald_test
* forecast

# statsmodels.logit

## Model

methods:

* estimate

## Equation

methods:

* dump
* load
* forecast
* table

# statsmodels.multinominal_logit

## Model

methods:

* estimate

## Equation

methods:

* dump
* load
* forecast
* table
