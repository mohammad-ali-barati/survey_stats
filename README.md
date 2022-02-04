survey_stats Package
====================

survey_stats is a simple and powerfull package for data processing and statistics for emprical studies.
You can use [Github](https://github.com/mohammad-ali-barati/survey_stats/)
[User Guide](https://raw.githubusercontent.com/mohammad-ali-barati/survey_stats/main/User%20Guide.docx)

survey_stats is a Python package that provides a set of statistical tools for emprical studies. This package includes some tools for data processing and estimation of observation weights, linear and logistic regressions, as well as tree regression and classification that are adapted for both numerical and categorical variables. data structure in this package is compatible with pandas package.

Install
-------
``pip install survey-stats``


Import Data
-----------

```python3
from survey_stats.data_processing import Data_Types, Data 
values = {
	'name': {1: 'Cyrus', 2: 'Mandana', 3: 'Atossa'},
	'age': {1: 32, 2: 65, 3: 40},
	'sex': {1:'male', 2: 'female', 3: 'female'}
}
data = Data(Data_Types.cross, values)
# or
data = Data('cross', values)
# or
data = Data.read_csv('data.csv')
```

Pandas <-> Data
---------------
- pd.DataFrame --> Data

```python3
# df = a DataFrame of Pandas
values = df.to_dict()
data = Data('cross', values)
```

- Data --> pd.DataFrame

```python3
import pandas as pd
# data = an object of Data
df = pd.DataFrame(data.values)
```

Data Processing
---------------
some functions on data:

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

Sample
------
sample is sub-set of a data.

```python3
from survey_stats.data_processing import Data, Sample
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

Variable
--------
A variable is a series of a data set. Variables can be numeric or categorical types.

```python3
from survey_stats.basic_model import Variable_Types, Variable
x = Variable('x', Variable_Types.numeric)
# or
x = Variable('x', 'numeric')
# or
x = Variable('x')
# or base on values in a data set
x = Variable.from_data(data, 'x')
# Values of a variable without repeating in a sample
values = x.values(sample)
# all sub-sets of values
x_values = x.values_set(sample)

# stats: mean, std, count, tss, distribution, sum
#	 median, mode, min, max, percentile
print(x.stats.mode(sample))

# mapping
old_values = [['under diploma', 'diploma'], ['bachelor', 'master', 'phd']]
new_values = ['non-academic', 'academic']
# or
old_values = ['<2012', '<=2015', '<=2017', '>2017']
new_values = ['before sanction', 'first oil sanction',
		'JCPOA period', 'second oil sanction']
v1 = Variable('year', 'numeric')
v2_data = v1.map(Sample(data), old_values,new_values,other='-')
```

Formula
-------
Formula is a mathematical relation. Formula class allows you to solve mathematical relationships on data, thereby either defining new variables or filtering the data set according to the values of a formula.

mathematic functions: 
only on time series variables: 
-	lag(‘variable_name’,’number of lags’)= var[i-lags]
-	dif(‘variable_name’,’number of lags’)=var[i]-var[i-lags]
-	gr(‘variable_name’,’number of lags’)=var[i]/var[i-lags]-1
on all variables:
-	log(‘variable_name’): Napierian logarithm (loge(x))
-	exp(‘variable_name’): Exponential function (ex)
statistic functions:
-	sum(‘variable_name’): summation of variable in data.
-	count(‘variable_name’): total non-blank values of variable in data.
-	mean(‘variable_name’): weighted mean of variable in data.
-	std(‘variable_name’): weighted standard deviation of variable in data.
-	min(‘variable_name’): minimum of values of variable in data.
--	max(‘variable_name’) maximum of values of variable in data.


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

FormulaTable
-----
Table is a pivot table.

```python3
from survey_stats.basic_model import FormulaTable
table = FormulaTable('p',['count(x)', 'mean(x)', 'std(x)'],Sample(data))
table_data = table.to_data()
table.plot()	# must install matplotlib
```

tree_based_regression
---------------------
Decision tree is one of the modelling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Decision trees are among the most popular machine learning algorithms given their intelligibility and simplicity.
Of course, since we also have regressions with discrete variables, such as logistic regressions, so in this package we have included both regression trees and classification trees in the tree_based_regression module.
Currently, the sklearn package is used for the regression tree and the classification tree. But this package does not work by categorical variables, and this is very restrictive for survey researches because many of the variables in this researches are categorical. It also does not take into account the weight of the observations in its calculations. Therefore, this package has been significantly developed compared to it.


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

linear_regressions
------------------
Linear regression consists of a equation, which are numerically independent variables and are combined linearly with each other.
Categorical variables are converted to dummy variables and then used as a numerical variable in the model. We use the Formula and Formulas class to construct these variables.
Simple regression is a linear regression with a numerically dependent variable that is estimated by the least squares method.
In logistic regression, the dependent variable is a binary variable, or a numerical variable consisting of zeros and ones, or a categorical variable with only two values.


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