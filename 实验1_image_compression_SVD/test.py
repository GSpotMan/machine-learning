import math
import pandas as pd
pd.options.display.float_format = '{:,.8f}'.format
e = math.e


def g_s(x):
    return e**x + x -7

table=[]

table=[1.5]
x = 1.5
x_l = 1
x_h = 2
while abs(g_s(x) - x) > 0.00000001:
    x = x - g_s(x)*(x-x_l)/(g_s(x) - g_s(x_l))
    if g_s(x)*g_s(x_l)<0:
        x_h = x
    if g_s(x)*g_s(x_l)>0:
        x_l = x


def s_iteration(min_value, max_value, table):
    x = min_value - g_s(min_value)*(max_value-min_value)/(g_s(max_value) - g_s(min_value))
    if abs(g_s(x)) < 0.00000001:
        return table
    if g_s(x)*g_s(min_value) < 0:
        table.append([x])
        dichotomy_iteration(min_value, x, table)
    if g_s(x)*g_s(min_value) > 0:
        table.append([x])
        dichotomy_iteration(x, max_value, table)

试位法



    table.append(x)

df = pd.DataFrame(table, columns=['value'], dtype=float)
print(df)