# Importar librerias necesarias 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import math as mth
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import mannwhitneyu


# Cargar los datasets

df_hipotesis = pd.read_csv('hypotheses_us.csv', sep=';')

df_orders = pd.read_csv('orders_us.csv')

df_visits = pd.read_csv('visits_us.csv')

# Limpiar df_hipotesis

df_hipotesis.info()

# limpiar df_orders
df_orders.info()

df_orders['date'] = pd.to_datetime(df_orders['date'], format='%Y-%m-%d') # transformar la columna date a datetime

# Limpiar df_visits
df_visits.info()

df_visits['date'] = pd.to_datetime(df_visits['date'], format='%Y-%m-%d') # transformar la columna date a datetime



# Confirmar que no haya errores en los datos
# Clonar el dataframe df_orders en un nuevo dataframe df_orders_clean
df_orders_copy = df_orders.copy() # clonar el dataframe df_orders en un nuevo dataframe df_orders_clean
visitors_A = set(df_orders[df_orders['group'] == 'A']['visitorId']) # obtener los visitantes del grupo A
visitors_B = set(df_orders[df_orders['group'] == 'B']['visitorId']) # obtener los visitantes del grupo B
# Verificar que no haya visitantes en ambos grupos 
intersecting_visitors = visitors_A & visitors_B # obtener los visitantes que estan en ambos grupos
# Eliminar los visitantes que estan en ambos grupos
df_orders = df_orders[~df_orders['visitorId'].isin(intersecting_visitors)] # eliminar los visitantes que estan en ambos grupos



# Priorizar hipotesis crear ICE y RICE con df_hipotesis
# Crear un dataframe llamado df_hipotesis_ICE a partir de df_hipotesis para puntaje ICE ordenando de mayor a menor
df_hipotesis['ICE'] = df_hipotesis['Impact'] * df_hipotesis['Confidence'] / df_hipotesis['Effort'] # calcular el puntaje ICE
df_hipotesis_ICE = df_hipotesis[['Hypothesis', 'ICE']].sort_values(by='ICE', ascending=False).reset_index(drop=True) # ordenar el dataframe por ICE de mayor a menor


# Crear un dataframe llamado df_hipotesis_RICE a partir de df_hipotesis para puntaje ICE ordenando de mayor a menor
df_hipotesis['RICE'] = df_hipotesis['Reach'] * df_hipotesis['Impact'] * df_hipotesis['Confidence'] / df_hipotesis['Effort'] # calcular el puntaje RICE
df_hipotesis_RICE = df_hipotesis[['Hypothesis', 'RICE']].sort_values(by='RICE', ascending=False).reset_index(drop=True) # ordenar el dataframe por RICE de mayor a menor

# Hacer una grafica comparando cada hipotesis con su puntaje ICE y RICE 
# Crear un dataframe para graficar
df_hipotesis_plot = df_hipotesis[['Hypothesis', 'ICE', 'RICE']].melt(id_vars='Hypothesis', var_name='Metric', value_name='Score') # transformar el dataframe para graficar
# Graficar
plt.figure(figsize=(12, 6))
sns.barplot(x='Score', y='Hypothesis', hue='Metric', data=df_hipotesis_plot, palette='Set2') # graficar
plt.title('ICE and RICE scores for hypotheses')
plt.xlabel('Score')
plt.ylabel('Hypothesis')
plt.legend(title='Metric')
plt.grid(axis='x', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xlim(0, df_hipotesis_plot['Score'].max() + 1) # agregar un limite al eje x
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.show() # mostrar la grafica




# Analisis del test A/B

# Crear una grafica del ingreso acumulado por grupo

# Crear un dataframe para graficar
df_orders_group = df_orders.groupby(['date', 'group']).agg({'revenue': 'sum'}).reset_index() # agrupar por fecha y grupo mostrando la suma de ingresos por dia por grupo
# Agruparlo por semana
df_orders_group['week'] = df_orders_group['date'].dt.isocalendar().week # agregar una columna con la semana
# Agrupar por semana y grupo mostrando la suma de ingresos por semana por grupo
df_orders_group_week = df_orders_group.groupby(['week', 'group']).agg({'revenue': 'sum'}).reset_index() # agrupar por semana y grupo mostrando la suma de ingresos por semana por grupo
# Graficar
plt.figure(figsize=(12, 6))
sns.lineplot(x='week', y='revenue', hue='group', data=df_orders_group_week, palette='Set2') # graficar
plt.title('Ingreso acumulado por grupo semanalmente')
plt.xlabel('Week')
plt.ylabel('Revenue')
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.show() # mostrar la grafica


# Representa gráficamente el tamaño de pedido promedio acumulado por grupo. 
# Crear un dataframe para graficar
df_orders_count = df_orders.groupby(['date', 'group']).agg({'transactionId': 'count'}).reset_index() # agrupar por fecha y grupo mostrando la cantidad de pedidos por dia por grupo
# Agruparlo por semana
df_orders_count['week'] = df_orders_count['date'].dt.isocalendar().week # agregar una columna con la semana
# Agrupar por semana y grupo mostrando la cantidad de pedidos por semana por grupo
df_orders_count_week = df_orders_count.groupby(['week', 'group']).agg({'transactionId': 'sum'}).reset_index() # agrupar por semana y grupo mostrando la cantidad de pedidos por semana por grupo
# Graficar
plt.figure(figsize=(12, 6))
sns.lineplot(x='week', y='transactionId', hue='group', data=df_orders_count_week, palette='Set2') # graficar
plt.title('Tamaño de pedido promedio acumulado por grupo semanalmente')
plt.xlabel('Week')
plt.ylabel('TransactionId')
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.show() # mostrar la grafica


# Representa gráficamente la diferencia relativa en el tamaño de pedido promedio acumulado para el grupo B en comparación con el grupo A. 

# Separar los grupos
df_orders_A_count = df_orders_count_week[df_orders_count_week['group'] == 'A'] # separar el grupo A 
df_orders_B_count = df_orders_count_week[df_orders_count_week['group'] == 'B'] # separar el grupo B
# unificar los dataframes cambiando el nombre de la columna transactionId por transactionId_A y transactionId_B
df_orders_count_week_seperate = df_orders_A_count.merge(df_orders_B_count, on='week', suffixes=('_A', '_B')) 
# Calcular la diferencia relativa
df_orders_count_week_seperate['relative_difference'] = (df_orders_count_week_seperate['transactionId_B'] - df_orders_count_week_seperate['transactionId_A']) / df_orders_count_week_seperate['transactionId_A'] * 100 # calcular la diferencia relativa
# Graficar por semana la diferencia relativa
plt.figure(figsize=(12, 6))
sns.lineplot(x='week', y='relative_difference', data=df_orders_count_week_seperate, palette='Set2') # graficar
plt.title('Diferencia relativa en el tamaño de pedido promedio acumulado para el grupo B en comparación con el grupo A')
plt.xlabel('Week')
plt.ylabel('Relative difference')
plt.grid(axis='y', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.axhline(0, color='red', linestyle='--') # agregar una linea horizontal en 0
plt.show() # mostrar la grafica



# Representa gráficamente la tasa de conversión por grupo. 
# Separar por grupos 
df_orders_count_A_day = df_orders_count[df_orders_count['group'] == 'A'].reset_index() # separar el grupo A
# nombrar columna transactionId por count_orders_A
df_orders_count_A_day.rename(columns={'transactionId': 'count_orders_A'}, inplace=True) # renombrar la columna transactionId por count_orders_A
df_orders_count_B_day = df_orders_count[df_orders_count['group'] == 'B'].reset_index() # separar el grupo B
# nombrar columna transactionId por count_orders_B
df_orders_count_B_day.rename(columns={'transactionId': 'count_orders_B'}, inplace=True) # renombrar la columna transactionId por count_orders_B
df_visits_A_day = df_visits[df_visits['group'] == 'A'].reset_index() # separar el grupo A
df_visits_B_day = df_visits[df_visits['group'] == 'B'].reset_index() # separar el grupo B
# Unir los dataframes por fecha 
df_orders_count_day = df_orders_count_A_day.merge(df_visits_A_day[['date', 'group', 'visits']], on=['date', 'group'], how='left') # unir por fecha y grupo
# nombrar columna visits por count_visits_A
df_orders_count_day.rename(columns={'visits': 'count_visits_A'}, inplace=True) # renombrar la columna visits por count_visits_A
# Crear una columna con la tasa de conversion
df_orders_count_day['conversion_rate'] = df_orders_count_day['count_orders_A'] / df_orders_count_day['count_visits_A'] # calcular la tasa de conversion

# Unir los dataframes por fecha de B
df_orders_count_B_day = df_orders_count_B_day.merge(df_visits_B_day[['date', 'group', 'visits']], on=['date', 'group'], how='left') # unir por fecha y grupo
# nombrar columna visits por count_visits_B
df_orders_count_B_day.rename(columns={'visits': 'count_visits_B'}, inplace=True) # renombrar la columna visits por count_visits_B
# Crear una columna con la tasa de conversion
df_orders_count_B_day['conversion_rate'] = df_orders_count_B_day['count_orders_B'] / df_orders_count_B_day['count_visits_B'] # calcular la tasa de conversion
# Unir los dataframes por fecha 
df_orders_count_day_2 = df_orders_count_day.merge(df_orders_count_B_day[['date', 'group', 'conversion_rate']], on=['date'], how='left') # unir por fecha y grupo
# graficar la tasa de conversion por grupo
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='conversion_rate_x', data=df_orders_count_day_2, label='A', color='blue') # graficar
sns.lineplot(x='date', y='conversion_rate_y', data=df_orders_count_day_2, label='B', color='orange') # graficar
plt.title('Tasa de conversion por grupo')
plt.xlabel('Date')
plt.ylabel('Conversion rate')  
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.show() # mostrar la grafica


# Traza un gráfico de dispersión del número de pedidos por usuario. Haz conclusiones y conjeturas.
# Agrupar por usuario y grupo mostrando la cantidad de pedidos por usuario por grupo
df_orders_user = df_orders.groupby(['date', 'group']).agg({'transactionId': 'count'}).reset_index() # agrupar por usuario y grupo mostrando la cantidad de pedidos por usuario por grupo
# Realizar graficar de dispersión
plt.figure(figsize=(12, 6))
sns.scatterplot(x='date', y='transactionId', hue='group', data=df_orders_user, palette='Set2') # graficar
plt.title('Cantidad de pedidos por usuario por grupo')
plt.xlabel('Date')
plt.ylabel('TransactionId')
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.show() # mostrar la grafica




# Calcular percentiles 95 y 99 de los numeros de pedidos por usuario
# Calcular el percentil 95
percentil_95 = np.percentile(df_orders_user['transactionId'], 95) # calcular el percentil 95
print('El percentil 95 es:', percentil_95) # imprimir el percentil 95
# Calcular el percentil 99
percentil_99 = np.percentile(df_orders_user['transactionId'], 99) # calcular el percentil 99
print('El percentil 99 es:', percentil_99) # imprimir el percentil 99



# Traza un gráfico de dispersión de los precios de los pedidos. Haz conclusiones y conjeturas.
# Agrupar por usuario y grupo mostrando la cantidad de pedidos por usuario por grupo
df_orders_price = df_orders.groupby(['date', 'group']).agg({'revenue': 'sum'}).reset_index() # agrupar por usuario y grupo mostrando la cantidad de pedidos por usuario por grupo
# Realizar graficar de dispersión
plt.figure(figsize=(12, 6))
sns.scatterplot(x='date', y='revenue', hue='group', data=df_orders_price, palette='Set2') # graficar
plt.title('Cantidad de pedidos por usuario por grupo')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend(title='Group')
plt.grid(axis='y', linestyle='--', alpha=0.7) # agregar una rejilla
plt.xticks(rotation=45) # rotar las etiquetas del eje x
plt.tight_layout() # ajustar la grafica
plt.show() # mostrar la grafica


# Calcular percentiles 95 y 99 de los precios de los pedidos
# Calcular el percentil 95
percentil_95 = np.percentile(df_orders_price['revenue'], 95) # calcular el percentil 95
print('El percentil 95 es:', percentil_95) # imprimir el percentil 95
# Calcular el percentil 99
percentil_99 = np.percentile(df_orders_price['revenue'], 99) # calcular el percentil 99
print('El percentil 99 es:', percentil_99) # imprimir el percentil 99





# Encuentra la significancia estadística de la diferencia en la conversión entre los grupos utilizando los datos filtrados. Haz conclusiones y conjeturas.

alpha = 0.05 # nivel de significancia

# Separar los grupos
df_orders_A = df_orders[df_orders['group'] == 'A'] # separar el grupo A
# Usuarios unicos del grupo A
users_unique_A = df_orders_A['visitorId'].nunique() # contar el total de visitante unicos del grupo A
# Total de usuarios del grupo A
total_users_A = df_orders_A['visitorId'].count() # contar el total de usuarios del grupo A

df_orders_B = df_orders[df_orders['group'] == 'B'] # separar el grupo B
# Usuarios unicos del grupo B
users_unique_B = df_orders_B['visitorId'].nunique() # contar el total de visitante unicos del grupo B  
# Total de usuarios del grupo B
total_users_B = df_orders_B['visitorId'].count() # contar el total de usuarios del grupo B

# porporcion de exito del grupo A
conversion_A = users_unique_A / total_users_A # calcular la proporcion de exito del grupo A
# porporcion de exito del grupo B
conversion_B = users_unique_B / total_users_B # calcular la proporcion de exito del grupo B

# prueba de hipotesis 
count = [users_unique_A, users_unique_B] # contar el total de pedidos del grupo A y B
nobs = [total_users_A, total_users_B] # contar el total de usuarios del grupo A y B

# Realizar la prueba de hipotesis
z_stat, p_value = proportions_ztest(count, nobs) # realizar la prueba de hipotesis
# Imprimir los resultados
print('Z-statistic:', z_stat) # imprimir el valor z
print('P-value:', p_value) # imprimir el valor p

# Comparar el valor p con el nivel de significancia
if p_value < alpha: # si el valor p es menor que el nivel de significancia
    print('Rechazamos la hipotesis nula; Sí hay diferencia estadisticamente significativa') # rechazar la hipotesis nula
else: # si el valor p es mayor que el nivel de significancia
    print('No rechazamos la hipotesis nula; No hay diferencia estadisticamente significativa')
    



# diferencia estadísticamente significativa en el tamaño promedio de pedido

# Separar pedidos por grupo
group_A = df_orders[df_orders['group'] == 'A']['revenue'] # separar el grupo A
group_B = df_orders[df_orders['group'] == 'B']['revenue'] # separar el grupo B

from scipy.stats import mannwhitneyu

stat, pval = mannwhitneyu(group_A, group_B, alternative='two-sided')

print(f'Estadístico U: {stat:.2f}')
print(f'Valor p: {pval:.4f}')

# Comparar el valor p con el nivel de significancia
if pval < alpha: # si el valor p es menor que el nivel de significancia
    print('Rechazamos la hipotesis nula; Sí hay diferencia estadisticamente significativa') # rechazar la hipotesis nula
else: # si el valor p es mayor que el nivel de significancia    
    print('No rechazamos la hipotesis nula; No hay diferencia estadisticamente significativa')
    
    
    
    

# Datos crudos

# Separar los grupos
df_orders_A = df_orders_copy[df_orders_copy['group'] == 'A'] # separar el grupo A
# Usuarios unicos del grupo A
users_unique_A = df_orders_A['visitorId'].nunique() # contar el total de visitante unicos del grupo A
# Total de usuarios del grupo A
total_users_A = df_orders_A['visitorId'].count() # contar el total de usuarios del grupo A

df_orders_B = df_orders_copy[df_orders_copy['group'] == 'B'] # separar el grupo B
# Usuarios unicos del grupo B
users_unique_B = df_orders_B['visitorId'].nunique() # contar el total de visitante unicos del grupo B  
# Total de usuarios del grupo B
total_users_B = df_orders_B['visitorId'].count() # contar el total de usuarios del grupo B

# porporcion de exito del grupo A
conversion_A = users_unique_A / total_users_A # calcular la proporcion de exito del grupo A
# porporcion de exito del grupo B
conversion_B = users_unique_B / total_users_B # calcular la proporcion de exito del grupo B

# prueba de hipotesis 
count = [users_unique_A, users_unique_B] # contar el total de pedidos del grupo A y B
nobs = [total_users_A, total_users_B] # contar el total de usuarios del grupo A y B

# Realizar la prueba de hipotesis
z_stat, p_value = proportions_ztest(count, nobs) # realizar la prueba de hipotesis
# Imprimir los resultados
print('Z-statistic:', z_stat) # imprimir el valor z
print('P-value:', p_value) # imprimir el valor p

# Comparar el valor p con el nivel de significancia
if p_value < alpha: # si el valor p es menor que el nivel de significancia
    print('Rechazamos la hipotesis nula; Sí hay diferencia estadisticamente significativa') # rechazar la hipotesis nula
else: # si el valor p es mayor que el nivel de significancia
    print('No rechazamos la hipotesis nula; No hay diferencia estadisticamente significativa')
    



# diferencia estadísticamente significativa en el tamaño promedio de pedido

# Separar pedidos por grupo
group_A = df_orders_copy[df_orders_copy['group'] == 'A']['revenue'] # separar el grupo A
group_B = df_orders_copy[df_orders_copy['group'] == 'B']['revenue'] # separar el grupo B

from scipy.stats import mannwhitneyu

stat, pval = mannwhitneyu(group_A, group_B, alternative='two-sided')

print(f'Estadístico U: {stat:.2f}')
print(f'Valor p: {pval:.4f}')

# Comparar el valor p con el nivel de significancia
if pval < alpha: # si el valor p es menor que el nivel de significancia
    print('Rechazamos la hipotesis nula; Sí hay diferencia estadisticamente significativa') # rechazar la hipotesis nula
else: # si el valor p es mayor que el nivel de significancia    
    print('No rechazamos la hipotesis nula; No hay diferencia estadisticamente significativa')