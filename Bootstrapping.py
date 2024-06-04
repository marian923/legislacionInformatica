import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

from sklearn.utils import resample

pd.options.display.max_rows = 100
df = pd.read_csv("datasetCampus.csv", delimiter=",")

pd.options.display.max_columns = 25
pd.options.display.max_rows = 120

df['categoria'] = df['categoria'].apply(lambda x: str(x))

for columna in df.columns:
    conteo_valores = df[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteo/{columna}_conteo.csv")

grupos = df.groupby("rol")

# Obtener los DataFrames resultantes
df_student = grupos.get_group("student")
df_editingteacher = grupos.get_group("editingteacher")

# Obtener el número de filas en el DataFrame "df_student"
num_filas_student = df_student.shape[0]

# Obtener el número de filas en el DataFrame "df_editingteacher"
num_filas_editingteacher = df_editingteacher.shape[0]

print(f"Número de filas en df_student: {num_filas_student}")
print(f"Número de filas en df_editingteacher: {num_filas_editingteacher}")

porcentajeStudent = (num_filas_student / len(df))
porcentajeTeacher = (num_filas_editingteacher / len(df))

print(f"Representación de porcentaje para df_student: {porcentajeStudent* 100}%")
print(f"Representación de porcentaje para df_editingteacher: {porcentajeTeacher* 100}%")

for columna in df_student.columns:
    conteo_valores = df_student[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoStudent/{columna}_conteo.csv")

for columna in df_editingteacher.columns:
    conteo_valores = df_editingteacher[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoTeacher/{columna}_conteo.csv")

# Función para generar datos sintéticos utilizando bootstrapping
def generadorBootstrap(df, columnas, num_muestras_sinteticas, num_muestras_bootstrap, tamanho_muestra_bootstrap, tipo):
    # Cargar los conteos de valores únicos de los archivos CSV
    conteos_columnas = {}
    if tipo == "student":
        for columna in columnas:
            conteo_df = pd.read_csv(f"../Docs/DocsConteoStudent/{columna}_conteo.csv")
            conteos_columnas[columna] = dict(zip(conteo_df[columna], conteo_df['conteo']))
    if tipo == "editingteacher":
        for columna in columnas:
            conteo_df = pd.read_csv(f"../Docs/DocsConteoTeacher/{columna}_conteo.csv")
            conteos_columnas[columna] = dict(zip(conteo_df[columna], conteo_df['conteo']))
    if tipo == "c":
        for columna in columnas:
            conteo_df = pd.read_csv(f"../Docs/DocsConteoCreate/{columna}_conteo.csv")
            conteos_columnas[columna] = dict(zip(conteo_df[columna], conteo_df['conteo']))
    if tipo == "r":
        for columna in columnas:
            conteo_df = pd.read_csv(f"../Docs/DocsConteoRead/{columna}_conteo.csv")
            conteos_columnas[columna] = dict(zip(conteo_df[columna], conteo_df['conteo']))
    if tipo == "u":
        for columna in columnas:
            conteo_df = pd.read_csv(f"../Docs/DocsConteoUpdate/{columna}_conteo.csv")
            conteos_columnas[columna] = dict(zip(conteo_df[columna], conteo_df['conteo']))
    if tipo == "d":
        for columna in columnas:
            conteo_df = pd.read_csv(f"../Docs/DocsConteoDelete/{columna}_conteo.csv")
            conteos_columnas[columna] = dict(zip(conteo_df[columna], conteo_df['conteo']))

    # Crear un DataFrame vacío para almacenar los datos sintéticos
    datos_sinteticos = pd.DataFrame(columns=columnas)

    for _ in range(num_muestras_bootstrap):
        # Tomar una muestra con reemplazo del conjunto de datos original
        muestra_bootstrap = resample(df[columnas], replace=True, n_samples=tamanho_muestra_bootstrap, random_state=42)

        # Generar datos sintéticos a partir de la muestra de bootstrap
        datos_sinteticos_muestra = pd.DataFrame(index=range(num_muestras_sinteticas), columns=columnas)
        for columna in columnas:
            valores_unicos = muestra_bootstrap[columna].unique()
            conteo_valores = conteos_columnas[columna]

            # Calcular las probabilidades de los valores únicos
            conteos_array = np.array([conteo_valores.get(valor, 0) for valor in valores_unicos])
            probabilidades = conteos_array / sum(conteos_array)

            # Generar datos sintéticos para la columna basados en las frecuencias observadas
            datos_sinteticos_muestra[columna] = np.random.choice(valores_unicos, size=num_muestras_sinteticas, p=probabilidades)

        # Agregar los datos sintéticos de la muestra al DataFrame principal
        datos_sinteticos = pd.concat([datos_sinteticos, datos_sinteticos_muestra], ignore_index=True)

    return datos_sinteticos

columnas_interes = ["crud", "accion", "fechaunix", "categoria", "curso", "rol" ]
desviacion_aleatoria = random.uniform(-0.0005, 0.0005)
num_muestras_student = 100000 * porcentajeStudent  # Número de muestras sintéticas a generar
num_muestras_teacher = 100000 * porcentajeTeacher # Número de muestras sintéticas a generar
num_muestras_bootstrap = 10  # Número de muestras de bootstrap
tamanho_muestra_bootstrap = len(df)  # Tamaño de cada muestra de bootstrap
num_muestras_student += num_muestras_student * desviacion_aleatoria
num_muestras_teacher += num_muestras_teacher * desviacion_aleatoria

datos_student = generadorBootstrap(df_student, columnas_interes, int(num_muestras_student), num_muestras_bootstrap, tamanho_muestra_bootstrap, "student")
datos_teacher = generadorBootstrap(df_editingteacher, columnas_interes, int(num_muestras_teacher), num_muestras_bootstrap, tamanho_muestra_bootstrap, "editingteacher")
#print(datos_sinteticos.head())

datos_sinteticos = pd.concat([datos_student, datos_teacher])

filtered_df = datos_sinteticos[(datos_sinteticos['crud'] == 'u') & (datos_sinteticos['rol'] == 'editingteacher')]

# Contamos el número de filas en el dataframe filtrado
count = filtered_df.shape[0]

print("El total de valores es:", count)

filtered_df = df[(df['crud'] == 'u') & (df['rol'] == 'editingteacher')]

# Contamos el número de filas en el dataframe filtrado
count = filtered_df.shape[0]

print("El total de valores es:", count)

datos_sinteticos.to_csv("DatosSinteticos.csv")

# Conteo de valores únicos y su frecuencia para cada columna
for var in datos_sinteticos.columns:
    conteo_valores = datos_sinteticos[var].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[var, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoNuevoSynt/{var}_conteo.csv")

for crud in df['rol'].unique():
    df_analityc = df[df['rol'] == crud].copy()

    df_analityc['crud'] = pd.Categorical(df_analityc['crud'], categories=['c', 'r', 'u', 'd'], ordered=True)
    crudA_counts = df_analityc.groupby(['rol', 'crud'], observed=True).size().reset_index(name='counts')

    totalA_counts = crudA_counts['counts'].sum()
    crudA_counts['percentage'] = (crudA_counts['counts'] / totalA_counts) * 100

    plt.figure(figsize=(10, 6), dpi = 300)
    bar_plot = sns.barplot(data=crudA_counts, x='crud', y='counts')   # Cada valor descrito entre '' corresponde al nombre de la columna del dataframe

    for bar in bar_plot.patches:
        if bar.get_height() > 0:
            height = bar.get_height()
            bar_plot.annotate(f'{int(height)}\n({height / totalA_counts * 100:.1f}%)',
                (bar.get_x() + bar.get_width() / 2., height), ha='center', va='bottom',
                fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    palette = ['#66b3ff','#99ff99','#ffcc99', '#ff9999']
    bar_plot = sns.barplot(data=crudA_counts, x='crud', y='counts', palette=palette, hue = 'crud', legend=False)

    plt.title(f'Acciones CRUD - {crud}')
    plt.xlabel('Acciones CRUD')
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.ylim(0, crudA_counts['counts'].max() * 1.1)
    plt.show()

for crud in datos_sinteticos['rol'].unique():
    df_analityc = datos_sinteticos[datos_sinteticos['rol'] == crud].copy()

    df_analityc['crud'] = pd.Categorical(df_analityc['crud'], categories=['c', 'r', 'u', 'd'], ordered=True)
    crudA_counts = df_analityc.groupby(['rol', 'crud'], observed=True).size().reset_index(name='counts')

    totalA_counts = crudA_counts['counts'].sum()
    crudA_counts['percentage'] = (crudA_counts['counts'] / totalA_counts) * 100

    plt.figure(figsize=(10, 6), dpi = 300)
    bar_plot = sns.barplot(data=crudA_counts, x='crud', y='counts')   # Cada valor descrito entre '' corresponde al nombre de la columna del dataframe

    for bar in bar_plot.patches:
        if bar.get_height() > 0:
            height = bar.get_height()
            bar_plot.annotate(f'{int(height)}\n({height / totalA_counts * 100:.1f}%)',
                (bar.get_x() + bar.get_width() / 2., height), ha='center', va='bottom',
                fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    palette = ['#66b3ff','#99ff99','#ffcc99', '#ff9999']
    bar_plot = sns.barplot(data=crudA_counts, x='crud', y='counts', palette=palette, hue = 'crud', legend=False)

    plt.title(f'Acciones CRUD - {crud}')
    plt.xlabel('Acciones CRUD')
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.ylim(0, crudA_counts['counts'].max() * 1.1)
    plt.show()

grupos = df.groupby("crud")

df_create = grupos.get_group("c")
df_read = grupos.get_group("r")
df_update = grupos.get_group("u")
df_delete = grupos.get_group("d")

num_create = df_create.shape[0]
num_read = df_read.shape[0]
num_update = df_update.shape[0]
num_delete = df_delete.shape[0]

print(f"Número de filas en df_create: {num_create}")
print(f"Número de filas en df_read: {num_read}")
print(f"Número de filas en df_update: {num_update}")
print(f"Número de filas en df_delete: {num_delete}")

total_filas = len(df)

porcentaje_create = (num_create / total_filas)
porcentaje_read = (num_read / total_filas)
porcentaje_update = (num_update / total_filas)
porcentaje_delete = (num_delete / total_filas)

print(f"Porcentaje de filas para CRUD 'c': {porcentaje_create*100}%")
print(f"Porcentaje de filas para CRUD 'r': {porcentaje_read*100}%")
print(f"Porcentaje de filas para CRUD 'u': {porcentaje_update*100}%")
print(f"Porcentaje de filas para CRUD 'd': {porcentaje_delete*100}%")

crud = ["Create","Read","Update","Delete"]
for i in crud:
    ruta_carpeta_rol = f"../Docs/DocsConteo{i}"
    os.makedirs(ruta_carpeta_rol, exist_ok=True)

for columna in df_create.columns:
    conteo_valores = df_create[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoCreate/{columna}_conteo.csv")

for columna in df_read.columns:
    conteo_valores = df_read[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoRead/{columna}_conteo.csv")

for columna in df_update.columns:
    conteo_valores = df_update[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoUpdate/{columna}_conteo.csv")

for columna in df_delete.columns:
    conteo_valores = df_delete[columna].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[columna, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoDelete/{columna}_conteo.csv")

columnas_interes = ["crud", "accion", "fechaunix", "categoria", "curso", "rol" ]
desviacion_aleatoria = random.uniform(-0.0005, 0.0005)
num_muestras_create = 100000 * porcentaje_create
num_muestras_read = 100000 * porcentaje_read
num_muestras_update = 100000 * porcentaje_update
num_muestras_delete = 100000 * porcentaje_delete

num_muestras_bootstrap = 10  # Número de muestras de bootstrap
tamanho_muestra_bootstrap = len(df)  # Tamaño de cada muestra de bootstrap
num_muestras_create += num_muestras_create * desviacion_aleatoria
num_muestras_read += num_muestras_read * desviacion_aleatoria
num_muestras_update += num_muestras_update * desviacion_aleatoria
num_muestras_delete += num_muestras_delete * desviacion_aleatoria


datos_create = generadorBootstrap(df_create, columnas_interes, int(num_muestras_create), num_muestras_bootstrap, tamanho_muestra_bootstrap, "c")
datos_read = generadorBootstrap(df_read, columnas_interes, int(num_muestras_read), num_muestras_bootstrap, tamanho_muestra_bootstrap, "r")
datos_update = generadorBootstrap(df_update, columnas_interes, int(num_muestras_update), num_muestras_bootstrap, tamanho_muestra_bootstrap, "u")
datos_delete = generadorBootstrap(df_delete, columnas_interes, int(num_muestras_delete), num_muestras_bootstrap, tamanho_muestra_bootstrap, "d")

#print(datos_sinteticos.head())

datos_sinteticos_crud = pd.concat([datos_create,datos_read,datos_update,datos_delete])

filtered_df = datos_sinteticos_crud[(datos_sinteticos_crud['crud'] == 'r') & (datos_sinteticos_crud['accion'] == 'viewed')]

# Contamos el número de filas en el dataframe filtrado
count = filtered_df.shape[0]

print("El total de valores es:", count)

datos_sinteticos_crud.to_csv("DatosSinteticosCrud.csv")

# Conteo de valores únicos y su frecuencia para cada columna
for var in datos_sinteticos_crud.columns:
    conteo_valores = datos_sinteticos_crud[var].value_counts().to_dict()
    df_conteo = pd.DataFrame(conteo_valores.items(), columns=[var, 'conteo'])
    df_conteo.to_csv(f"../Docs/DocsConteoNuevoSyntCrud/{var}_conteo.csv")

filtered_df = datos_sinteticos[(datos_sinteticos['curso'] == 'DISEÑO SISMICO_GRUPO-A_GRUPO-B') & (datos_sinteticos['rol'] == 'student')]

# Contamos el número de filas en el dataframe filtrado
count = filtered_df.shape[0]

print("El total de valores es:", count)