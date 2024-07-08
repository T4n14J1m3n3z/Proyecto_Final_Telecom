import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_eda(df):
    # Verificar si la carpeta 'plots' existe, si no, crearla
    if not os.path.exists('plots'):
        os.makedirs('plots')

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Histograma para monthly_charges
    ax = axs[0, 0]
    sns.histplot(df['monthly_charges'], kde=True, ax=ax, color='teal')
    ax.lines[0].set_color('red')
    ax.set_title('Distribución de Cargos Mensuales')
    ax.set_xlabel('Cargos Mensuales')
    ax.set_ylabel('Frecuencia')

    # Boxplot para monthly_charges
    ax = axs[0, 1]
    sns.boxplot(x=df['monthly_charges'], ax=ax, color='teal')
    ax.set_title('Boxplot de Cargos Mensuales')
    ax.set_xlabel('Cargos Mensuales')

    # Histograma para total_charges
    ax = axs[1, 0]
    sns.histplot(df['total_charges'], kde=True, ax=ax, color='steelblue')
    ax.lines[0].set_color('red')
    ax.set_title('Distribución de Cargos Totales')
    ax.set_xlabel('Cargos Totales')
    ax.set_ylabel('Frecuencia')

    # Boxplot para total_charges
    ax = axs[1, 1]
    sns.boxplot(x=df['total_charges'], ax=ax, color='steelblue')
    ax.set_title('Boxplot de Cargos Totales')
    ax.set_xlabel('Cargos Totales')

    fig.tight_layout()
    plt.savefig(r'plots\eda_plots.png')
    plt.close()

    # Gráfico de barras para churn
    palette = sns.color_palette('cividis', 2)
    plt.figure(figsize=(10, 5))
    sns.countplot(x='churn', data=df, hue='churn', palette=palette)
    plt.title('Distribución de abandono de servicio')
    plt.xlabel('Churn')
    plt.ylabel('Frecuencia')
    plt.legend(['con servicio', 'sin servicio'],
               title='Churn', loc='upper right')
    plt.savefig(r'plots\churn_distribution.png')
    plt.close()

    # Scatterplot para monthly_charges y churn
    plt.figure(figsize=(15, 8))
    sns.scatterplot(x=df['monthly_charges'],
                    y=df['total_charges'], hue=df['churn'])
    plt.title('Monthly Charges vs Total Charges por Churn')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    plt.legend(title='Churn')
    plt.savefig(r'plots\scatter_charges_churn.png')
    plt.close()
