import matplotlib.pyplot as plt
import seaborn as sns


def visualize_data(df_intercon):
    # Configurar la figura y los ejes para los subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Histograma para monthly_charges
    ax = axs[0, 0]
    sns.histplot(df_intercon['monthly_charges'], kde=True, ax=ax, color='teal')
    ax.lines[0].set_color('red')
    ax.set_title('Distribución de Cargos Mensuales')
    ax.set_xlabel('Cargos Mensuales')
    ax.set_ylabel('Frecuencia')

    # Boxplot para monthly_charges
    ax = axs[0, 1]
    sns.boxplot(x=df_intercon['monthly_charges'], ax=ax, color='teal')
    ax.set_title('Boxplot de Cargos Mensuales')
    ax.set_xlabel('Cargos Mensuales')

    # Histograma para total_charges
    ax = axs[1, 0]
    sns.histplot(df_intercon['total_charges'],
                 kde=True, ax=ax, color='steelblue')
    ax.lines[0].set_color('red')
    ax.set_title('Distribución de Cargos Totales')
    ax.set_xlabel('Cargos Totales')
    ax.set_ylabel('Frecuencia')

    # Boxplot para total_charges
    ax = axs[1, 1]
    sns.boxplot(x=df_intercon['total_charges'], ax=ax, color='steelblue')
    ax.set_title('Boxplot de Cargos Totales')
    ax.set_xlabel('Cargos Totales')

    fig.tight_layout()
    plt.show()

    # Gráfico de barras para churn
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette('cividis', 2)
    sns.countplot(x='churn', data=df_intercon, hue='churn', palette=palette)
    plt.title('Distribución de abandono de servicio')
    plt.xlabel('Churn')
    plt.ylabel('Frecuencia')
    plt.legend(['con servicio', 'sin servicio'],
               title='Churn', loc='upper right')
    plt.show()

    # Scatterplot para monthly_charges y churn
    plt.figure(figsize=(15, 8))
    sns.scatterplot(x=df_intercon['monthly_charges'],
                    y=df_intercon['total_charges'], hue=df_intercon['churn'])
    plt.title('Monthly Charges vs Total Charges por Churn')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    plt.legend(title='Churn')
    plt.show()
