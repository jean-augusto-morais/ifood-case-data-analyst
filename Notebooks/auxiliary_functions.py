import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
import os
from matplotlib.ticker import PercentFormatter
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"


def inspect_outliers(dataframe, column, whisker_width=1.5):
    """Função para inspecionar outliers.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    column : List[str]
        Lista com o nome das colunas (strings) a serem ultilizadas.
    whisker_width : float, opcional
        Valor considerado para detecção de outliers, por padrão 1.5

    returns
    -------
    pd.DataFrame
        Dataframe com outliers.
    """

    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    return dataframe[
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]

    
def pairplot(
    dataframe, columns, hue_column=None, alpha=0.5, corner=True, palette="tab10"
):
    """Função para gerar pairplot.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    hue_column : str, opcional
        Coluna utilizada para hue, por padrão None
    alpha : float, opcional
        Valor de alfa para transparência, por padrão 0.5
    corner : bool, opcional
        Se o pairplot terá apenas a diagonal inferior ou será completo, por padrão True
    palette : str, opcional
        Paleta a ser utilizada, por padrão "tab10"
    """
    analysis = columns.copy() + [hue_column]

    sns.pairplot(
        dataframe[analysis],
        diag_kind="kde",
        hue=hue_column,
        plot_kws=dict(alpha=alpha),
        corner=corner,
        palette=palette,
    )




os.environ["OMP_NUM_THREADS"] = "1"

def plot_elbow_silhouette(x, random_state=42, range_k=(2, 11)):
    """Gera os gráficos para os métodos Elbow e Silhouette.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe com os dados.
    random_state : int, opcional
        Valor para fixar o estado aleatorio para a reprodutibilidade, por padrão 42.
    range_k : tuple, opcional
        Intervalo de valores de cluster, por padrão (2, 11) 
    """
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)
    
    elbow = {}
    silhouette = []
    
    k_range = range(*range_k)
    
    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(x)
        elbow[i] = kmeans.inertia_
    
        labels = kmeans.labels_
        silhouette.append(silhouette_score(x, labels))
    
    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_xlabel("K")
    axs[0].set_xlabel("inertia")
    axs[0].set_title("Elbow Method")
    
    
    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].set_xlabel("K")
    axs[1].set_xlabel("Silhouette Score")
    axs[1].set_title("Silhouette Method")
    
    plt.show()
    


def plot_clusters_2D(
    dataframe,
    columns,
    n_colors,
    centroids,
    show_centroids=True,
    show_points=False,
    column_clusters=None,
):
    """Gerar gráficos 2D com os clusters.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    n_colors : int
        Númerode cores para o grafico
    centroids : np.ndarray
        Array com os centroides
    show_centroids : bool, opcional
        Se o gráfico irá mostrar centroids ou não, por padrão True
    show_points : bool, opcional
        Se o gráfico irá mostrar pontos ou não, por padrão False"
    column_clusters : List[int], opcional
        Coluna com os números dos clusters para coloriros pontos
        (caso mostrar_pontos seja True), por padrão None
    """
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    
    cores = plt.cm.tab10.colors[:n_colors]
    cores = ListedColormap(cores)
    

    x = dataframe[columns[0]]
    y = dataframe[columns[1]]
    
    ligar_centroids = show_centroids
    ligar_pontos = show_points
    
    for i, centroid in enumerate(centroids):
        if ligar_centroids:
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(
                *centroid,
                f"{i}",
                fontsize=20,
                horizontalalignment="center",
                verticalalignment="center"
            )
        if ligar_pontos:
            s = ax.scatter(x, y, c=column_clusters, cmap=cores)
            ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))
    
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_title("Clusters")
    
    plt.show()

def plot_columns_percent_by_clusters(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="cluster"
):
    """Função para gerar gráficos de barras com a porcentagem de cada valor por cluster.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    rows_cols : tuple, opcional
        Tuplas co o numeros de linhas e colunas do grid de eixos, por padrão (2, 3)
    figsize : tuple, opcional
        Tupla com a largura e altura da figura , por padrão (15, 8)
    column_cluster : str, opcional
        Nome da coluna  com os numeros dos clusters, por padrão "cluster"
    """
    
    fig, axs = plt.subplots(nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True)

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(x=column_cluster, hue=col, data=dataframe, ax=ax, multiple="fill", stat="percent", discrete=True, shrink=0.8)
    
        n_clusters = dataframe[column_cluster].nunique()
        h.set_xticks(range(n_clusters))
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_label("")
        h.tick_params(axis="both", which="both", length=0)
    
        for bars in h.containers:
            h.bar_label(bars, label_type="center", labels=[f'{b.get_height():.1%}' for b in bars], color="white", weight="bold", fontsize=11)
    
        for bar in h.patches:
            bar.set_linewidth(0)
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    plt.show()


def plot_columns_percent_hue_clusters(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="cluster",
    palette="tab10"
):
    """Função para gerar gráficos de barras com a porcentagem de cada valor por cluster.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    rows_cols : tuple, opcional
        Tuplas co o numeros de linhas e colunas do grid de eixos, por padrão (2, 3)
    figsize : tuple, opcional
        Tupla com a largura e altura da figura , por padrão (15, 8)
    column_cluster : str, opcional
        Nome da coluna  com os numeros dos clusters, por padrão "cluster"
    palette : str, opcional
        Paleta a ser utilizada, por padrao "tab10"
    """
    
    fig, axs = plt.subplots(nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True)

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(x=col, hue=column_cluster, data=dataframe, ax=ax, multiple="fill", stat="percent", discrete=True, shrink=0.8, palette=palette)
    
        if dataframe[col].dtype != "object":
            h.set_xticks(range(dataframe[col].nunique()))
        
       
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_label("")
        h.tick_params(axis="both", which="both", length=0)
    
        for bars in h.containers:
            h.bar_label(bars, label_type="center", labels=[f'{b.get_height():.1%}' for b in bars], color="white", weight="bold", fontsize=11)
    
        for bar in h.patches:
            bar.set_linewidth(0)

        legend = h.get_legend()
        legend.remove()

    labels = [text.get_text() for text in legend.get_texts()]

    fig.legend(handles=legend.legend_handles, labels=labels, loc="upper center", ncols=dataframe[column_cluster].nunique(), title="clusters")

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    plt.show()