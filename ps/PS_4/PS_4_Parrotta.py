# %%
# -- Imports --
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score

# %%
# -- Reading data --
fondos_comunes = pd.read_csv('..\\..\\data\\fondos_comunes.csv')
features = ['VOLAT', 'r3m', 'r6m']

# %%
# -- Preprocessing --
standar = StandardScaler()
train_df = standar.fit_transform(fondos_comunes[features])

# %%
# -- Training --

models = ['KMeans', 'AgglomerativeClustering']

for model in models:
    scores = []
    K = range(2, 10)
    for k in K:
        if model == 'KMeans':
            cluster_model = KMeans(n_clusters=k)
        elif model == 'AgglomerativeClustering':
            cluster_model = AgglomerativeClustering(n_clusters=k)
        
        labels = cluster_model.fit_predict(train_df)
        scores.append(calinski_harabasz_score(train_df, labels))

    plt.plot(K, scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title(f'Calinski-Harabasz Score Method For Optimal k - {model}')
    plt.show()

# %%
# -- Analisis --
# Elijo k = 5

final_models = [('KMeans', KMeans(n_clusters=5)), 
                ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=5))]

for model in final_models:
    labels = model[1].fit_predict(train_df)
    fondos_comunes[model[0]] = labels

# %%
sns.scatterplot(x="VOLAT", y="r6m", hue="TIPO", data=fondos_comunes, palette="deep")
plt.show()
sns.scatterplot(x="VOLAT", y="r6m", hue="KMeans", data=fondos_comunes, palette="deep")
plt.show()
sns.scatterplot(x="VOLAT", y="r6m", hue="AgglomerativeClustering", data=fondos_comunes, palette="deep")
plt.show()
# %%
sns.scatterplot(x="r3m", y="r6m", hue="TIPO", data=fondos_comunes, palette="deep")
plt.show()
sns.scatterplot(x="r3m", y="r6m", hue="KMeans", data=fondos_comunes, palette="deep")
plt.show()
sns.scatterplot(x="r3m", y="r6m", hue="AgglomerativeClustering", data=fondos_comunes, palette="deep")
plt.show()
# %%
sns.scatterplot(x="VOLAT", y="r3m", hue="TIPO", data=fondos_comunes, palette="deep")
plt.show()
sns.scatterplot(x="VOLAT", y="r3m", hue="KMeans", data=fondos_comunes, palette="deep")
plt.show()
sns.scatterplot(x="VOLAT", y="r3m", hue="AgglomerativeClustering", data=fondos_comunes, palette="deep")
plt.show()
# %%
print(fondos_comunes[["TIPO", "KMeans", "AgglomerativeClustering"]])

# Descartaría los puntos correspondientes al cluster 3 de KMeans.
