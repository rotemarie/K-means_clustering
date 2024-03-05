# Spotify Inspired - Music Reccommender 

I tried to imitate the Spotify algorithm by taking songs a user likes as input and returning a list of similar songs recommended for him.
The app uses the Spotify API, and is written in Python.
The following libraries are used: Spotipy, Streamlit, Numpy, Pandas, scipy, sklearn.

## FrontEnd

The app screens are presented here:
![](https://github.com/rotemarie/Music-Recommender/blob/main/graphs/home.png)
![](https://github.com/rotemarie/Music-Recommender/blob/main/graphs/entersong.png)
![](https://github.com/rotemarie/Music-Recommender/blob/main/graphs/newsongs.png)


## Cleaning & Initial Analysis

- All data was numerical and there were no null values - so no cleaning was necessary.
- To study the data, initial plots were created.

```bash
genre_data = pd.read_csv('data/data_by_genres.csv')
print(genre_data.info()) #get info about colums and values
print(genre_data.isnull().sum()) #check for missing values
```
![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/popularity.png)
![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/trends.png)

## The Elbow Method & K-Means Clustering

- The elbow method was used to find the ideal number of clusters, but after trying various options, 5 clusters were selected.

```bash
# Elbow method to find the number of clusters
X = genre_data.select_dtypes(np.number) # only cluster columns with numerical values (so not the "genre" column)

wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

# trying k-means with k=5
scaler = StandardScaler()
X = genre_data.select_dtypes(np.number)
X_scaled = scaler.fit_transform(X)  # Use fit_transform instead of fit
kmeans = KMeans(n_clusters=5, random_state=42)
genre_data['cluster'] = kmeans.fit_predict(X_scaled)


# Plotting with TSNE
tsne = TSNE(n_components=2, verbose=1)
genre_embedding = tsne.fit_transform(X_scaled)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']
fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()
```

![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/elbowmethod.png)
![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/kmeans.png)

## Analysis

- To analyze the data, a parllel coordinates plot, a PCA plot, and a heat map were plotted.
- Based on the predictions shown in the plots, in each cluster features that showed correlation were plotted.

```bash
# plot centroids in each cluster
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
centroids['cluster'] = centroids.index

fig = px.parallel_coordinates(centroids, color="cluster",
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()

X = genre_data.select_dtypes(np.number).drop('cluster', axis=1)
y = genre_data['cluster']

X_std = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_std)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = y

# Visualize the results
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df)
plt.title('PCA of k-means Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Calculate the correlation matrix
correlation_matrix = genre_data.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Music Genres Variables')
plt.show()

#cluster 1 analysis
print(cluster_1_data)

#expect: low liveness, medium-high energy
fig1 = px.scatter(cluster_1_data, x='liveness', y='energy', color='cluster')
fig1.show()

#expect: low acousticness, medium-low danceability
fig2 = px.scatter(cluster_1_data, x='acousticness', y='danceability', color='cluster')
fig2.show()

#expec: medium temopo, medium loudness 
fig3 = px.scatter(cluster_1_data, x='tempo', y='loudness', color='cluster')
fig3.show()
```

![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/parallelcoord.png)
![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/pca.png){width=1px}
![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/heatmap.png)
![](https://github.com/rotemarie/K-means_clustering/blob/main/graphs/trends2.png)








