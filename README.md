# Spotify Inspired - Music Recommender 

I tried to imitate the Spotify algorithm by taking songs a user likes as input and returning a list of similar songs recommended for him.
The app uses the Spotify API, and is written in Python.
The following libraries are used: Spotipy, Streamlit, Numpy, Pandas, scipy, sklearn.

## Front End

The app screens are presented here:
![](https://github.com/rotemarie/Music-Recommender/blob/main/graphs/home.png)
![](https://github.com/rotemarie/Music-Recommender/blob/main/graphs/entersong.png)
![](https://github.com/rotemarie/Music-Recommender/blob/main/graphs/newsongs.png)

Streamlit code:
![file](https://github.com/rotemarie/Music-Recommender/blob/main/app.py)
```bash
data = pd.read_csv('data/data.csv')

st.set_page_config(page_title='Music Recommender', page_icon = "spotify-icon-logos/logos/02_CMYK/02_PNG/Spotify_Logo_RGB_Black.png")
st.image("spotify-icons-logos/logos/01_RGB/02_PNG/Spotify_Logo_RGB_Green.png", width=200)


st.title("Music Recommender")
st.subheader("Welcome to the Music Recommender!")
st.caption("You are a click away from your new favorite music")

song_name = st.text_input("Enter the song name:")
release_year = st.text_input("Enter the release year:")

# Define an empty dictionary to store the liked songs
if 'liked_songs' not in st.session_state:
    st.session_state['liked_songs'] = []

# Submit button
if st.button("Submit"):
    if song_name and release_year:
        submitted_info = {'name': song_name, 'year': int(release_year)}
        st.session_state.liked_songs.append(submitted_info)
        #st.write(st.session_state.liked_songs)
        st.success(f"Song name: {song_name}, Release year: {release_year}")
    else:
        st.error("Please enter both the song name and release year.")

st.caption("Click here when you are done entering songs:")
if st.button("Find me new music"):
    new_songs = mr.recommend_songs(st.session_state.liked_songs, data)
    pp = pd.DataFrame(columns=['Song Name', 'Artists'])
    for song in new_songs:
        pp = pp.append({'Song Name': song['name'], 'Artists': song['artists'][1:-1]}, ignore_index=True)
    st.data_editor(
        pp,
        column_config={
            "widgets": st.column_config.TextColumn(
                "Widgets",
                help="Streamlit **widget** commands 🎈",
                default="st.",
                max_chars=50,
                validate="^st\.[a-z_]+$",
            )
        },
        hide_index=True,
    )

```

Formatting:
```bash
[theme]
base = "dark"
primaryColor = "#1DB954"
backgroundColor = "#191414"
textColor = "#FFFFFF"
font = "sans serif"
```

## Back End 
![zipped file](https://github.com/rotemarie/Music-Recommender/blob/main/musicrec.zip)
Replace the Spotify client ID and secret key with your credentials:
```bash
import spotipy
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SPOTIPY_CLIENT_ID = ''
SPOTIPY_CLIENT_SECRET = ''
SPOTIPY_REDIRECT_URI = 'http://localhost:8501/'

liked_songs = []
```

Data reading and clustering:
```bash
# read data
genre_data = pd.read_csv('data/data_by_genres.csv')
year_data = pd.read_csv('data/data_by_year.csv')
data = pd.read_csv('data/data.csv')
artist_data = pd.read_csv('data/data_by_artist.csv')

datasets = [("song_data", data), ("genre_data", genre_data), ("year_data", year_data), ("artist_data", artist_data)]

# clustering
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels
```

Building the recommender
```bash

# list for numerical columns for comparison
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))


# if a song doesn't show up in the data we loaded, this method will try to find it using the Spotipy API
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if not results['tracks']['items']:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


# search for a song in the existing dataset
def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])


# get mean and flatten songs so that they are comparable to the data in the clusters
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


# using the clusters to find songs to recommend
def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')
```


## DATA ANALYSIS - LEARNING ABOUT THE DATA
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








