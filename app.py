import pandas

import musicrec as mr
import streamlit as st
import pandas as pd

data = pd.read_csv('data/data.csv')

# -----------------------------------------------------------------------------------
# web app code


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


