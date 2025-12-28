"""Streamlit UI for medley recommender."""

import requests
import streamlit as st
from api.models import AddSongRequest, SearchRequest

from src.utils.config import settings

# Configuration - use API_URL from .env (defaults to localhost for local dev)
# In Docker, set API_URL=http://api:9876 to use service name
API_BASE_URL = f"{settings.api_url}/api"


def search_songs(query: str, bpm_min: float | None, bpm_max: float | None, keys: list[str] | None, limit: int) -> dict:
    """Search for songs via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "bpm_min": bpm_min,
                "bpm_max": bpm_max,
                "keys": keys if keys else None,
                "limit": limit,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching songs: {str(e)}")
        return {"results": [], "total": 0}


def add_song(title: str, youtube_url: str, lyrics: str) -> dict:
    """Add a new song via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/add_song",
            json={
                "title": title,
                "youtube_url": youtube_url,
                "lyrics": lyrics,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error adding song: {str(e)}")
        return {"success": False, "message": str(e)}


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Medley Recommender",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 Medley Recommender")
    st.markdown("Search for worship/praise songs by lyrics similarity and musical metadata")

    # Sidebar for filters
    with st.sidebar:
        st.header("Filters")
        bpm_min = st.slider("Min BPM", 0, 200, 0)
        bpm_max = st.slider("Max BPM", 0, 200, 200)

        # Key filter - multi-select dropdown
        available_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        selected_keys = st.multiselect("Keys", available_keys, default=[])

        limit = st.slider("Results Limit", 5, 50, 10)

    # Main search interface
    tab1, tab2 = st.tabs(["🔍 Search", "➕ Add Song"])

    with tab1:
        st.header("Search Songs")
        query = st.text_input("Enter search query (lyrics or description)", "")

        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    results = search_songs(
                        query,
                        bpm_min if bpm_min > 0 else None,
                        bpm_max if bpm_max < 200 else None,
                        selected_keys if selected_keys else None,
                        limit
                    )

                if results["total"] > 0:
                    st.success(f"Found {results['total']} result(s)")

                    # Display results
                    for i, song in enumerate(results["results"], 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.subheader(f"{i}. {song['title']}")
                                if song.get("bpm"):
                                    st.write(f"**BPM:** {song['bpm']:.1f}")
                                if song.get("key"):
                                    st.write(f"**Key:** {song['key']}")
                                st.write(f"**Similarity:** {song['similarity_score']:.3f}")

                            with col2:
                                st.link_button("🎬 YouTube", song["youtube_url"])

                            st.divider()
                else:
                    st.info("No results found. Try a different query.")
            else:
                st.warning("Please enter a search query")

    with tab2:
        st.header("Add New Song")
        with st.form("add_song_form"):
            title = st.text_input("Song Title *")
            youtube_url = st.text_input("YouTube URL *")
            lyrics = st.text_area("Lyrics *", height=200)

            submitted = st.form_submit_button("Add Song", type="primary")

            if submitted:
                if title and youtube_url and lyrics:
                    with st.spinner("Adding song..."):
                        result = add_song(title, youtube_url, lyrics)

                    if result.get("success"):
                        st.success(f"✅ Song added successfully!")
                        st.info(f"Song ID: {result.get('song_id')}")
                        st.info(f"Note: {result.get('message')}")
                    else:
                        st.error(f"❌ Failed to add song: {result.get('message')}")
                else:
                    st.warning("Please fill in all required fields (*)")


if __name__ == "__main__":
    main()


