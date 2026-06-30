use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde_json::json;

use crate::auth::require_mcp_authenticated;
use crate::state::AppState;

#[derive(Clone)]
pub struct MedleyMcp {
    state: AppState,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl MedleyMcp {
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }
}

impl std::fmt::Debug for MedleyMcp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MedleyMcp").finish_non_exhaustive()
    }
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchSongsArgs {
    pub query: String,
    pub keys: Option<Vec<String>>,
    pub bpm_min: Option<f64>,
    pub bpm_max: Option<f64>,
    pub limit: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AddSongArgs {
    pub title: String,
    pub youtube_url: String,
    pub lyrics: String,
    pub bpm: f64,
    pub key: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SongIdArgs {
    pub song_id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UpdateSongArgs {
    pub song_id: String,
    pub title: Option<String>,
    pub youtube_url: Option<String>,
    pub lyrics: Option<String>,
    pub bpm: Option<f64>,
    pub key: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListSongsArgs {
    pub q: Option<String>,
    pub key: Option<String>,
    pub bpm_min: Option<f64>,
    pub bpm_max: Option<f64>,
    pub limit: Option<u32>,
    pub last_id: Option<String>,
    pub last_rank: Option<f64>,
}

#[tool_router]
impl MedleyMcp {
    #[tool(description = "Semantic search for songs by lyrics similarity")]
    async fn search_songs(
        &self,
        Parameters(args): Parameters<SearchSongsArgs>,
    ) -> Result<String, String> {
        tracing::info!(query = %args.query, "mcp search_songs");
        let query = medley_core::domain::models::SearchQuery {
            query: args.query,
            bpm_min: args.bpm_min,
            bpm_max: args.bpm_max,
            keys: args.keys,
            limit: args.limit,
        };
        let results = self.state.search.search(query).await.map_err(|e| e.to_string())?;
        tracing::info!(count = results.len(), "mcp search_songs ok");
        serde_json::to_string(&json!({
            "results": results,
            "total": results.len(),
        }))
        .map_err(|e| e.to_string())
    }

    #[tool(description = "Add a new song to the catalog")]
    async fn add_song(
        &self,
        Parameters(args): Parameters<AddSongArgs>,
    ) -> Result<String, String> {
        require_mcp_authenticated()?;
        tracing::info!(title = %args.title, "mcp add_song");
        let new_song = medley_core::domain::models::NewSong {
            title: args.title,
            youtube_url: args.youtube_url,
            lyrics: args.lyrics,
            bpm: args.bpm,
            key: args.key,
        };
        let song = self.state.songs.create(new_song).await.map_err(|e| e.to_string())?;
        tracing::info!(song_id = %song.song_id, "mcp add_song ok");
        serde_json::to_string(&json!({
            "success": true,
            "song_id": song.song_id,
            "message": "Song added and indexed successfully",
        }))
        .map_err(|e| e.to_string())
    }

    #[tool(description = "Get a song by ID")]
    async fn get_song(
        &self,
        Parameters(args): Parameters<SongIdArgs>,
    ) -> Result<String, String> {
        tracing::info!(song_id = %args.song_id, "mcp get_song");
        let song = self
            .state
            .songs
            .get(&args.song_id)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&song).map_err(|e| e.to_string())
    }

    #[tool(description = "Update a song")]
    async fn update_song(
        &self,
        Parameters(args): Parameters<UpdateSongArgs>,
    ) -> Result<String, String> {
        require_mcp_authenticated()?;
        tracing::info!(song_id = %args.song_id, "mcp update_song");
        let patch = medley_core::domain::models::SongPatch {
            title: args.title,
            youtube_url: args.youtube_url,
            lyrics: args.lyrics,
            bpm: args.bpm,
            key: args.key,
        };
        let song = self
            .state
            .songs
            .update(&args.song_id, patch)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&json!({ "success": true, "song": song })).map_err(|e| e.to_string())
    }

    #[tool(description = "Delete a song")]
    async fn delete_song(
        &self,
        Parameters(args): Parameters<SongIdArgs>,
    ) -> Result<String, String> {
        require_mcp_authenticated()?;
        tracing::info!(song_id = %args.song_id, "mcp delete_song");
        self.state
            .songs
            .delete(&args.song_id)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string(&json!({ "success": true })).map_err(|e| e.to_string())
    }

    #[tool(description = "List or keyword-search songs in the catalog")]
    async fn list_songs(
        &self,
        Parameters(args): Parameters<ListSongsArgs>,
    ) -> Result<String, String> {
        tracing::info!(?args, "mcp list_songs");
        let query = medley_core::domain::models::SongListQuery {
            q: args.q,
            key: args.key,
            bpm_min: args.bpm_min,
            bpm_max: args.bpm_max,
            limit: args.limit,
            last_id: args.last_id,
            last_rank: args.last_rank,
        };
        let page = self.state.songs.list(query).await.map_err(|e| e.to_string())?;
        serde_json::to_string(&page).map_err(|e| e.to_string())
    }
}

#[tool_handler]
impl ServerHandler for MedleyMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Medley worship song catalog with semantic search")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::with_mcp_authenticated;
    use crate::app::{build_state, test_config};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn test_state(voyage_base_url: &str) -> (tempfile::TempDir, tempfile::TempDir, AppState) {
        let db_dir = tempfile::TempDir::new().unwrap();
        let edge_dir = tempfile::TempDir::new().unwrap();
        let config = test_config(
            db_dir.path().join("medley.db"),
            edge_dir.path().join("edge_shard"),
            voyage_base_url,
        );
        let state = build_state(&config).await.unwrap();
        (db_dir, edge_dir, state)
    }

    async fn mount_voyage_embeddings(server: &MockServer) {
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }]
            })))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn get_song_not_found() {
        let (_db, _edge, state) = test_state("http://127.0.0.1:9").await;
        let mcp = MedleyMcp::new(state);
        let err = mcp
            .get_song(Parameters(SongIdArgs {
                song_id: "00000000-0000-7000-8000-000000000000".into(),
            }))
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn add_song_requires_mcp_auth() {
        let (_db, _edge, state) = test_state("http://127.0.0.1:9").await;
        let mcp = MedleyMcp::new(state);
        let err = mcp
            .add_song(Parameters(AddSongArgs {
                title: "Blocked".into(),
                youtube_url: "https://www.youtube.com/watch?v=blocked1234".into(),
                lyrics: "nope".into(),
                bpm: 100.0,
                key: "C".into(),
            }))
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn add_and_get_song() {
        let server = MockServer::start().await;
        mount_voyage_embeddings(&server).await;

        let (_db, _edge, state) = test_state(&server.uri()).await;
        let mcp = MedleyMcp::new(state);

        let created = with_mcp_authenticated(
            true,
            mcp.add_song(Parameters(AddSongArgs {
                title: "MCP Song".into(),
                youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".into(),
                lyrics: "mcp lyrics".into(),
                bpm: 100.0,
                key: "C".into(),
            })),
        )
        .await
        .unwrap();
        let created_json: serde_json::Value = serde_json::from_str(&created).unwrap();
        let song_id = created_json["song_id"].as_str().unwrap().to_string();

        let fetched = mcp
            .get_song(Parameters(SongIdArgs { song_id: song_id.clone() }))
            .await
            .unwrap();
        let song: medley_core::domain::models::Song = serde_json::from_str(&fetched).unwrap();
        assert_eq!(song.song_id, song_id);
        assert_eq!(song.title, "MCP Song");
    }

    #[tokio::test]
    async fn update_and_delete_song() {
        let server = MockServer::start().await;
        mount_voyage_embeddings(&server).await;

        let (_db, _edge, state) = test_state(&server.uri()).await;
        let mcp = MedleyMcp::new(state);

        let created = with_mcp_authenticated(
            true,
            mcp.add_song(Parameters(AddSongArgs {
                title: "To Update".into(),
                youtube_url: "https://www.youtube.com/watch?v=abc12345678".into(),
                lyrics: "original".into(),
                bpm: 90.0,
                key: "D".into(),
            })),
        )
        .await
        .unwrap();
        let song_id = serde_json::from_str::<serde_json::Value>(&created).unwrap()["song_id"]
            .as_str()
            .unwrap()
            .to_string();

        let updated = with_mcp_authenticated(
            true,
            mcp.update_song(Parameters(UpdateSongArgs {
                song_id: song_id.clone(),
                title: Some("Updated Title".into()),
                youtube_url: None,
                lyrics: None,
                bpm: None,
                key: None,
            })),
        )
        .await
        .unwrap();
        let updated_json: serde_json::Value = serde_json::from_str(&updated).unwrap();
        assert_eq!(updated_json["song"]["title"], "Updated Title");

        let deleted = with_mcp_authenticated(
            true,
            mcp.delete_song(Parameters(SongIdArgs {
                song_id: song_id.clone(),
            })),
        )
        .await
        .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&deleted).unwrap()["success"],
            true
        );

        assert!(mcp
            .get_song(Parameters(SongIdArgs { song_id }))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn list_songs_empty_catalog() {
        let (_db, _edge, state) = test_state("http://127.0.0.1:9").await;
        let mcp = MedleyMcp::new(state);
        let page = mcp
            .list_songs(Parameters(ListSongsArgs {
                q: None,
                key: None,
                bpm_min: None,
                bpm_max: None,
                limit: None,
                last_id: None,
                last_rank: None,
            }))
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&page).unwrap();
        assert_eq!(parsed["total"], 0);
    }

    #[tokio::test]
    async fn search_songs_returns_json_payload() {
        let server = MockServer::start().await;
        mount_voyage_embeddings(&server).await;

        let (_db, _edge, state) = test_state(&server.uri()).await;
        let mcp = MedleyMcp::new(state);

        with_mcp_authenticated(
            true,
            mcp.add_song(Parameters(AddSongArgs {
                title: "Searchable".into(),
                youtube_url: "https://www.youtube.com/watch?v=search12345".into(),
                lyrics: "find me in the index".into(),
                bpm: 110.0,
                key: "A".into(),
            })),
        )
        .await
        .unwrap();

        let out = mcp
            .search_songs(Parameters(SearchSongsArgs {
                query: "find me".into(),
                keys: None,
                bpm_min: None,
                bpm_max: None,
                limit: Some(5),
            }))
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(parsed["results"].is_array());
        assert!(parsed["total"].as_u64().unwrap() >= 1);
    }
}
