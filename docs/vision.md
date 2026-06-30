# Medley Recommender Roadmap

## Vision

> **Build the open knowledge platform for worship music that AI can understand and reason over.**

Not another worship planning app.

Not another vector search demo.

A semantic knowledge graph with an MCP interface that enables any LLM to understand worship music, recommend songs, and build meaningful worship sets.

The technology will evolve. The dataset is the long-term asset.

---

# Phase 1 — Open Source Personal Tool

## Goal

Solve my own problem while building in the open.

## Deliverables

- Rust
- Qdrant
- Voyage AI embeddings
- CLI
- Docker
- GitHub repository
- Documentation
- Semantic search
- BPM & key extraction

## Users

- Me
- Engineers interested in AI, Rust, and semantic search
- Worship leaders who want to self-host

## Outcome

A practical personal tool that also demonstrates semantic search, recommendation systems, embeddings, and Rust.

---

# Phase 2 — MCP Server

## Goal

Allow any AI assistant to use the recommender as a tool.

## MCP Tools

- `search_by_theme`
- `similar_songs`
- `build_medley`
- `find_transition`
- `song_details`

Example prompt:

> Build a five-song worship set around Hebrews 11. Start around 70 BPM, finish around 120 BPM, avoid songs we've sung recently, and keep transitions smooth.

The LLM performs the planning while the MCP provides domain-specific retrieval and reasoning.

---

# Phase 3 — The Dataset

The dataset becomes the project's moat.

Instead of storing only:

```yaml
title
lyrics
bpm
key
```

each song evolves into a rich, structured representation.

## Basic

General identification and administrative information.

| Field | Description |
|-------|-------------|
| Title | Song title. |
| Artist | Original artist or band. |
| Album | Album or release. |
| Languages | Languages of the lyrics. |
| Duration | Total song length. |
| Copyright | Copyright holder or publisher. |
| CCLI Number | Optional CCLI identifier. |

---

## Musical

Describes the musical characteristics of the song.

| Field | Description |
|-------|-------------|
| BPM | Beats per minute (tempo). |
| Key | Musical key. |
| Time Signature | Meter (4/4, 3/4, 6/8, etc.). |
| Energy | Overall musical intensity (0–1). |
| Valence | Emotional positivity (0–1). Low = reflective, High = joyful. |
| Danceability | Rhythmic "groove" of the song. Useful for distinguishing musical styles. |
| Acousticness | Likelihood the arrangement is primarily acoustic. |
| Instrumentalness | Likelihood the song contains little or no vocals. |
| Loudness | Average perceived volume. |
| Vocal Range | Lowest and highest notes required for the lead singer. |
| Difficulty | Estimated difficulty for the worship team. |
| Dynamic Curve | How the song develops (e.g. Soft → Build → Big Finish). |
| Intro Length | Time before vocals begin. |
| Outro Length | Time after the final lyric ends. |
| Male/Female Lead | Typical vocal lead arrangement. |


### Dynamic Curve

Describes how the musical intensity evolves throughout the song, rather than its average intensity.

Unlike **Energy**, which represents the song's overall intensity, the **Dynamic Curve** captures the progression of the arrangement from beginning to end. This helps AI reason about worship flow and transitions between songs.

#### Shape

The overall pattern of the song's dynamics.

| Value | Description |

|-------|-------------|

| `FLAT` | Little variation in intensity throughout the song. |

| `BUILD` | Gradually increases in intensity. |

| `BUILD_DROP` | Builds to a climax, then returns to a quieter ending. |

| `WAVE` | Multiple rises and falls in intensity. |

| `EXPLOSIVE` | Quickly reaches high intensity and remains energetic. |

| `REFLECTIVE` | Predominantly soft and intimate throughout. |

| `ANTHEM` | Starts strong and remains consistently energetic. |

| `CRESCENDO` | Continuously builds until the very end. |

#### Profile

A normalized representation of the song's energy over time.

Example:

```yaml

dynamic_curve:

  shape: BUILD

  profile: [0.2, 0.3, 0.4, 0.7, 1.0, 0.5]

```

Where each value represents the relative energy at evenly spaced points throughout the song.

#### Why it matters

Dynamic curves allow AI to recommend songs that not only share similar themes, but also create a natural emotional progression during worship.

For example, a worship set can be planned to:

- Begin quietly and reflectively

- Gradually build in intensity

- Reach a climactic final song

- End with a gentle ministry moment

This makes **Dynamic Curve** one of the most important musical features for building coherent worship sets.

---

## Semantic

Represents what the song communicates.

| Field | Description |
|-------|-------------|
| Embedding | Vector representation used for semantic search. |
| Semantic Tags | High-level concepts (Faith, Grace, Hope, Holiness, etc.). Or More specific topics (Cross, Resurrection, Salvation, Healing, Holy Spirit, etc.). |
| Theological Summary | Human- or AI-generated summary of the song's doctrine. |
| Scripture References | Bible passages referenced directly or indirectly. like GEN.1.1 |

---

## Worship

Describes how the song functions during a worship service.

| Field | Description |
|-------|-------------|
| Service Role | Opening, Worship, Response, Communion, Offering, Closing, etc. |
| Congregational Familiarity | How likely an average congregation knows the song. |
| Church Familiarity | Whether the local church regularly sings the song. |
| Choir Friendlyiness | Suitable for choir arrangements. |
| Band Difficultiness | Difficulty specifically for the worship team. |

---

## Relationships

Represents how songs connect to one another.

| Field | Description |
|-------|-------------|
| Similar Songs | Songs with similar lyrical meaning or musical feel. |
| Transition Compatibility | How smoothly the song flows into another song. |
| Common Medleys | Songs frequently paired together. |
| Same Artist | Songs by the same artist. |
| Same Theme | Songs sharing similar theological themes. |
| Same Scripture | Songs referencing similar Bible passages. |
| Same Musical Style | Songs with similar musical style or arrangement. |

---

## Usage

Captures real-world worship experience.

| Field | Description |
|-------|-------------|
| Times Used | Number of times the song has been used. |
| Last Used | Most recent service date. |
| Congregation Response | Estimated engagement or participation. |
| Worship Leader Notes | Personal notes and observations. |
| Works Well With | Songs that naturally transition before or after. |
| Seasonal Usage | Christmas, Easter, Pentecost, Thanksgiving, etc. |
| Average Transition Rating | Community-rated transition quality. |

---

# Phase 4 — Knowledge Graph

Move beyond a flat vector database.

Represent songs as connected entities with meaningful relationships.

Example relationships:

- Similar themes
- Musical compatibility
- Transition quality
- Shared Scripture
- Shared theology
- Same artist
- Similar arrangement
- Frequently used together

Recommendations become a combination of:

- Semantic search
- Graph traversal
- Rule-based filtering
- Musical constraints

instead of pure vector similarity.

---

# Phase 5 — Community

Open the dataset for contributions.

Contributors can submit:

- BPM corrections
- Key corrections
- Theme tags
- Scripture references
- Transition ratings
- Worship usage
- Song relationships
- New metadata

Over time, the dataset continuously improves.

---

# Phase 6 — AI Worship Assistant

Move beyond search.

Users interact naturally with an AI assistant.

Example:

> The sermon is about Romans 8.
>
> Build a 25-minute worship set.
>
> Start reflective and finish celebratory.
>
> Keep transitions smooth.
>
> Avoid songs we've sung during the past month.
>
> Prefer songs our congregation already knows.

The assistant reasons over the knowledge graph and produces an explainable recommendation.

---

# Long-Term Architecture

```text
                 ChatGPT
                  Claude
                  Gemini
                    │
                    ▼
               MCP Server
                    │
      ┌─────────────┴─────────────┐
      │                           │
Semantic Search             Rule Engine
      │                           │
      └─────────────┬─────────────┘
                    ▼
         Worship Knowledge Graph
                    │
      Songs + Metadata + Relationships
```

---

# Why This Project Exists

General-purpose LLMs can recommend worship songs.

However, they typically lack:

- Church-specific repertoire
- Consistent BPM and key information
- Transition awareness
- Structured worship metadata
- Knowledge of how songs function within a worship service

This project fills those domain-specific gaps by combining semantic search, musical metadata, and practical worship knowledge.

---

# Tech Stack

- Rust
- Qdrant
- Voyage AI
- MCP
- Semantic Search
- Embeddings
- Knowledge Graph

---

# Ultimate Goal

Build the **open knowledge graph for worship music** that any AI assistant can reason over.

The technology stack will continue to evolve.

The true long-term value lies in a curated dataset that combines:

- Theology
- Music theory
- Worship practice
- Semantic relationships
- Community knowledge
- Real-world worship experience

into an open platform for AI-powered worship planning.