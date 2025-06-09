# Anime Dataset Suggestions for Final Project

## Recommended Anime Datasets on Kaggle:

### 1. MyAnimeList Dataset (Most Popular)
- **URL:** https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- **Description:** Large dataset with 12,000+ anime entries from MyAnimeList
- **Features:** Name, genre, type, episodes, rating, members, popularity
- **Size:** ~3MB, 12,294 anime entries
- **Good for:** Rating prediction, genre analysis, popularity trends
- **Download command:** `kaggle datasets download -d cooperunion/anime-recommendations-database`

### 2. Anime Streaming Platforms Dataset
- **URL:** https://www.kaggle.com/datasets/vishalmane10/anime-dataset-with-streaming-platform-info
- **Description:** Anime data with streaming platform availability
- **Features:** Title, genres, studio, year, streaming platforms (Netflix, Crunchyroll, etc.)
- **Good for:** Platform analysis, availability trends
- **Download command:** `kaggle datasets download -d vishalmane10/anime-dataset-with-streaming-platform-info`

### 3. Anime Statistics Dataset (2023)
- **URL:** https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset
- **Description:** Updated MyAnimeList data with recent anime
- **Features:** Title, score, genres, studios, aired dates, popularity, members
- **Good for:** Recent trends analysis, studio performance
- **Download command:** `kaggle datasets download -d dbdmobile/myanimelist-dataset`

### 4. Jikan API Anime Dataset
- **URL:** https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist
- **Description:** Comprehensive anime dataset from Jikan API
- **Features:** Detailed anime information including ratings, genres, studios, aired dates
- **Good for:** Comprehensive analysis, trend prediction
- **Download command:** `kaggle datasets download -d andreuvallhernndez/myanimelist`

### 5. Anime Recommendation System Dataset
- **URL:** https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020
- **Description:** Updated version with user ratings and recommendations
- **Features:** Anime info + user rating data (7M+ ratings)
- **Good for:** Recommendation systems, user behavior analysis
- **Download command:** `kaggle datasets download -d hernan4444/anime-recommendation-database-2020`

## Dataset Selection for Your Project:

### Recommended: MyAnimeList Dataset (Option 1)
**Why this is perfect for your project:**
- Contains popularity metrics (members, favorites)
- Has genres for analysis
- Includes ratings and episode counts
- Time-based data (though not streaming-specific)
- Large enough for meaningful analysis
- Clean and well-structured

### Alternative: Anime Streaming Platforms Dataset (Option 2)
**If you want streaming-specific analysis:**
- Shows which platforms host which anime
- Good for platform comparison analysis
- Smaller but focused on streaming

## Research Questions You Can Answer:

1. **Popularity Analysis:**
   - What genres are most popular in recent years?
   - Which studios produce the highest-rated anime?
   - How does episode count affect popularity?

2. **Trend Analysis:**
   - How have anime ratings changed over time?
   - What are the emerging popular genres?
   - Which studios are gaining popularity?

3. **Predictive Modeling:**
   - Can we predict anime ratings based on features?
   - What factors contribute most to anime popularity?
   - Can we classify anime by genre based on other features?

4. **Streaming Platform Analysis (if using streaming dataset):**
   - Which platforms have the most popular anime?
   - How does platform availability affect ratings?
   - What genres are most common on each platform?

## Setup Instructions:

1. **Install Kaggle API:**
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle credentials:**
   - Go to https://www.kaggle.com/account
   - Create API token (downloads kaggle.json)
   - Move to ~/.kaggle/kaggle.json
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download your chosen dataset:**
   ```bash
   # Example for MyAnimeList dataset
   kaggle datasets download -d cooperunion/anime-recommendations-database
   unzip anime-recommendations-database.zip -d data/
   ```

4. **Verify download:**
   ```bash
   ls data/
   head data/anime.csv  # Check the first few rows
   ```

## Project Adaptation Notes:

Since you're interested in "most streamed anime shows for the past 5 years":

- **Focus on recent data:** Filter datasets for anime from 2019-2024
- **Use popularity metrics:** Members, favorites, or scores as proxies for "streaming popularity"
- **Analyze trends:** Look at how popularity changes over time
- **Genre analysis:** Identify trending genres in recent years
- **Studio analysis:** Find which studios produce the most popular recent content

The MyAnimeList dataset is your best bet as it has the most comprehensive data with popularity metrics that can serve as proxies for streaming popularity!

