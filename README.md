# Flood + Hurricane Viewer

A comprehensive Streamlit application for visualizing coastal flooding and hurricane data using Google Earth Engine. This interactive web app allows users to explore sea level rise projections, hurricane tracks, and flood risk assessments.

## Features

- **Interactive Map**: Explore coastal areas with satellite imagery and flood depth visualization
- **Sea Level Analysis**: View sea level anomaly data from 1993-2022 with monthly variations
- **Hurricane Tracking**: Analyze historical hurricane data from NOAA IBTrACS v4 database
- **Flood Risk Assessment**: Calculate flood depths based on elevation and sea level projections
- **Real-time Analytics**: Get elevation, sea level, and flood statistics for any clicked location
- **Time Series Visualization**: Plot sea level trends over time for specific locations

## Prerequisites

- Python 3.8 or higher
- Google Earth Engine account and authentication
- Internet connection for data access

## Installation

### Option 1: Using requirements.txt (Recommended)

1. Clone this repository:
```bash
git clone <your-repo-url>
cd CoastalFloodHurricane
```

2. Create a virtual environment:
```bash
python -m venv hurricane_env
source hurricane_env/bin/activate  # On Windows: hurricane_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Manual Installation

```bash
pip install streamlit==1.50.0
pip install streamlit-folium==0.25.2
pip install earthengine-api==1.6.9
pip install geemap==0.36.3
pip install folium==0.20.0
pip install pandas==2.3.2
pip install matplotlib==3.10.6
pip install requests==2.32.5
pip install numpy==2.3.3
pip install pillow==11.3.0
```

## Authentication

Before running the app, you need to authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

Follow the prompts to authenticate using your Google account.

## Running the Application

### Local Development

```bash
streamlit run streamlit_app4_hurricane3.py
```

The app will be available at `http://localhost:8501`

### Production Deployment

**Important Note**: GitHub Pages cannot host Streamlit applications directly as it only serves static content. Streamlit apps require a Python server environment.

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and main file (`streamlit_app4_hurricane3.py`)
5. Add the following secrets in the Streamlit Cloud dashboard:
   - No additional secrets needed for basic Earth Engine access

**Advantages:**
- Free hosting
- Automatic deployments from GitHub
- Built-in authentication handling
- Easy to share and update

### Option 2: Heroku

1. Create a `Procfile`:
```
web: streamlit run streamlit_app4_hurricane3.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create a `runtime.txt`:
```
python-3.11.0
```

3. Deploy to Heroku following their standard process

### Option 3: Railway

1. Connect your GitHub repository to Railway
2. Railway will automatically detect it's a Streamlit app
3. Deploy with one click

### Option 4: Google Cloud Run

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app4_hurricane3.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

2. Deploy to Google Cloud Run

## Usage

1. **Select Parameters**: Use the sidebar to choose:
   - Sea level rise year (1993-2022)
   - Month for analysis
   - Additional sea level rise adjustment
   - Hurricane search radius

2. **Navigate the Map**: 
   - Click anywhere on the map to analyze that location
   - Use the search box to find specific locations
   - Toggle between different map layers

3. **Analyze Data**:
   - View flood depth visualization
   - Check hurricane tracks and impact areas
   - Examine elevation and sea level statistics
   - Review time series plots

## Data Sources

- **Elevation Data**: USGS SRTM Global 1 Arc-Second
- **Sea Level Data**: Custom sea level anomaly dataset (requires project access)
- **Hurricane Data**: NOAA IBTrACS v4 database
- **Basemap**: Google Maps satellite imagery

## Troubleshooting

### Earth Engine Authentication Issues
- Ensure you've run `earthengine authenticate`
- Check that your Google account has Earth Engine access
- Verify your internet connection

### Data Access Issues
- Some features require access to the 'sea-level-analysis' project
- The app will show warnings for unavailable data sources
- Fallback data is provided where possible

### Performance Issues
- Large search radii may slow down hurricane queries
- Consider reducing the analysis scale for faster processing
- The app uses caching to improve performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Support

For issues and questions:
- Check the troubleshooting section above
- Open an issue on GitHub
- Review the Streamlit documentation for general app issues








