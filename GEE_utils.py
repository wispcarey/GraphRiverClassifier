from typing import List, Dict, Optional, Union
from duckduckgo_search import DDGS
import ee
import geemap

import openai
from openai import OpenAI
from IPython.display import display

import time
import os
import json

def search_ddg(keyword: str,
               areas: Dict[str, int],
               news_params: Optional[Dict[str, Union[str, int]]] = None
               ) -> Dict[str, Union[List[Dict[str, Optional[str]]], Dict[str, Optional[str]]]]:
    """
    Use the DuckDuckGo API for searching.

    Args:
        keyword: The keyword for the search.
        areas: A dictionary specifying the search scopes and the number of search results, where the keys can be any combination of 'text', 'answers', 'news', 'suggestions', and the values are positive integers indicating the number of results.
        news_params: Parameters for news search, which is a dictionary with the following keys:
            - region: The geographical region for the news, default is 'wt-wt', representing worldwide.
            - safesearch: Whether to enable safe search, default is 'Off', meaning disabled.
            - timelimit: The time limit for the news, default is 'm', representing news from the last month.

    Returns:
        A dictionary containing the search results.
    """
    if news_params is None:
        news_params = {"region": "wt-wt", "safesearch": "Off", "timelimit": "m"}

    # Initialize DDGS instance
    with DDGS() as ddgs:
        results = {}
        info_strings = {}
        # Perform a text search with the keyword and keep the top n results
        if 'text' in areas:
            try:
                results["text"] = [r for r in ddgs.text(keyword)][:areas['text']]
            except Exception:
                results["text"] = []
        # Get instant answers and keep the top n results
        if 'answers' in areas:
            try:
                results["answers"] = [r for r in ddgs.answers(keyword)][:areas['answers']]
            except Exception:
                results["answers"] = []
        # Fetch news and keep the top n results
        if 'news' in areas:
            try:
                results["news"] = [r for r in ddgs.news(keyword,
                                                        region=news_params["region"],
                                                        safesearch=news_params["safesearch"],
                                                        timelimit=news_params["timelimit"]
                                                        )
                                   ][:areas['news']]
            except Exception:
                results["news"] = []

        if 'suggestions' in areas:
            try:
                results["suggestions"] = [r for r in ddgs.suggestions(keyword)][:areas['suggestions']]
            except Exception:
                results["suggestions"] = []

    # Return all results
    return results

def get_lon_lat(location):
    search_keyword = f"Longtitude and latitude of {location}."
    search_areas = {'text': 5}
    results = search_ddg(keyword=search_keyword, areas=search_areas, news_params=None)

    ref_info = "\n\n".join([result_text['body'] for result_text in results['text']])

    prompt = f"""Given the location "{location}", provide the longitude and latitude values in a JSON format. The response should include keys 'longitude' and 'latitude' with their respective values. For example, if the location was 'Central Park, New York', the output should be in the following format:

{{"longitude": -73.965355, "latitude": 40.782865}}

You should choose a location near to water like lake, sea or rivers.

Now, please give me the longitude and latitude values of the location "{location}" in a similar JSON format.

Here are some references about "{location}":

{ref_info}

Your response (in JSON format):
"""

    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        temperature=0.9,
        max_tokens=250,
    )

    location_dict = json.loads(chat_completion.choices[0].message.content)
    return location_dict['longitude'], location_dict['latitude'], ref_info

def get_landsat_data(lon, lat,
                     dataset='LANDSAT/LT05/C01/T1_TOA',
                     start_date='2010-01-01',
                     end_date='2010-12-31',
                     lonRange=0.40,
                     latRange=0.20,
                     bands=None,
                     display_image=False,
                     save_data=True,
                     save_name='Landsat5_image',
                     scale=30,
                     region=None,
                     folder=None,
                     ):
    if bands is None:
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']

    # 1. Load the Landsat 5 Image Collection.
    landsat5Collection = ee.ImageCollection(dataset)

    # 2. Set the time range.
    startDate = ee.Date(start_date)
    endDate = ee.Date(end_date)
    landsat2010 = landsat5Collection.filterDate(startDate, endDate)

    # Define rectangle parameters.
    # lon = -149.50  # Center longitude of the rectangle. Uncomment and adjust as necessary.
    # lat = 64.00    # Center latitude of the rectangle. Uncomment and adjust as necessary.
    # lonRange = 0.40  # Longitude range.
    # latRange = 0.20  # Latitude range.

    # 3. Create a rectangle using the defined parameters.
    if region is None:
        rectangle = ee.Geometry.Rectangle(
            [lon - lonRange / 2, lat - latRange / 2, lon + lonRange / 2, lat + latRange / 2])
    else:
        rectangle = region

    # 4. Set the geographical bounds to the rectangle.
    landsatAlaska2010 = landsat2010.filterBounds(rectangle)

    # 5. Filter for images with low cloud cover.
    lowCloudImages = landsatAlaska2010.filter(ee.Filter.lt('CLOUD_COVER', 10))

    # Use a reducer for a smoother image composition.
    smoothMosaicImage = lowCloudImages.reduce(ee.Reducer.median())

    # 6. Select bands B1 to B5, B7. Note that after using the reducer, band names will change and need a suffix.
    selectedBandsSmooth = smoothMosaicImage.select([band + '_median' for band in bands])

    # Set visualization parameters.
    if display_image:
        vis_params = {
            'bands': ['B3_median', 'B2_median', 'B1_median'],
            'min': 0,
            'max': 0.3,
            'gamma': 1.4
        }

        # Create a geemap map object.
        m = geemap.Map()

        # Add the smooth Landsat 5 image to the map using the visualization parameters.
        m.addLayer(selectedBandsSmooth, vis_params, 'Smooth Landsat 5 mosaic')

        # Add the rectangle as a layer.
        m.addLayer(rectangle, {'color': 'red'}, 'Rectangle Region')

        # Center the map to the rectangle's location.
        m.centerObject(rectangle, zoom=6)

        # Display the map.
        display(m)

    # Export the specified rectangular region of the image.
    if save_data:
        task = ee.batch.Export.image.toDrive(**{
            'image': selectedBandsSmooth,
            'description': save_name,
            'scale': scale,
            'region': rectangle,
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e9,
            'folder': folder  # Specify the folder name here if needed.
        })
        task.start()

    # Warning if the image is composed of more than one Landsat patch.
    # Note: This simplistic check assumes a single image should cover the rectangle. For a more accurate check, consider image footprint overlaps with the rectangle.
    if lowCloudImages.size().getInfo() > 1:
        print("Warning: The final image is composed of more than one Landsat patch.")