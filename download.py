"""
Copyright (c) Dennis Kwame Addo

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

# eml_url = "https://www.win2day.at/lotterie/euromillionen"
euarch_url = "https://www.euro-millions.com/de-at/{}-zahlen-archiv"
yr_count = 2023

eml_url = euarch_url.format(yr_count)
file_name = "{}_eml.json".format(yr_count)


headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-US,en;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
}
import requests
from bs4 import BeautifulSoup


def extract_balls(dac):
    # Find all the <li> elements with the class "resultBall" or "lucky-star"
    balls_raw = data.find_all("li", class_=["resultBall", "lucky-star"])
    lucky_stars = []
    balls = []

    # Loop through each <li> element and get its text content and class attribute value
    for ball in balls_raw:
        # Convert the text content to an integer
        number = int(ball.text)
        # Get the class attribute value
        class_value = ball["class"]
        # Check if the class attribute value contains "lucky-star"
        if "lucky-star" in class_value:
            # Append the number to the lucky stars array
            lucky_stars.append(number)
        # Check if the class attribute value contains "ball"
        elif "ball" in class_value:
            # Append the number to the balls array
            balls.append(number)

    return [balls, lucky_stars]


def save_to_jsonl(out_file_name, data_out):
    # Open the file in write mode
    with open(out_file_name, "w") as f:
        json.dump(data_out, f)


# Create a session object that can store cookies
#session = requests.Session()
# Get the main page of the website and store the cookies
#response = session.get(eml_url, headers)
# Get the data from the website using the same session and cookies
#data = session.get(eml_url, cookies=response.cookies, headers)
data = requests.get(eml_url, headers)

# Create a soup object from the HTML content
soup = BeautifulSoup(data.text, "html.parser")
# Find the table element with id "resultsTable" or class "mobFormat"
table = soup.find("table", {"id": "resultsTable", "class": "mobFormat"})
# Find the table body element with tag name "tbody"

tbody = table.find("tbody")
# Find all tr elements with class "resultRow" within the tbody element
result_rows = soup.find_all("tr", {"class": "resultRow"})

dict_eml = {}

# Print the content of each result_row element
for result_row in result_rows:
    data = result_row.find("ul", {"class": "balls"})
    dr_date = result_row.get("title").split(",")[1]
    drw_lst = extract_balls(data)
    dict_eml[dr_date] = drw_lst
    #print("Date:{0} --> {1}".format(dr_date, drw_lst ))

print(dict_eml)
save_to_jsonl("./dataset/{}".format(file_name), dict_eml)

