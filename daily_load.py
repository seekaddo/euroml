"""
Copyright (c) Dennis Kwame Addo

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
import os.path
import logging

logging.basicConfig(level=logging.INFO)

eml_url = "https://www.win2day.at/lotterie/euromillionen"
# euarch_url = "https://www.euro-millions.com/de-at/{}-zahlen-archiv"
yr_count = 2023

# eml_url = euarch_url.format(yr_count)
file_name = "{}_eml.json".format(yr_count)

import requests
from bs4 import BeautifulSoup


def extract_balls(dac):
    # Find all the <li> elements with the class "resultBall" or "lucky-star"
    balls_raw = dac.find_all("span", class_=["ball", "star"])
    lucky_stars = []
    balls = []

    # Loop through each <li> element and get its text content and class attribute value
    for ball in balls_raw:
        # Convert the text content to an integer
        number = int(ball.text)
        # Get the class attribute value
        class_value = ball["class"]
        # Check if the class attribute value contains "lucky-star"
        if "star" in class_value:
            # Append the number to the lucky stars array
            lucky_stars.append(number)
        # Check if the class attribute value contains "ball"
        elif "euro" in class_value:
            # Append the number to the balls array
            balls.append(number)

    return [balls, lucky_stars]


def save_to_jsonl(out_file_name, data_out):
    # Open the file in write mode
    with open(out_file_name, "w") as f:
        json.dump(data_out, f)


def rdata_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)  # Pass the file object here
    else:
        return {}


# Create a session object that can store cookies
session = requests.Session()
# Get the main page of the website and store the cookies
response = session.get(eml_url)
# Get the data from the website using the same session and cookies
data = session.get(eml_url, cookies=response.cookies)

# Create a soup object from the HTML content
soup = BeautifulSoup(data.text, "html.parser")
# Find the table element with id "resultsTable" or class "mobFormat"
items_acord = soup.find("div", {"class": "accordion"})
# Find the table body element with tag name "tbody"


items_acord = items_acord.find_all("div", {"class": "accordion-item"})

dict_eml = {}

# Print the content of each result_row element
for result_row in items_acord:
    span = result_row.find("span", {"class": "accordion-title"})
    drawdiv = result_row.find("div", {"class": "win-numbers"})
    # dr_date = result_row.get("title").split(",")[1]
    # drw_lst = extract_balls(data)
    text = span.get_text()
    datedr = text.split(",")[1].strip()
    draml = extract_balls(drawdiv)
    dict_eml[datedr] = draml

    # print("Date:{0} --> {1}".format(datedr, draml ))

print(dict_eml)
file_name = "./dataset/{}_eml.json".format(yr_count)
logging.info("Loading json eml data... at: {}".format(file_name))
dryr_data = rdata_json(file_name)
# print(dryr_data)

for k, v in dict_eml.items():
    if k not in dryr_data:
        logging.info("Data not found: --> add {0}:{1}".format(k, v))
        dryr_data[k] = v
save_to_jsonl(file_name, dryr_data)
