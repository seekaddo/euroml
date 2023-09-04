"""
Copyright (c) Dennis Kwame Addo

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import csv
import os
import os.path
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
yr_count = 2022
file_name = "./dataset/{}_eml.json".format(yr_count)


def getdata_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)  # Pass the file object here
    else:
        return {}


def dumpto_csv(out_file, data):
    # Create a csv object using the open() function and the csv.writer() method
    with open(out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":

    data = [
        ["Date", "main-numbers", "lucky-stars"]
    ]
    for i in range(2):
        # logging.info("year with filen: {}".format(file_name))
        jsond = getdata_json(file_name)
        for k, v in jsond.items():
            vlk = [k, str(v[0]), str(v[1])]
            data.append(vlk)
            #print(vlk)
        print(jsond)
        yr_count += 1
        file_name = "./dataset/{}_eml.json".format(yr_count)

    # Sort the data by date
    sorted_data = sorted(data[1:], key=lambda x: datetime.strptime(x[0], "%d.%m.%Y"))
    sorted_data.insert(0, data[0])  # Reinsert the header row

    # Define the filename for your CSV file
    filename = "./dataset/test_data.csv"
    dumpto_csv(filename, sorted_data)
