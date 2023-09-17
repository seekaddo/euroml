import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # Import the tabulate library

# Load the CSV data into a Pandas DataFrame
#data = pd.read_csv('./dataset/train_data.csv', parse_dates=['Date'], dayfirst=True)

# Read data from the first CSV file
file1 = './dataset/train_data.csv'
data1 = pd.read_csv(file1, parse_dates=['Date'], dayfirst=True)

# Read data from the second CSV file
file2 = './dataset/test_data.csv'
data2 = pd.read_csv(file2, parse_dates=['Date'], dayfirst=True)

# Concatenate the two DataFrames into one
data = pd.concat([data1, data2], ignore_index=True)

# Extract main numbers and lucky stars as lists of integers
data['main-numbers'] = data['main-numbers'].apply(lambda x: [int(num) for num in x.strip('[]').split(',')])
data['lucky-stars'] = data['lucky-stars'].apply(lambda x: [int(num) for num in x.strip('[]').split(',')])

# Calculate additional features
data['main-sum'] = data['main-numbers'].apply(lambda x: sum(x))
data['lucky-star-sum'] = data['lucky-stars'].apply(lambda x: sum(x))
data['main-range'] = data['main-numbers'].apply(lambda x: max(x) - min(x))
data['lucky-star-range'] = data['lucky-stars'].apply(lambda x: max(x) - min(x))
data['main-avg'] = data['main-numbers'].apply(lambda x: np.mean(x))
data['main-median'] = data['main-numbers'].apply(lambda x: np.median(x))
data['lucky-star-avg'] = data['lucky-stars'].apply(lambda x: np.mean(x))
data['lucky-star-median'] = data['lucky-stars'].apply(lambda x: np.median(x))

# Function to find consecutive triples in a list
def find_consecutive_triples(numbers):
    triples = []
    for i in range(len(numbers) - 2):
        if numbers[i] + 1 == numbers[i + 1] and numbers[i + 1] + 1 == numbers[i + 2]:
            triples.append([numbers[i], numbers[i + 1], numbers[i + 2]])
    return triples

# Find consecutive triples of Main Numbers
data['main-numbers-triples'] = data['main-numbers'].apply(find_consecutive_triples)

# Find consecutive triples of Lucky Stars
data['lucky-stars-triples'] = data['lucky-stars'].apply(find_consecutive_triples)

# Frequency of Consecutive Triples of Main Numbers
main_number_triples = [triple for triples in data['main-numbers-triples'] for triple in triples]
main_number_triples_freq = pd.Series(main_number_triples).value_counts().sort_index()

# Frequency of Consecutive Triples of Lucky Stars
lucky_star_triples = [triple for triples in data['lucky-stars-triples'] for triple in triples]
lucky_star_triples_freq = pd.Series(lucky_star_triples).value_counts().sort_index()

# Create subplots for each feature
fig, axs = plt.subplots(4, 2, figsize=(12, 8))
fig.tight_layout(pad=3.0)


# Plot 1: Frequency of Main Numbers
axs[0, 0].bar(data['main-numbers'].apply(lambda x: pd.Series(x)).stack().value_counts().sort_index().index.astype(str),
              data['main-numbers'].apply(lambda x: pd.Series(x)).stack().value_counts().sort_index())
axs[0, 0].set_title('Frequency of Main Numbers')

# Plot 2: Frequency of Lucky Stars
axs[0, 1].bar(data['lucky-stars'].apply(lambda x: pd.Series(x)).stack().value_counts().sort_index().index.astype(str),
              data['lucky-stars'].apply(lambda x: pd.Series(x)).stack().value_counts().sort_index())
axs[0, 1].set_title('Frequency of Lucky Stars')

# Plot 3: Sum of Main Numbers
axs[1, 0].plot(data['main-sum'])
axs[1, 0].set_title('Sum of Main Numbers')

# Plot 4: Sum of Lucky Stars
axs[1, 1].plot(data['lucky-star-sum'])
axs[1, 1].set_title('Sum of Lucky Stars')

# Plot 5: Range of Main Numbers
axs[2, 0].plot(data['main-range'])
axs[2, 0].set_title('Range of Main Numbers')

# Plot 6: Range of Lucky Stars
axs[2, 1].plot(data['lucky-star-range'])
axs[2, 1].set_title('Range of Lucky Stars')

# Plot 7: Average of Main Numbers
axs[3, 0].plot(data['main-avg'])
axs[3, 0].set_title('Average of Main Numbers')

# Plot 8: Average of Lucky Stars
axs[3, 1].plot(data['lucky-star-avg'])
axs[3, 1].set_title('Average of Lucky Stars')

# Save the figure as an image file
plt.savefig('lottery_features1.png')

# Show the plots
plt.show()

# Display Frequency of Consecutive Triples of Main Numbers in a table
main_number_triples_table = pd.DataFrame({'Triple': main_number_triples_freq.index.astype(str),
                                          'Frequency': main_number_triples_freq.values})
print("Frequency of Consecutive Triples of Main Numbers:")
print(tabulate(main_number_triples_table, headers='keys', tablefmt='grid'))

# Display
