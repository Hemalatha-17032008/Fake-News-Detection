import csv
import random


fake_file = "data/Fake.csv"
true_file = "data/True.csv"
output_file = "data/news.csv"


rows = []


# Read fake news
with open(fake_file, "r", encoding="utf-8", errors="ignore") as file:
    reader = csv.DictReader(file)

    for row in reader:
        title = row.get("title", "")
        text = row.get("text", "")

        if title and text:
            rows.append([title, text, "FAKE"])


# Read real news
with open(true_file, "r", encoding="utf-8", errors="ignore") as file:
    reader = csv.DictReader(file)

    for row in reader:
        title = row.get("title", "")
        text = row.get("text", "")

        if title and text:
            rows.append([title, text, "REAL"])


# Shuffle the data
random.seed(42)
random.shuffle(rows)


# Save combined dataset
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["title", "text", "label"])

    writer.writerows(rows)


print("Dataset prepared successfully!")
print("Total articles:", len(rows))
print("Saved as:", output_file)