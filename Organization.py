import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, f1_score

# Define the dataset (replace with your actual dataset source)
data = {
    "Country": [
        "Papua New Guinea", "Finland", "China", "Turkmenistan", "Mauritius", "Bahamas",
        "Pakistan", "Heard Island and McDonald Islands", "Kuwait", "Uzbekistan", "Bouvet Island (Bouvetoya)",
        "Denmark", "Liberia", "United Arab Emirates", "Sweden", "Honduras", "Uganda", "Hong Kong", "Botswana",
        "Korea", "Luxembourg", "Guadeloupe", "Monaco", "Belgium", "South Africa", "Romania", "Czech Republic",
        "Christmas Island", "Philippines", "Australia", "Chad", "Zimbabwe", "Nepal", "Taiwan", "Kyrgyz Republic",
        "Bolivia", "Kenya", "Guatemala", "Belarus", "Jersey", "Grenada", "Cape Verde", "Trinidad and Tobago",
        "Sweden", "Benin", "Western Sahara", "Northern Mariana Islands", "Germany", "Canada", "Tonga",
        "French Southern Territories", "Korea", "Cote d'Ivoire", "Cote d'Ivoire", "Mayotte", "Cayman Islands",
        "Nigeria", "Marshall Islands", "Palau", "Turkey", "Timor-Leste", "Vietnam", "Reunion", "Brazil",
        "Eritrea", "United States Virgin Islands", "Falkland Islands (Malvinas)", "Luxembourg", "Northern Mariana Islands",
        "Netherlands Antilles", "Guernsey", "Uruguay", "Benin", "Suriname", "Pakistan", "Mongolia",
        "Svalbard & Jan Mayen Islands", "Togo", "Latvia", "Cuba", "Liechtenstein", "Djibouti", "Micronesia",
        "Cameroon", "Sweden", "Cocos (Keeling) Islands", "Mali", "United States Virgin Islands", "Eritrea",
        "Burundi", "Benin", "Gibraltar", "El Salvador", "Taiwan", "Zimbabwe", "Ethiopia", "Anguilla",
        "Falkland Islands (Malvinas)", "Kyrgyz Republic", "Togo"
    ]
}

# Define regions mapping
regions = {
    "Africa": ["Mauritius", "Liberia", "Botswana", "South Africa", "Zimbabwe", "Chad", "Kenya", "Cote d'Ivoire", "Benin", "Togo", "Eritrea", "Burundi", "Djibouti", "Mali", "Ethiopia"],
    "Asia": ["China", "Turkmenistan", "Pakistan", "Kuwait", "Uzbekistan", "United Arab Emirates", "Hong Kong", "Philippines", "Taiwan", "Kyrgyz Republic", "Mongolia", "Micronesia", "Turkey", "Timor-Leste", "Vietnam"],
    "Europe": ["Finland", "Denmark", "Sweden", "Romania", "Czech Republic", "Luxembourg", "Monaco", "Belgium", "Jersey", "Guernsey", "Latvia", "Liechtenstein", "Germany"],
    "Oceania": ["Papua New Guinea", "Australia", "Bouvet Island (Bouvetoya)", "Northern Mariana Islands", "Christmas Island", "Marshall Islands", "Palau", "Cocos (Keeling) Islands", "Falkland Islands (Malvinas)"],
    "Americas": ["Bahamas", "Honduras", "Grenada", "Canada", "Brazil", "Cuba", "Anguilla", "Cayman Islands", "El Salvador", "French Southern Territories"],
    "Middle East": ["United Arab Emirates", "Kuwait", "Turkey"]
}

# Function to categorize countries into regions
def get_region(country):
    for region, countries in regions.items():
        if country in countries:
            return region
    return "Unknown"

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Add a Region column
df["Region"] = df["Country"].apply(get_region)

# Output the result
print(df)

# Assuming we have true labels and predicted labels (for example purposes)
true_labels = ["Oceania", "Europe", "Asia", "Asia", "Africa"]  # Example true labels
predicted_labels = ["Oceania", "Europe", "Asia", "Africa", "Africa"]  # Example predicted labels

# Calculate precision, accuracy, and F1 score
precision = precision_score(true_labels, predicted_labels, average='macro')
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='macro')

# Print the results
print("Precision:", precision)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
