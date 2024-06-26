
import pandas as pd
import numpy as np
import requests
import json

x = requests.get("https://raw.githubusercontent.com/ashwath92/nobel_json/main/laureates.json")
y = requests.get("https://raw.githubusercontent.com/ashwath92/nobel_json/main/countries.json")

# y.text

dict1 = json.loads(x.text)
dict2 = json.loads(y.text)

# dict1.keys()

# dict1['laureates']



laurates_data = pd.DataFrame(dict1['laureates'])
country_data = pd.DataFrame(dict2['countries'])
# country_data = pd.read_json("https://raw.githubusercontent.com/ashwath92/nobel_json/main/countries.json")

# laurates_data

# country_data

combined_df = laurates_data[['id','firstname','surname','born','prizes','gender','bornCountryCode']]

# combined_df

combined_df['name'] = combined_df['firstname']+' '+combined_df['surname']

# combined_df

combined_df['dob'] = combined_df['born']

# combined_df

combined_df['prizes']

combined_df['year'] = combined_df['prizes'].apply(lambda x: x[0]['year'])

combined_df['unique_prize_years'] = combined_df.groupby(combined_df.index)['year'].agg(lambda x: ';'.join(x))

# combined_df

combined_df['category'] = combined_df['prizes'].apply(lambda x: x[0]['category'])

combined_df['unique_prize_category'] = combined_df.groupby(combined_df.index)['category'].agg(lambda x: ';'.join(x))

# combined_df


# combined_df = laurates_data['bornCountryCode']

# combined_df

combined_df = combined_df.merge(country_data,how='inner', left_on='bornCountryCode', right_on='code')

# combined_df
combined_df.to_csv("result.csv")


