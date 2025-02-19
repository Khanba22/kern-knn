from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def clean_and_load_data():
    complete_df = pd.DataFrame()
    
    for file in os.listdir("./dealer_data"):
        if file.endswith('.csv'):
            # Extract provider name correctly (assuming format: qwuhidbsd - provider_name.csv)
            providerName = file.split(" - ")[-1].replace(".csv", "").strip()

            # Read CSV file
            df = pd.read_csv(os.path.join("./dealer_data", file))

            # Add provider name column
            df["Provider"] = providerName

            # Check if 'Pincode' column exists
            if 'Pincode' not in df.columns:
                continue

            # Drop rows where Pincode is missing
            df = df.dropna(subset=['Pincode'])

            # Extract only numeric part of Pincode and convert to integer
            df['Pincode'] = df['Pincode'].astype(str).str.extract(r'(\d+)')
            df = df.dropna(subset=['Pincode'])
            df['Pincode'] = df['Pincode'].astype(int)

            # Append to complete DataFrame
            complete_df = pd.concat([complete_df, df], ignore_index=True)
    
    return complete_df


def create_knn_model(df):
    knn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
    knn.fit(df[['Pincode']])
    return knn

# Load and clean data
df = clean_and_load_data()
print(df.head())
knn = create_knn_model(df)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/nearest', methods=['GET'])
def nearest():
    pincode = int(request.args.get('pincode'))
    
    distances, indices = knn.kneighbors([[pincode]])
    nearest_dealers = df.iloc[indices[0]].to_dict(orient='records')
    
    return jsonify(nearest_dealers)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
