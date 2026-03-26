from flask import Flask, render_template, request
import csv
import time

app = Flask(__name__)

# Initialize the CSV file with headers
csv_file = 'geolocation_data.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'latitude', 'longitude', 'accuracy', 'altitude', 'altitudeAccuracy', 'heading', 'speed'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/location', methods=['POST'])
def location():
    data = request.get_json()
    timestamp = time.time()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    accuracy = data.get('accuracy')
    altitude = data.get('altitude')
    altitudeAccuracy = data.get('altitudeAccuracy')
    heading = data.get('heading')
    speed = data.get('speed')
    # Print to terminal
    print(f"Received data: {data}")
    # Append to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, latitude, longitude, accuracy, altitude, altitudeAccuracy, heading, speed])
    return 'OK'

if __name__ == '__main__':
    # Use the server certificate and key
    app.run(debug=True, host='0.0.0.0', port=8000, ssl_context=('server.crt', 'server.key'))
