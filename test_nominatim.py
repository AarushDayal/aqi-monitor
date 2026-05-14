import requests

def test_nominatim(pin_code):
    headers = {"User-Agent": "AQI-App-Test/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?postalcode={pin_code}&country=india&format=json"
    resp = requests.get(url, headers=headers)
    print(f"Testing {pin_code}: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if data:
            print(f"Lat: {data[0]['lat']}, Lon: {data[0]['lon']}, Name: {data[0]['display_name']}")
        else:
            print("Not found")

test_nominatim("201001") # gzb pincode
test_nominatim("226001") # lucknow pincode
test_nominatim("110001") # delhi pincode
