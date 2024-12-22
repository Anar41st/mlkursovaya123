import os
import requests
import base64
from PIL import Image
import io

image_dir = 'C:/Users/nrbch/mlkursects/pythonProject2/image'

api_url = 'http://127.0.0.1:5000'

def test_api_with_file(image_path):
    with open(image_path, 'rb') as image_file:
        files = {'file': (image_file.name, image_file, 'image/jpeg')}
        response = requests.post(f'{api_url}/api/detect', files=files)
    if response.status_code == 200:
        print(f'Результаты детекции для {image_path}:')
        results = response.json()
        for result in results:
            print(f'{result["class_name"]}: {result["confidence"]}')

def test_api_with_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    data = {'image': f'data:image/jpeg;base64,{base64_data}'}
    response = requests.post(f'{api_url}/api/detect_base64', json=data)
    if response.status_code == 200:
        print(f'Результаты детекции для {image_path}:')
        results = response.json()
        for result in results:
            print(f'{result["class_name"]}: {result["confidence"]}')

if __name__ == '__main__':
    for image_path in [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]:
        test_api_with_file(image_path)

    for image_path in [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]:
        test_api_with_base64(image_path)