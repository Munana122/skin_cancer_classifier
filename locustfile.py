import io
from PIL import Image
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        """Generate a lightweight valid 224x224 JPEG image in RAM for all request loads."""
        img = Image.new('RGB', (224, 224), color=(128, 64, 32))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        self.image_data = img_bytes.getvalue()

    @task
    def test_predict(self):
        files = {'file': ('test.jpg', self.image_data, 'image/jpeg')}
        self.client.post("/predict", files=files)