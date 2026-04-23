import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import cv2

class ProductRecognitionSystem:
    def __init__(self, data_dir='data/raw', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.img_size = (224, 224)
        self.batch_size = 32
        self.model = None
        self.features = []
        self.labels = []
        self.le = None
        self.model_path = os.path.join(self.model_dir, 'svm_model.pkl')
        self.encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')

        # Tạo thư mục nếu chưa có
        os.makedirs(model_dir, exist_ok=True)

    def prepare_data(self):
        """Chuẩn bị dữ liệu training"""
        print("Chuẩn bị dữ liệu...")

        # Giả sử có các thư mục sản phẩm trong data/raw
        categories = [d for d in os.listdir(self.data_dir)
                     if os.path.isdir(os.path.join(self.data_dir, d))]

        if not categories:
            print("Không tìm thấy thư mục sản phẩm. Tạo dữ liệu mẫu...")
            self.create_sample_data()
            categories = [d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d))]

        print(f"Tìm thấy {len(categories)} loại sản phẩm: {categories}")

        # Load images and extract features
        hog = cv2.HOGDescriptor()
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                features = hog.compute(gray)
                self.features.append(features.flatten())
                self.labels.append(category)

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.le = LabelEncoder()
        self.labels_encoded = self.le.fit_transform(self.labels)
        self.num_classes = len(np.unique(self.labels_encoded))

        print(f"Đã load {len(self.features)} ảnh với {self.num_classes} lớp.")
        return categories

    def create_sample_data(self):
        """Tạo dữ liệu mẫu nếu không có dữ liệu thật"""
        categories = ['sua', 'bia', 'nuoc_ngot', 'banh_keo']

        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            os.makedirs(category_path, exist_ok=True)

            # Tạo ảnh mẫu (màu đơn giản)
            for i in range(10):  # 10 ảnh mỗi loại
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(f'{category_path}/{category}_{i}.jpg', img)

        print("Đã tạo dữ liệu mẫu!")

    def build_model(self):
        """Xây dựng model với SVM"""
        print("Xây dựng model...")

        self.model = SVC(kernel='linear', probability=True)

        print("Model đã được xây dựng!")

    def train(self, epochs=10):
        """Train model"""
        print("Bắt đầu training...")

        self.model.fit(self.features, self.labels_encoded)
        self.save_model()

        print("Training hoàn thành!")
        return None

    def save_model(self):
        """Lưu model và LabelEncoder"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.le, f)
        print(f"Đã lưu model vào {self.model_path}")

    def predict(self, image_path):
        """Dự đoán sản phẩm từ ảnh"""
        if self.model is None:
            self.load_model()
            if self.model is None:
                return "Model chưa sẵn sàng. Vui lòng train lại."

        # Load và preprocess ảnh
        img = cv2.imread(image_path)
        if img is None:
            return "Không thể đọc ảnh"

        img = cv2.resize(img, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray).flatten()
        features = features.reshape(1, -1)

        # Predict
        predictions = self.model.predict_proba(features)[0]
        class_idx = np.argmax(predictions)

        # Lấy tên class
        class_labels = self.le.classes_
        confidence = predictions[class_idx] * 100

        return {
            'product': class_labels[class_idx],
            'confidence': f"{confidence:.2f}%"
        }

    def load_model(self, model_path=None):
        """Load model đã train"""
        if model_path is None:
            model_path = self.model_path

        if os.path.exists(model_path) and os.path.exists(self.encoder_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.encoder_path, 'rb') as f:
                self.le = pickle.load(f)
            print(f"Đã load model từ {model_path}")
        else:
            print("Không tìm thấy model hoặc LabelEncoder đã lưu!")
if __name__ == "__main__":
    # Khởi tạo hệ thống
    system = ProductRecognitionSystem()

    # Chuẩn bị dữ liệu
    categories = system.prepare_data()

    # Xây dựng model
    system.build_model()

    # Training
    history = system.train()  # Train với SVM

    print("Hệ thống đã sẵn sàng!")
