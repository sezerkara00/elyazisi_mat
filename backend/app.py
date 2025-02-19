import os
os.environ['TENSORFLOW_CPU_ONLY'] = '1'

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import cv2
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.serving import run_simple
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# CORS ayarlarını güncelle
CORS(app, resources={
    r"/*": {  # Tüm endpoint'ler için CORS
        "origins": [
            "https://elyazisi-mat.vercel.app",
            "http://localhost:3000",
            "*"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Rate limiting ekle
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Model oluşturma
def create_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        # 14 sınıf: 0-9 rakamları ve +,-,×,÷ sembolleri
        layers.Dense(14, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def load_dataset(data_path):
    images = []
    labels = []
    
    # Rakamlar için (0-9)
    for digit in range(10):
        digit_path = os.path.join(data_path, str(digit))
        if os.path.exists(digit_path):
            for img_name in os.listdir(digit_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(digit_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (28, 28))
                        images.append(img)
                        labels.append(digit)
    
    # İşlem sembolleri için (10: +, 11: -, 12: ×, 13: ÷)
    symbols = {
        'add': 10,
        'subtract': 11,
        'multiply': 12,
        'divide': 13
    }
    
    for symbol, label in symbols.items():
        symbol_path = os.path.join(data_path, symbol)
        if os.path.exists(symbol_path):
            for img_name in os.listdir(symbol_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(symbol_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (28, 28))
                        images.append(img)
                        labels.append(label)
    
    return np.array(images), np.array(labels)

def train_model(model, data_path):
    print("Model eğitimi başlıyor...")
    
    images = []
    labels = []
    
    if not os.path.exists(data_path):
        raise Exception(f"Veri seti klasörü bulunamadı: {data_path}")
    print(f"Veri seti klasörü: {data_path}")
    
    # Rakam isimlerini tanımla
    digit_names = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine'
    }
    
    # Rakamları yükle (0-9)
    for digit in range(10):
        digit_path = os.path.join(data_path, digit_names[digit])  # Rakam isimlerini kullan
        if os.path.exists(digit_path):
            print(f"Rakam {digit} klasörü işleniyor: {digit_path}")
            file_count = 0
            
            for img_name in os.listdir(digit_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(digit_path, img_name)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"Görüntü okunamadı: {img_path}")
                            continue
                            
                        img = cv2.resize(img, (28, 28))
                        img = img.reshape(28, 28, 1)
                        img = img / 255.0
                        
                        repeat = 3 if digit in [8, 9] else 1
                        for _ in range(repeat):
                            images.append(img)
                            labels.append(digit)
                            file_count += 1
                            
                    except Exception as e:
                        print(f"Hata ({img_path}): {str(e)}")
                        continue
            print(f"Rakam {digit} için {file_count} görüntü yüklendi")
        else:
            print(f"Uyarı: {digit_names[digit]} klasörü bulunamadı: {digit_path}")
    
    # İşlem sembollerini yükle
    symbols = {
        'add': 10,      # + işareti
        'subtract': 11,  # - işareti
        'multiply': 12,  # × işareti
        'divide': 13    # ÷ işareti
    }
    
    for symbol_name, label in symbols.items():
        symbol_path = os.path.join(data_path, symbol_name)
        if os.path.exists(symbol_path):
            print(f"{symbol_name} sembolü klasörü işleniyor...")
            file_count = 0
            
            for img_name in os.listdir(symbol_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(symbol_path, img_name)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"Görüntü okunamadı: {img_path}")
                            continue
                            
                        img = cv2.resize(img, (28, 28))
                        img = img.reshape(28, 28, 1)
                        img = img / 255.0
                        
                        # Sembolleri tekrarlamayı kaldırdık
                        images.append(img)
                        labels.append(label)
                        file_count += 1
                            
                    except Exception as e:
                        print(f"Hata ({img_path}): {str(e)}")
                        continue
            print(f"{symbol_name} sembolü için {file_count} görüntü yüklendi")
        else:
            print(f"Uyarı: {symbol_name} klasörü bulunamadı: {symbol_path}")
    
    # Veri kontrolü
    if len(images) == 0:
        raise Exception("Hiç görüntü yüklenemedi!")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nToplam {len(X)} görüntü yüklendi")
    print(f"Veri şekli: {X.shape}")
    print("\nEtiket dağılımı:")
    
    # Rakamlar için dağılım
    for digit in range(10):
        count = np.sum(y == digit)
        print(f"Rakam {digit}: {count}")
    
    # Semboller için dağılım
    for symbol_name, label in symbols.items():
        count = np.sum(y == label)
        print(f"{symbol_name}: {count}")
    
    # Veriyi karıştır ve eğit
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    split = int(0.8 * len(X))
    train_images = X[:split]
    train_labels = y[:split]
    val_images = X[split:]
    val_labels = y[split:]
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=(val_images, val_labels),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    print("Model eğitimi tamamlandı!")
    
    if not os.path.exists('./model'):
        os.makedirs('./model')
    model.save('./model/digits_model.keras')
    print("Model kaydedildi!")
    
    return history

def preprocess_image(base64_data):
    try:
        # Base64'ten görüntüyü al
        image_data = base64.b64decode(base64_data.split(',')[1])
        img = Image.open(io.BytesIO(image_data))
        
        # Gri tonlamaya çevir
        img = img.convert('L')
        
        # 28x28 boyutuna yeniden boyutlandır
        img = img.resize((28, 28))
        
        # NumPy dizisine dönüştür ve normalize et
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1)
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        print("Görüntü işleme hatası:", str(e))
        raise e

# Global değişkenler
model = None

def load_model_once():
    global model
    if model is None:
        print("Model yükleniyor...")
        try:
            model = keras.models.load_model('./model/digits_model.keras')
        except:
            print("Model bulunamadı, yeni model oluşturuluyor")
            model = create_model()
            train_model(model, './digits')
    return model

@app.route('/', methods=['GET'])
def home():
    logger.info("Root endpoint called")
    return jsonify({
        'status': 'online',
        'message': 'Matematik El Yazısı Tanıma API',
        'endpoints': {
            'predict': '/predict (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called")
    return jsonify({
        'status': 'healthy'
    })

@app.route('/predict', methods=['POST'])
@limiter.limit("1 per second")  # Her kullanıcı için saniyede 1 istek
def predict():
    try:
        model = load_model_once()
        data = request.json
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Görüntü verisi bulunamadı'
            })
            
        processed_image = preprocess_image(data['image'])
        
        with tf.device('/CPU:0'):
            prediction = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Sınıf etiketlerini sembollere dönüştür
        symbols = {
            10: '+',
            11: '-',
            12: '×',
            13: '÷'
        }
        
        result = symbols.get(predicted_class, str(predicted_class))
        
        print(f"Tahmin: {result}, Güven: {confidence:.4f}")
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    logger.info("Application starting...")
    load_model_once()
    
    if os.environ.get('RENDER'):
        # Render.com'da Gunicorn kullanılacak
        logger.info("Running on Render.com")
        app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
    else:
        # Lokal geliştirme için
        logger.info("Running locally")
        port = int(os.environ.get("PORT", 10000))
        run_simple('0.0.0.0', port, app, use_reloader=True) 