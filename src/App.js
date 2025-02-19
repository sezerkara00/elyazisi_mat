import React, { useState, useRef } from 'react';
import './App.css';
import DrawingCanvas from './components/DrawingCanvas';

// API URL'ini environment variable olarak tanımla
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [drawing, setDrawing] = useState(null);
  const [result, setResult] = useState(null);
  const [calculatedResult, setCalculatedResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  const calculateExpression = (expression) => {
    try {
      // İşlem sembollerini JavaScript operatörlerine dönüştür
      const jsExpression = expression
        .replace(/×/g, '*')  // × işaretini * ile değiştir
        .replace(/÷/g, '/'); // ÷ işaretini / ile değiştir
      
      // eval kullanmak yerine daha güvenli bir hesaplama yöntemi
      const calculate = new Function('return ' + jsExpression);
      const result = calculate();
      
      // Sonuç geçerli bir sayı mı kontrol et
      if (!isNaN(result) && isFinite(result)) {
        return result;
      }
      return null;
    } catch (error) {
      console.log('Hesaplama hatası:', error);
      return null;
    }
  };

  const handlePredict = async () => {
    if (!drawing) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setCalculatedResult(null);

    try {
      // Her bölüm için tahmin yap
      const predictions = [];
      for (let segment of drawing) {
        if (segment) {
          try {
            const response = await fetch(`${API_URL}/predict`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ image: segment })
            });

            const data = await response.json();
            if (data.success) {
              predictions.push(data.prediction);
            } else {
              throw new Error(data.error || 'Tahmin başarısız');
            }
          } catch (err) {
            console.error('Segment tahmin hatası:', err);
            setError(`Tahmin hatası: ${err.message}`);
            break;
          }
        }
      }

      if (predictions.length > 0) {
        // Tahminleri birleştir
        const expression = predictions.join('');
        setResult(expression);

        // Matematiksel işlemi hesapla
        const calculated = calculateExpression(expression);
        if (calculated !== null) {
          setCalculatedResult(calculated);
        }
      }

    } catch (err) {
      console.error('Genel hata:', err);
      setError('Tahmin sırasında bir hata oluştu: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Matematik El Yazısı Tanıma</h1>
      <div className="container">
        <DrawingCanvas 
          setDrawing={setDrawing} 
          ref={canvasRef}
        />
        <div className="controls">
          <button 
            onClick={handlePredict} 
            disabled={!drawing || isLoading}
            className={`predict-button ${isLoading ? 'loading' : ''} ${!drawing ? 'disabled' : ''}`}
          >
            {isLoading ? 'Tahmin ediliyor...' : drawing ? 'Tahmin Et' : 'Önce bir rakam çizin'}
          </button>
        </div>
        <div className="result-section">
          {error ? (
            <div className="error-message">{error}</div>
          ) : (
            <>
              <h2>İfade: {result || 'Henüz tahmin yapılmadı'}</h2>
              {calculatedResult !== null && (
                <h2>Sonuç: {calculatedResult}</h2>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
