import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import './DrawingCanvas.css';

const DrawingCanvas = forwardRef(({ setDrawing }, ref) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [context, setContext] = useState(null);
  const [hasDrawing, setHasDrawing] = useState(false);
  const [currentSegment, setCurrentSegment] = useState(0);
  const [segments, setSegments] = useState([]);

  const SEGMENT_WIDTH = 150;  // Her kutunun genişliği
  const SEGMENT_HEIGHT = 150; // Her kutunun yüksekliği
  const SEGMENT_MARGIN = 20;  // Kutular arası boşluk
  const MAX_SEGMENTS = 5;     // Kutu sayısı

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Arka planı beyaz yap
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Kutuları çiz
    drawBoxes(ctx);
    
    setContext(ctx);
  }, []);

  const drawBoxes = (ctx) => {
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < MAX_SEGMENTS; i++) {
      const x = i * (SEGMENT_WIDTH + SEGMENT_MARGIN);
      // Kutu çiz
      ctx.beginPath();
      ctx.rect(x, 0, SEGMENT_WIDTH, SEGMENT_HEIGHT);
      ctx.stroke();
    }
    
    // Çizim ayarlarını geri al
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 8;
  };

  const clearCanvas = () => {
    if (context && canvasRef.current) {
      context.fillStyle = 'white';
      context.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      drawBoxes(context);
      setSegments([]);
      setCurrentSegment(0);
      setDrawing(null);
      setHasDrawing(false);
    }
  };

  useImperativeHandle(ref, () => ({
    clear: clearCanvas
  }));

  const getSegmentFromX = (x) => {
    return Math.floor(x / (SEGMENT_WIDTH + SEGMENT_MARGIN));
  };

  const isInCurrentSegment = (x, y) => {
    const segment = getSegmentFromX(x);
    const segmentX = segment * (SEGMENT_WIDTH + SEGMENT_MARGIN);
    return (
      x >= segmentX && 
      x < segmentX + SEGMENT_WIDTH && 
      y >= 0 && 
      y < SEGMENT_HEIGHT
    );
  };

  const startDrawing = (e) => {
    if (!context) return;
    const { offsetX, offsetY } = e.nativeEvent;
    if (!isInCurrentSegment(offsetX, offsetY)) return;
    
    const segment = getSegmentFromX(offsetX);
    setCurrentSegment(segment);
    context.beginPath();
    context.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing || !context) return;
    const { offsetX, offsetY } = e.nativeEvent;
    if (!isInCurrentSegment(offsetX, offsetY)) return;
    
    context.lineTo(offsetX, offsetY);
    context.stroke();
    setHasDrawing(true);
  };

  const stopDrawing = () => {
    if (isDrawing && context && canvasRef.current) {
      context.closePath();
      setIsDrawing(false);
      
      if (!isCanvasEmpty()) {
        // Mevcut bölümü işle
        const segmentImage = processSegment(currentSegment);
        if (segmentImage) {
          const newSegments = [...segments];
          newSegments[currentSegment] = segmentImage;
          setSegments(newSegments);
          setDrawing(newSegments);
        }
      }
    }
  };

  const processSegment = (segmentIndex) => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 28, 28);
    
    tempCtx.fillStyle = 'black';
    tempCtx.strokeStyle = 'black';
    tempCtx.lineWidth = 1;
    
    // İlgili kutuyu kırp
    const x = segmentIndex * (SEGMENT_WIDTH + SEGMENT_MARGIN);
    tempCtx.drawImage(
      canvasRef.current,
      x, 0, SEGMENT_WIDTH, SEGMENT_HEIGHT,
      0, 0, 28, 28
    );
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      const value = avg < 150 ? 0 : 255;
      data[i] = data[i + 1] = data[i + 2] = value;
      data[i + 3] = 255;
    }
    
    tempCtx.putImageData(imageData, 0, 0);
    return tempCanvas.toDataURL('image/png');
  };

  const isCanvasEmpty = () => {
    const canvas = canvasRef.current;
    if (!canvas) return true;
    
    const context = canvas.getContext('2d');
    const imageData = context.getImageData(
      currentSegment * (SEGMENT_WIDTH + SEGMENT_MARGIN),
      0,
      SEGMENT_WIDTH,
      SEGMENT_HEIGHT
    ).data;
    
    for (let i = 0; i < imageData.length; i += 4) {
      const avg = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
      if (avg < 240) return false;
    }
    return true;
  };

  // Tek bir kutuyu temizle
  const clearSegment = (segmentIndex) => {
    if (context && canvasRef.current) {
      // Sadece seçili kutuyu beyaza boya
      const x = segmentIndex * (SEGMENT_WIDTH + SEGMENT_MARGIN);
      context.fillStyle = 'white';
      context.fillRect(x, 0, SEGMENT_WIDTH, SEGMENT_HEIGHT);
      
      // Kutu çerçevesini yeniden çiz
      context.strokeStyle = '#666';
      context.lineWidth = 2;
      context.beginPath();
      context.rect(x, 0, SEGMENT_WIDTH, SEGMENT_HEIGHT);
      context.stroke();
      
      // Çizim ayarlarını geri al
      context.strokeStyle = 'black';
      context.lineWidth = 8;

      // Segments array'inden ilgili segmenti kaldır
      const newSegments = [...segments];
      newSegments[segmentIndex] = null;
      setSegments(newSegments);
      setDrawing(newSegments);
    }
  };

  return (
    <div className="canvas-container">
      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          width={(SEGMENT_WIDTH + SEGMENT_MARGIN) * MAX_SEGMENTS - SEGMENT_MARGIN}
          height={SEGMENT_HEIGHT}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
        />
        <div className="segment-buttons">
          {Array.from({ length: MAX_SEGMENTS }).map((_, index) => (
            <button
              key={index}
              className="clear-segment-button"
              onClick={() => clearSegment(index)}
              style={{
                left: `${index * (SEGMENT_WIDTH + SEGMENT_MARGIN) + SEGMENT_WIDTH/2 - 15}px`
              }}
            >
              ×
            </button>
          ))}
        </div>
      </div>
      <button onClick={clearCanvas} className="clear-all-button">Tümünü Temizle</button>
      {!hasDrawing && <div className="drawing-hint">Lütfen bir sayı veya işlem çizin</div>}
    </div>
  );
});

export default DrawingCanvas; 