# RedStudio AI Paint Backend API

FastAPI backend for AI-powered virtual wall painting using semantic segmentation.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Server

```bash
# Development mode (auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python main.py
```

### 3. Test API

**Health Check:**
```bash
curl http://localhost:8000/
```

**Process Image:**
```bash
curl -X POST "http://localhost:8000/segment" \
  -F "image=@test_room.jpg" \
  -F "color_r=235" \
  -F "color_g=57" \
  -F "color_b=53"
```

## 📡 API Endpoints

### `GET /`
Health check and API info

### `GET /health`
Detailed health status

### `POST /segment`
Main image processing endpoint

**Parameters:**
- `image` (file): Room/wall image to process
- `color_r` (int): Red value (0-255)
- `color_g` (int): Green value (0-255)
- `color_b` (int): Blue value (0-255)
- `blend_alpha` (float, optional): Blend strength (default: 0.6)

**Response:**
```json
{
  "success": true,
  "processed_image": "base64_encoded_image_string",
  "original_size": [1920, 1080],
  "mask_coverage_percent": 45.2,
  "color_applied": {
    "r": 235,
    "g": 57,
    "b": 53
  }
}
```

## 🧠 AI Model

**Model:** `nvidia/segformer-b0-finetuned-ade-512-512`
- Semantic segmentation on ADE20K dataset
- Lightweight B0 variant for CPU inference
- Segments: wall, building, ceiling, floor, etc.

**Paintable Classes:**
- 0: Wall
- 1: Building
- 5: Ceiling

## 🎨 Processing Pipeline

```
1. Upload Image
   ↓
2. AI Segmentation (SegFormer)
   ↓
3. Create Mask (wall/ceiling areas)
   ↓
4. Realistic Color Blending
   - Weighted overlay (cv2.addWeighted)
   - HSV luminance preservation
   - Texture & shadow retention
   ↓
5. Return Base64 Image
```

## 🔧 Configuration

**For Android Emulator:**
- Use `http://10.0.2.2:8000` in Flutter app

**For Physical Device:**
- Use your PC's local IP: `http://192.168.1.XXX:8000`

**For Production:**
- Deploy to cloud (AWS, GCP, Azure)
- Use HTTPS
- Add authentication

## 📦 Project Structure

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠️ Development

**Hot Reload:**
```bash
uvicorn main:app --reload
```

**Custom Port:**
```bash
uvicorn main:app --port 5000
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🐛 Troubleshooting

**Model Download Issue:**
- First run downloads ~90MB model from Hugging Face
- Requires internet connection
- Model cached in `~/.cache/huggingface/`

**CORS Error:**
- Check CORS middleware in `main.py`
- Update `allow_origins` for production

**Memory Issue:**
- Using B0 (smallest) variant
- For large images, consider resizing
- Or upgrade to GPU server

## 📝 Notes

- CPU inference ~3-5 seconds per image
- GPU can reduce to <1 second
- Realistic blending preserves shadows/texture
- HSV color space for better results

## 🔒 Security (Production)

- [ ] Add API key authentication
- [ ] Rate limiting
- [ ] Input validation (file size, type)
- [ ] HTTPS only
- [ ] Specific CORS origins

## 📄 License

MIT License - RedStudio Project
