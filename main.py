"""
RedStudio AI Paint Segmentation API
====================================

FastAPI backend for AI-powered wall/ceiling painting visualization.
Uses Hugging Face SegFormer model for semantic segmentation.

Run Server:
-----------
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Test:
-----
curl -X POST "http://localhost:8000/segment" \
  -F "image=@test_room.jpg" \
  -F "color_r=235" \
  -F "color_g=57" \
  -F "color_b=53"

Requirements:
-------------
pip install fastapi uvicorn python-multipart pillow numpy opencv-python transformers torch
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import io
import base64
from typing import Optional
import logging

# Hugging Face Transformers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RedStudio AI Paint API",
    description="Semantic segmentation API for virtual wall painting",
    version="1.0.0"
)

# CORS middleware (allow Flutter app to access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loading (loaded once at startup)
model = None
processor = None


@app.on_event("startup")
async def load_model():
    """Load AI model on server startup"""
    global model, processor
    
    logger.info("Loading SegFormer model from Hugging Face...")
    
    try:
        # Load pretrained model and processor
        # Using lightweight B0 variant for faster CPU inference
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("✓ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "RedStudio AI Paint API",
        "version": "1.0.0",
        "model": "nvidia/segformer-b0-finetuned-ade-512-512",
        "endpoints": {
            "segment": "POST /segment (image + color_r + color_g + color_b)"
        }
    }


@app.post("/segment")
async def segment_and_paint(
    image: UploadFile = File(..., description="Room/wall image to process"),
    color_r: int = Form(..., description="Red component (0-255)", ge=0, le=255),
    color_g: int = Form(..., description="Green component (0-255)", ge=0, le=255),
    color_b: int = Form(..., description="Blue component (0-255)", ge=0, le=255),
    blend_alpha: Optional[float] = Form(0.6, description="Blend strength (0.0-1.0)", ge=0.0, le=1.0)
):
    """
    AI-powered wall/ceiling painting endpoint
    
    Args:
        image: Uploaded image file
        color_r: Red color value (0-255)
        color_g: Green color value (0-255)
        color_b: Blue color value (0-255)
        blend_alpha: Blending strength for realistic overlay (default 0.6)
    
    Returns:
        JSON with base64 encoded processed image
    """
    
    try:
        logger.info(f"Processing image: {image.filename}")
        logger.info(f"Target color: RGB({color_r}, {color_g}, {color_b})")
        
        # ===== 1. Load and preprocess image =====
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Store original size for later
        original_size = pil_image.size
        logger.info(f"Original image size: {original_size}")
        
        # Convert to numpy array for OpenCV
        original_array = np.array(pil_image)
        
        # ===== 2. Run AI Segmentation =====
        logger.info("Running semantic segmentation...")
        
        # Preprocess image for model
        inputs = processor(images=pil_image, return_tensors="pt")
        
        # Run inference (no gradient calculation for faster inference)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Upscale logits to original image size
        logits_upscaled = torch.nn.functional.interpolate(
            logits,
            size=original_size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )
        
        # Get predicted class for each pixel
        predicted_seg = logits_upscaled.argmax(dim=1).squeeze().cpu().numpy()
        
        logger.info(f"Segmentation shape: {predicted_seg.shape}")
        
        # ===== 3. Create mask for paintable surfaces =====
        # ADE20K dataset class IDs (from SegFormer training):
        # 0: wall, 1: building, 2: sky, 3: floor, 4: tree, 5: ceiling, etc.
        # We want to paint: wall (0), building (1), ceiling (5)
        
        paintable_classes = [0, 1, 5]  # wall, building, ceiling
        mask = np.isin(predicted_seg, paintable_classes).astype(np.uint8) * 255
        
        logger.info(f"Mask coverage: {(mask > 0).sum() / mask.size * 100:.2f}%")
        
        # Optional: Apply morphological operations for smoother edges
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # ===== 4. Apply realistic color overlay =====
        logger.info("Applying color with realistic blending...")
        
        # Create color overlay (BGR format for OpenCV)
        color_overlay = np.full_like(original_array, (color_b, color_g, color_r), dtype=np.uint8)
        
        # Convert mask to 3-channel for blending
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Method 1: Weighted blending (preserves texture and shadows)
        # This creates a realistic "paint overlay" effect
        blended = cv2.addWeighted(
            original_array.astype(np.float32),
            1 - blend_alpha,
            color_overlay.astype(np.float32),
            blend_alpha,
            0
        )
        
        # Apply mask to blend only painted areas
        result = (original_array * (1 - mask_3channel) + blended * mask_3channel).astype(np.uint8)
        
        # Optional: Enhance realism with HSV adjustment
        # Convert to HSV to preserve luminance (brightness/shadows)
        hsv_original = cv2.cvtColor(original_array, cv2.COLOR_RGB2HSV)
        hsv_result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        
        # Preserve original Value (brightness) channel for realistic shadows
        hsv_result[:, :, 2] = hsv_original[:, :, 2]
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2RGB)
        
        # ===== 5. Encode result as Base64 =====
        logger.info("Encoding result as Base64...")
        
        # Convert to PIL Image
        result_pil = Image.fromarray(result)
        
        # Encode to JPEG bytes
        buffer = io.BytesIO()
        result_pil.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        
        # Convert to Base64
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        
        logger.info("✓ Processing complete!")
        
        return JSONResponse(content={
            "success": True,
            "processed_image": base64_image,
            "original_size": list(original_size),
            "mask_coverage_percent": float((mask > 0).sum() / mask.size * 100),
            "color_applied": {
                "r": color_r,
                "g": color_g,
                "b": color_b
            }
        })
        
    except Exception as e:
        logger.error(f"✗ Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "device": "CPU" if not torch.cuda.is_available() else "CUDA"
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render uses PORT)
    port = int(os.environ.get("PORT", 8000))
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
