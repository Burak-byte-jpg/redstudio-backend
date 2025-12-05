"""
Test Script for RedStudio AI Paint API
=======================================

Simple Python script to test the segmentation endpoint.
"""

import requests
import base64
from pathlib import Path

# API configuration
API_URL = "http://localhost:8000/segment"

# Test image path (replace with your test image)
IMAGE_PATH = "test_room.jpg"

# Target color (RGB)
COLOR_R = 235  # Red
COLOR_G = 57   # Green  
COLOR_B = 53   # Blue

def test_api():
    """Test the segmentation API"""
    
    print("🔍 Testing RedStudio AI Paint API...\n")
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        print("   Please provide a test image (room/wall photo)")
        return
    
    print(f"📷 Image: {IMAGE_PATH}")
    print(f"🎨 Color: RGB({COLOR_R}, {COLOR_G}, {COLOR_B})\n")
    
    try:
        # Prepare request
        with open(IMAGE_PATH, 'rb') as f:
            files = {'image': f}
            data = {
                'color_r': COLOR_R,
                'color_g': COLOR_G,
                'color_b': COLOR_B,
                'blend_alpha': 0.6
            }
            
            print("📤 Sending request to API...")
            response = requests.post(API_URL, files=files, data=data)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Success!\n")
            print(f"   Original Size: {result['original_size']}")
            print(f"   Mask Coverage: {result['mask_coverage_percent']:.2f}%")
            print(f"   Color Applied: RGB{tuple(result['color_applied'].values())}\n")
            
            # Save result
            base64_image = result['processed_image']
            output_path = "output_painted.jpg"
            
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(base64_image))
            
            print(f"💾 Processed image saved to: {output_path}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
    
    except requests.ConnectionError:
        print("❌ Connection Error!")
        print("   Make sure the API server is running:")
        print("   uvicorn main:app --reload")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_api()
