from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# For Vercel deployment, use temp directories
if os.environ.get('VERCEL'):
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
    app.config['OUTPUT_FOLDER'] = tempfile.gettempdir()
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['OUTPUT_FOLDER'] = 'outputs'
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# No file size limit - support any size image
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Removed size restriction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'ico', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image, target_width=None, target_height=None, maintain_aspect_ratio=True):
    """
    Resize image to target dimensions while optionally maintaining aspect ratio
    """
    if target_width is None and target_height is None:
        return image
    
    height, width = image.shape[:2]
    
    if maintain_aspect_ratio:
        if target_width and target_height:
            # Calculate scale factor to fit within both dimensions (preserve aspect ratio)
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
            new_width = int(width * scale)
            new_height = int(height * scale)
            logger.info(f"Aspect ratio preserved: scaled by {scale:.3f} to fit within {target_width}x{target_height}")
        elif target_width:
            # Scale based on width
            scale = target_width / width
            new_width = target_width
            new_height = int(height * scale)
            logger.info(f"Scaled by width: {scale:.3f}")
        elif target_height:
            # Scale based on height
            scale = target_height / height
            new_width = int(width * scale)
            new_height = target_height
            logger.info(f"Scaled by height: {scale:.3f}")
    else:
        # Use exact dimensions (may distort image)
        new_width = target_width if target_width is not None else width
        new_height = target_height if target_height is not None else height
        logger.info(f"Using exact dimensions: {new_width}x{new_height} (aspect ratio may be distorted)")
    
    # Ensure minimum size of 1 pixel
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
    
    # Resize using OpenCV
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return resized


def replace_colors(image_path, blue_to_red=True, beige_to_navy=True):
    """
    Replace colors in the image:
    - Blue to Red (if blue_to_red is True)
    - Beige to Navy Blue (if beige_to_navy is True)
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    # Convert BGR to RGB for easier color manipulation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define color ranges for replacement
    # Blue color range (adjust these values based on your specific image)
    blue_lower = np.array([80, 50, 50])   # Lower bound for blue
    blue_upper = np.array([130, 255, 255]) # Upper bound for blue
    
    # Beige/tan color range
    beige_lower = np.array([180, 160, 120])  # Lower bound for beige
    beige_upper = np.array([220, 200, 180])  # Upper bound for beige
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Create masks for color replacement
    if blue_to_red:
        # Create mask for blue colors
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        # Replace blue with red
        img_rgb[blue_mask > 0] = [255, 0, 0]  # Red in RGB
    
    if beige_to_navy:
        # Create mask for beige colors
        beige_mask = cv2.inRange(hsv, beige_lower, beige_upper)
        # Replace beige with navy blue
        img_rgb[beige_mask > 0] = [0, 0, 128]  # Navy blue in RGB
    
    # Convert back to BGR for saving
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def replace_colors_custom(image_path, from_color_hex, to_color_hex, tolerance=30):
    """
    Custom color replacement using user-defined colors
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    # Convert hex colors to RGB
    from_rgb = hex_to_rgb(from_color_hex)
    to_rgb = hex_to_rgb(to_color_hex)
    
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a copy for modifications
    result = img_rgb.copy()
    
    # Calculate color distance for each pixel
    # Convert to float for better precision
    img_float = img_rgb.astype(np.float32)
    from_color = np.array(from_rgb, dtype=np.float32)
    
    # Calculate Euclidean distance in RGB space
    diff = img_float - from_color
    distance = np.sqrt(np.sum(diff**2, axis=2))
    
    # Create mask based on tolerance
    mask = distance <= tolerance
    
    # Apply color replacement
    result[mask] = to_rgb
    
    # Convert back to BGR
    result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return result_bgr

def replace_colors_advanced(image_path, from_color_hex, to_color_hex, tolerance=30):
    """
    Advanced color replacement using multiple color space conversions
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    # Convert hex colors to RGB
    from_rgb = hex_to_rgb(from_color_hex)
    to_rgb = hex_to_rgb(to_color_hex)
    
    # Convert to different color spaces for better color detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Create a copy for modifications
    result = img_rgb.copy()
    
    # Method 1: RGB distance-based replacement
    img_float = img_rgb.astype(np.float32)
    from_color = np.array(from_rgb, dtype=np.float32)
    diff = img_float - from_color
    distance = np.sqrt(np.sum(diff**2, axis=2))
    rgb_mask = distance <= tolerance
    
    # Method 2: HSV-based replacement for better color matching
    from_hsv = cv2.cvtColor(np.uint8([[from_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Create HSV range based on tolerance
    h_tolerance = int(tolerance * 0.5)  # Hue tolerance
    s_tolerance = int(tolerance * 1.5)  # Saturation tolerance
    v_tolerance = int(tolerance * 1.5)  # Value tolerance
    
    lower_hsv = np.array([
        max(0, from_hsv[0] - h_tolerance),
        max(0, from_hsv[1] - s_tolerance),
        max(0, from_hsv[2] - v_tolerance)
    ])
    upper_hsv = np.array([
        min(179, from_hsv[0] + h_tolerance),
        min(255, from_hsv[1] + s_tolerance),
        min(255, from_hsv[2] + v_tolerance)
    ])
    
    hsv_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
    # Method 3: LAB color space for better perceptual matching
    from_lab = cv2.cvtColor(np.uint8([[from_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    
    lab_tolerance = int(tolerance * 1.2)
    lower_lab = np.array([
        max(0, from_lab[0] - lab_tolerance),
        max(0, from_lab[1] - lab_tolerance),
        max(0, from_lab[2] - lab_tolerance)
    ])
    upper_lab = np.array([
        min(255, from_lab[0] + lab_tolerance),
        min(255, from_lab[1] + lab_tolerance),
        min(255, from_lab[2] + lab_tolerance)
    ])
    
    lab_mask = cv2.inRange(img_lab, lower_lab, upper_lab)
    
    # Combine all masks
    combined_mask = rgb_mask | (hsv_mask > 0) | (lab_mask > 0)
    
    # Apply color replacement
    result[combined_mask] = to_rgb
    
    # Convert back to BGR
    result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return result_bgr

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return f"Error loading page: {str(e)}", 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Design Color Variant Generator is running'})

@app.route('/test')
def test():
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint working',
        'environment': 'Vercel' if os.environ.get('VERCEL') else 'Local',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'output_folder': app.config['OUTPUT_FOLDER']
    })

@app.route('/test-resize')
def test_resize():
    """Test endpoint to verify resize function works"""
    import numpy as np
    
    # Create a test image (100x100)
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
    
    # Test resize to 200x150
    resized = resize_image(test_img, 200, 150, False)
    
    return jsonify({
        'status': 'success',
        'original_size': f"{test_img.shape[1]}x{test_img.shape[0]}",
        'resized_size': f"{resized.shape[1]}x{resized.shape[0]}",
        'resize_working': resized.shape == (150, 200, 3)
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("=== UPLOAD REQUEST STARTED ===")
        logger.info("Upload request received")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No filename provided")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, or TIFF files.'}), 400
        
        # Generate unique filename to avoid conflicts
        import uuid
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Verify file was saved and is readable
        if not os.path.exists(filepath):
            logger.error(f"File not saved properly: {filepath}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Get color replacement options
        from_color = request.form.get('from_color', '#1a237e')
        to_color = request.form.get('to_color', '#d32f2f')
        tolerance = int(request.form.get('tolerance', 30))
        
        # Get output settings
        output_width_raw = request.form.get('output_width', '').strip()
        output_height_raw = request.form.get('output_height', '').strip()
        maintain_aspect_ratio = request.form.get('maintain_aspect_ratio', 'false').lower() == 'true'
        image_quality = int(request.form.get('image_quality', 95))
        image_format = request.form.get('image_format', 'png')
        
        logger.info(f"Raw form data - width: '{output_width_raw}', height: '{output_height_raw}', maintain_aspect: {maintain_aspect_ratio}")
        
        # Convert to integers if provided and not empty
        if output_width_raw and output_width_raw != '':
            output_width = int(output_width_raw)
        else:
            output_width = None
            
        if output_height_raw and output_height_raw != '':
            output_height = int(output_height_raw)
        else:
            output_height = None
        
        # Validate hex colors
        if not from_color.startswith('#') or len(from_color) != 7:
            from_color = '#1a237e'
        if not to_color.startswith('#') or len(to_color) != 7:
            to_color = '#d32f2f'
        
        # Clamp tolerance between 10 and 100
        tolerance = max(10, min(100, tolerance))
        
        logger.info(f"Processing image with colors: {from_color} -> {to_color}, tolerance: {tolerance}")
        logger.info(f"Output settings: width={output_width}, height={output_height}, quality={image_quality}, format={image_format}")
        logger.info(f"Maintain aspect ratio: {maintain_aspect_ratio}")
        
        # Process the image
        processed_img = replace_colors_advanced(filepath, from_color, to_color, tolerance)
        
        # Resize image if dimensions are specified
        if output_width is not None or output_height is not None:
            logger.info(f"Resizing image: width={output_width}, height={output_height}, maintain_aspect={maintain_aspect_ratio}")
            processed_img = resize_image(processed_img, output_width, output_height, maintain_aspect_ratio)
        else:
            logger.info("No resizing requested - keeping original dimensions")
        
        # Save processed image with quality settings
        output_filename = f"processed_{filename.rsplit('.', 1)[0]}.{image_format}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Set compression parameters based on format
        if image_format.lower() in ['jpg', 'jpeg']:
            success = cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
        elif image_format.lower() == 'png':
            # PNG compression level (0-9, where 9 is highest compression)
            compression_level = int((100 - image_quality) / 10)
            success = cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        else:
            # For other formats (BMP, TIFF), save without compression
            success = cv2.imwrite(output_path, processed_img)
        
        if not success:
            logger.error(f"Failed to save processed image: {output_path}")
            return jsonify({'error': 'Failed to save processed image'}), 500
        
        # Convert to base64 for display
        _, buffer = cv2.imencode('.png', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("Image processed successfully")
        
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'image_data': img_base64,
            'download_url': url_for('download_file', filename=output_filename)
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"Download file not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/demo')
def demo():
    return render_template('demo.html')

# Global error handler to ensure JSON responses
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# Removed 413 error handler since we support any file size

# For Vercel deployment
app = app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
