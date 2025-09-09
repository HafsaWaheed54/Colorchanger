from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import cv2
import numpy as np
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'ico', 'tif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_image(image, target_width=None, target_height=None, maintain_aspect_ratio=True):
    """
    Resize image to target dimensions while optionally maintaining aspect ratio.
    Accepts a numpy image (H x W x C, BGR or RGB depending on caller).
    """
    logger.info(f"resize_image called with: target_width={target_width}, target_height={target_height}, maintain_aspect_ratio={maintain_aspect_ratio}")

    if target_width is None and target_height is None:
        logger.info("No target dimensions provided, returning original image")
        return image

    height, width = image.shape[:2]
    logger.info(f"Original image dimensions: {width}x{height}")

    if maintain_aspect_ratio:
        if target_width and target_height:
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)
            new_width = int(width * scale)
            new_height = int(height * scale)
            logger.info(f"Aspect ratio preserved: scaled by {scale:.3f} to fit within {target_width}x{target_height}")
        elif target_width:
            scale = target_width / width
            new_width = target_width
            new_height = int(height * scale)
            logger.info(f"Scaled by width: {scale:.3f}")
        elif target_height:
            scale = target_height / height
            new_width = int(width * scale)
            new_height = target_height
            logger.info(f"Scaled by height: {scale:.3f}")
    else:
        logger.info("EXACT DIMENSIONS MODE - aspect ratio will be ignored")
        new_width = target_width if target_width is not None else width
        new_height = target_height if target_height is not None else height
        logger.info(f"Calculated exact dimensions: {new_width}x{new_height} (aspect ratio may be distorted)")

    new_width = max(1, int(new_width))
    new_height = max(1, int(new_height))

    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    actual_height, actual_width = resized.shape[:2]
    logger.info(f"FINAL RESULT: Image resized to {actual_width}x{actual_height}")

    return resized
from PIL import Image
def replace_colors_strict_exact(image_path, from_color_hex, to_color_hex):
    """
    Replace pixels matching exact color in image, accounting for minor rounding differences.
    Returns BGR NumPy array compatible with OpenCV.
    """
    # Convert hex to RGB
    from_rgb = tuple(int(from_color_hex[i:i+2], 16) for i in (1,3,5))
    to_rgb = tuple(int(to_color_hex[i:i+2], 16) for i in (1,3,5))

    # Open image as RGB
    img = np.array(Image.open(image_path).convert("RGB"))

    # Tiny tolerance to catch minor rounding differences
    tolerance = 1
    diff = np.abs(img - from_rgb)
    mask = np.all(diff <= tolerance, axis=2)

    img[mask] = to_rgb

    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

def replace_colors_strict(image_path, from_color_hex, to_color_hex):
    """
    Replace ONLY exact matches of from_color with to_color (no tolerance).
    """
    # Convert hex (#RRGGBB) to RGB tuple
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    from_rgb = hex_to_rgb(from_color_hex)
    to_rgb = hex_to_rgb(to_color_hex)

    # Open image
    img = Image.open(image_path).convert("RGB")
    pixels = img.load()

    # Go through each pixel
    for y in range(img.height):
        for x in range(img.width):
            if pixels[x, y] == from_rgb:   # Strict match
                pixels[x, y] = to_rgb

    return img


def replace_colors(image_path, blue_to_red=True, beige_to_navy=True):
    """
    Simple replacements for some presets. (Kept for backward compatibility if needed.)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HSV thresholds - tune as needed
    blue_lower = np.array([80, 50, 50])
    blue_upper = np.array([130, 255, 255])

    beige_lower = np.array([10, 30, 200])  # note: original values may not detect 'beige' well; this is example
    beige_upper = np.array([40, 120, 255])

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    if blue_to_red:
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        img_rgb[blue_mask > 0] = [255, 0, 0]

    if beige_to_navy:
        beige_mask = cv2.inRange(hsv, beige_lower, beige_upper)
        img_rgb[beige_mask > 0] = [0, 0, 128]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def hex_to_rgb(hex_color):
    """Convert hex color (e.g. '#aabbcc') to tuple (r,g,b)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def replace_colors_custom(image_path, from_color_hex, to_color_hex, tolerance=30):
    """
    Replace pixels whose RGB distance to from_color is <= tolerance with to_color.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    from_rgb = hex_to_rgb(from_color_hex)
    to_rgb = hex_to_rgb(to_color_hex)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img_rgb.copy().astype(np.float32)

    img_float = img_rgb.astype(np.float32)
    from_color = np.array(from_rgb, dtype=np.float32)

    diff = img_float - from_color
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = distance <= tolerance

    result[mask] = to_rgb
    result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr


def replace_colors_advanced(image_path, from_color_hex, to_color_hex, tolerance=30):
    """
    Advanced replacement using RGB distance (primary) and optional HSV/LAB heuristics.
    This implementation uses only RGB distance to avoid over-replacing.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    from_rgb = hex_to_rgb(from_color_hex)
    to_rgb = hex_to_rgb(to_color_hex)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img_rgb.copy().astype(np.float32)

    img_float = img_rgb.astype(np.float32)
    from_color = np.array(from_rgb, dtype=np.float32)
    diff = img_float - from_color
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    rgb_mask = distance <= tolerance

    result[rgb_mask] = to_rgb
    result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr


def replace_colors_precise(image_path, from_color_hex, to_color_hex, tolerance=15):
    """
    Precise RGB-distance based replacement (strict tolerance).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    from_rgb = hex_to_rgb(from_color_hex)
    to_rgb = hex_to_rgb(to_color_hex)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img_rgb.copy().astype(np.float32)

    img_float = img_rgb.astype(np.float32)
    from_color = np.array(from_rgb, dtype=np.float32)
    diff = img_float - from_color
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    mask = distance <= tolerance
    result[mask] = to_rgb

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


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("=== UPLOAD REQUEST STARTED ===")

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

        import uuid
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)

        if not os.path.exists(filepath):
            logger.error(f"File not saved properly: {filepath}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500

        # Read user form fields
        from_color = request.form.get('from_color', '').strip()
        to_color = request.form.get('to_color', '').strip()
        tolerance_raw = request.form.get('tolerance', '20').strip()
        try:
            tolerance = int(tolerance_raw)
        except ValueError:
            tolerance = 20

        output_width_raw = request.form.get('output_width', '').strip()
        output_height_raw = request.form.get('output_height', '').strip()
        maintain_aspect_ratio = request.form.get('maintain_aspect_ratio', 'false').lower() == 'true'
        image_quality = int(request.form.get('image_quality', 95))
        image_format = request.form.get('image_format', 'png')

        output_width = int(output_width_raw) if output_width_raw else None
        output_height = int(output_height_raw) if output_height_raw else None

        # Validate hex colors: accept only full 7-char hex (#rrggbb). If invalid or empty, treat as None.
        if not (from_color and from_color.startswith('#') and len(from_color) == 7):
            from_color = None
        if not (to_color and to_color.startswith('#') and len(to_color) == 7):
            to_color = None

        # Clamp tolerance to reasonable range (0-255)
        tolerance = max(0, min(255, tolerance))

        logger.info(f"Processing image with colors: {from_color} -> {to_color}, tolerance: {tolerance}")
        logger.info(f"Output settings: width={output_width}, height={output_height}, quality={image_quality}, format={image_format}")
        logger.info(f"Maintain aspect ratio: {maintain_aspect_ratio}")

        # Only perform replacement if both colors are provided and valid
        if from_color and to_color:
         processed_img = replace_colors_strict_exact(filepath, from_color, to_color)


        else:
            # Keep original image if colors not provided
            processed_img = cv2.imread(filepath)
            if processed_img is None:
                logger.error("Failed to read saved file for processing")
                return jsonify({'error': 'Failed to read uploaded file for processing'}), 500

        # Resize if requested
        if output_width is not None or output_height is not None:
            processed_img = resize_image(processed_img, output_width, output_height, maintain_aspect_ratio)
        else:
            logger.info("No resizing requested - keeping original dimensions")

        # Save processed image
        output_filename = f"processed_{filename.rsplit('.', 1)[0]}.{image_format}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        if image_format.lower() in ['jpg', 'jpeg']:
            success = cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
        elif image_format.lower() == 'png':
            compression_level = int((100 - image_quality) / 10)
            success = cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        else:
            success = cv2.imwrite(output_path, processed_img)

        if not success:
            logger.error(f"Failed to save processed image: {output_path}")
            return jsonify({'error': 'Failed to save processed image'}), 500

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
