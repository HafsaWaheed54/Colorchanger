# Design Color Variant Generator

A powerful Flask web application that uses AI-powered image processing to generate color variants of your designs. Transform blue colors to red and beige colors to navy blue with just a few clicks!

## Features

- 🎨 **AI-Powered Color Replacement**: Advanced algorithms detect and replace colors while preserving image quality
- 🔄 **Multiple Color Options**: Change blue to red, beige to navy blue, or customize your own color transformations
- 📱 **Modern UI**: Beautiful, responsive interface that works on all devices
- ⚡ **Fast Processing**: Process images in seconds with optimized algorithms
- 💾 **Instant Download**: Download your transformed designs immediately
- 🖼️ **Multiple Formats**: Supports PNG, JPG, JPEG, GIF, BMP, and TIFF files

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload Your Image**: Drag and drop or click to browse for your design image
2. **Choose Color Options**: Select which colors to replace (blue to red, beige to navy blue)
3. **Generate Variant**: Click the "Generate Color Variant" button
4. **Download Result**: Download your transformed design instantly

## Supported File Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- TIFF

## Technical Details

### Color Detection Algorithm

The application uses multiple color space conversions for accurate color detection:

- **HSV Color Space**: For better color range detection
- **RGB Color Space**: For precise color matching
- **LAB Color Space**: For improved beige/tan color detection

### Color Replacement

- **Blue to Red**: Detects blue colors in the range of HSV(100-130, 50-255, 50-255) and RGB(0-50, 0-50, 80-150)
- **Beige to Navy Blue**: Detects beige colors in multiple color spaces and replaces with navy blue

### File Structure

```
AIProject/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── create_sample_image.py # Sample image generator
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   └── demo.html
├── static/              # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── uploads/             # Uploaded images (auto-created)
└── outputs/             # Processed images (auto-created)
```

## API Endpoints

- `GET /` - Home page with upload interface
- `POST /upload` - Upload and process image
- `GET /download/<filename>` - Download processed image
- `GET /demo` - Demo page with examples

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Performance

- Maximum file size: 16MB
- Processing time: 1-5 seconds (depending on image size)
- Supported image dimensions: Up to 4K resolution

## Troubleshooting

### Common Issues

1. **"Could not read the image" error**:
   - Ensure the file is a valid image format
   - Check file size (must be under 16MB)
   - Try converting to PNG format

2. **Colors not being replaced**:
   - The algorithm works best with distinct color differences
   - Try adjusting the original image contrast
   - Ensure colors are within the detection ranges

3. **Application won't start**:
   - Check that all dependencies are installed: `pip install -r requirements.txt`
   - Ensure port 5000 is not in use by another application
   - Try running with: `python app.py`

### Getting Help

If you encounter any issues:

1. Check the browser console for JavaScript errors
2. Verify all dependencies are installed correctly
3. Ensure your image file is in a supported format
4. Try with the sample images provided in the `uploads/` folder

## Sample Images

The application includes sample images for testing:
- `uploads/sample_pattern.png` - Geometric pattern example
- `uploads/test_image.png` - Simple test image

## Future Enhancements

- Custom color picker for user-defined color replacements
- Batch processing for multiple images
- More color space options
- Advanced image filters and effects
- API integration for external services

## License

This project is open source and available under the MIT License.

---

**Ready to transform your designs?** Start the application and upload your first image!
