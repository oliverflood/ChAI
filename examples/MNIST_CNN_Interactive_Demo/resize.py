from PIL import Image

image_path = 'number.png'

# Open the original image
with Image.open(image_path) as img:
    # Resize the image to 28x28 pixels
    resized_img = img.resize((28, 28))
    
    # Save the resized image back to the same file
    resized_img.save(image_path, format='PNG')