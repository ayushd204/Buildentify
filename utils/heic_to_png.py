import os
from heic2png import HEIC2PNG

def convert_heic_to_png(images_folder):
    # Create the output folder if it doesn't exist
    output_folder = os.path.join(images_folder, "png_imgs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(images_folder):
        if file_name.lower().endswith('.heic'):
            # Full path of the HEIC image
            heic_image_path = os.path.join(images_folder, file_name)

            try:
                heic_img = HEIC2PNG(heic_image_path, quality=90)
                png_image_name = f"{os.path.splitext(file_name)[0]}.png"
                png_image_path = os.path.join(output_folder, png_image_name)
                
                heic_img.save(png_image_path)  # Save the PNG image
                print(f"Converted {file_name} to {png_image_name}")
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

if __name__ == "__main__":
    images_folder = r"C:\Users\works\OneDrive\Desktop\projects\vision\images\cvc"
    
    convert_heic_to_png(images_folder)
