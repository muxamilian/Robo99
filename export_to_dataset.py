import h5py
from PIL import Image
import os
from tqdm import tqdm

fonts = h5py.File('fonts.hdf5')['fonts']

num_fonts = fonts.shape[0]
num_chars = fonts.shape[1]
image_shape = fonts.shape[2:]
input_shape = (*image_shape, 3)

os.makedirs('dataset', exist_ok=True)
pbar = tqdm(total=num_chars*num_fonts)
for char_index in range(num_chars):
    os.makedirs(f'dataset/{char_index}')
    for font_index in range(num_fonts):
        char = fonts[font_index,char_index,:,:]
        im = Image.fromarray(char)
        im.save(f'datasets/{char_index}/{font_index}.png')
        pbar.update(1)
