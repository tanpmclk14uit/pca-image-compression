from PIL import Image
import numpy as np        
import os

def img_data(imgPath):
    
    orig_img = Image.open(imgPath)
    
    img_size_kb = os.stat(imgPath).st_size/1024
    
    ori_pixels = np.array(orig_img.getdata()).reshape(*orig_img.size, -1)
    
    img_dim = ori_pixels.shape 
    
    data_dict = {}
    data_dict['img_size_kb'] = img_size_kb
    data_dict['img_dim'] = img_dim
    data_dict['img_data'] = np.array(orig_img)
    
    return data_dict

def img_desc(img_data):
    return f"size: {img_data['img_size_kb']}kb - dim: {img_data['img_dim']}"

def save_img(img_data, filepath): 
    Image.fromarray(img_data).save(filepath)
    return