import numpy as np
from sklearn.decomposition import PCA                     

def pca_transform(pca_channel, n_components):
    
    temp_res = []
    
    for channel in range(len(pca_channel)):
        
        pca, fit_pca = pca_channel[channel]
        
        pca_pixel = fit_pca[:, :n_components]
        
        pca_comp = pca.components_[:n_components, :]
        
        compressed_pixel = np.dot(pca_pixel, pca_comp) + pca.mean_
        
        temp_res.append(compressed_pixel)
            
    # (height, width, channel)
    compressed_image = np.transpose(temp_res, (1, 2, 0))  
    
    compressed_image = np.array(compressed_image,dtype=np.uint8)
    
    return compressed_image

def pca_compose(img_data):
    
    if img_data.shape[-1] > 3:
        img_data = img_data[:, :, :3]
    
    # (channel, height, width)
    img_t = np.transpose(img_data, (2, 0, 1))  

    pca_channel = {}

    for i in range(img_t.shape[0]):   
        channel = img_t[i]
        
        pca = PCA(random_state=13)
        fit_pca = pca.fit_transform(channel)

        pca_channel[i] = (pca, fit_pca)
    
    return pca_channel

def pca_find_valuable_comp(pca_channel, threshold=99.995):

    n_components_list = []
    
    for channel in pca_channel:

        pca, _ = pca_channel[channel]
        
        cum_var_exp = np.cumsum(pca.explained_variance_ratio_)  # Cumulative variance
        
        # Find the smallest number of components that reach the threshold
        n_components = np.argmax(cum_var_exp >= threshold / 100) + 1
        n_components_list.append(n_components)

    # Return the maximum `n_components` across all channels to satisfy the requirement globally
    return max(n_components_list)
