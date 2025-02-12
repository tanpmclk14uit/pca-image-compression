import sys
sys.path.extend(["pca-transform", "img-processing", "filepath-processing"])

import click
from pca_transform import pca_transform, pca_compose, pca_find_valuable_comp
from img_processing import img_data, save_img, img_desc
from filepath_processing import compressed_filepath

def process_image(filepath, comp_per):
    org_image_data = img_data(filepath)
    org_image_desc = img_desc(org_image_data)
    
    pca_channel = pca_compose(org_image_data['img_data'])
    compressed_percent = comp_per or 99.9995
    valuable_comp = pca_find_valuable_comp(pca_channel, threshold=compressed_percent)
    
    compressed_image = pca_transform(pca_channel, n_components=valuable_comp)
    compressed_image_filepath = compressed_filepath(filepath)
    save_img(compressed_image, compressed_image_filepath)
    
    return org_image_desc, compressed_image_filepath, compressed_percent

def display_results(org_desc, compressed_filepath, compressed_percent):
    compressed_image_desc = img_desc(img_data(compressed_filepath))
    click.echo(f"Origin: {org_desc}")
    click.echo(f"Compressed success: {compressed_image_desc} - compressed percent: {compressed_percent}")

@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--comp-per", type=float, help="Compressed percent (%)")
def main(filepath, comp_per):
    org_desc, compressed_filepath, compressed_percent = process_image(filepath, comp_per)
    display_results(org_desc, compressed_filepath, compressed_percent)

if __name__ == "__main__":
    main()
