import sys
sys.path.append("pca-transform")
sys.path.append("img-processing")
sys.path.append("filepath-processing")

import click
from pca_transform import pca_transform, pca_compose, pca_find_valuable_comp
from img_processing import img_data, save_img, img_desc
from filepath_processing import compressed_filepath


@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--comp-per", type=float, help="Compressed percent (%)")
def main(filepath, comp_per):
    org_image_data = img_data(filepath)
    org_image_desc = img_desc(org_image_data)

    pca_channel = pca_compose(org_image_data['img_data'])
    compressed_percent = comp_per if comp_per else 99.9995
    valuable_comp = pca_find_valuable_comp(pca_channel, threshold=compressed_percent)

    compressed_image_filepath = compressed_filepath(filepath)
    compressed_image = pca_transform(pca_channel,n_components=valuable_comp)
    save_img(compressed_image, compressed_image_filepath)

    compressed_image_data = img_data(compressed_image_filepath)
    compressed_image_desc = img_desc(compressed_image_data)

    click.echo(f"Origin: {org_image_desc}")
    click.echo(f"Compressed success: {compressed_image_desc} - compressed percent: {compressed_percent}")

if __name__ == "__main__":
    main()
