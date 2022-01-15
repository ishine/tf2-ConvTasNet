import keras_autodoc
import pathlib
import shutil

PAGES = {
    "The Algorithm.md": [
        "convtasnet.algorithm.ConvTasNet",
        "convtasnet.algorithm.MaskGenerator",
        "convtasnet.algorithm.ConvBlock",
    ]
}

root_dir = pathlib.Path(__file__).resolve().parents[1]


def generate(dest_dir):
    doc_generator = keras_autodoc.DocumentationGenerator(PAGES)
    doc_generator.generate(dest_dir)
    shutil.copyfile(root_dir / "README.md", dest_dir / "index.md")


if __name__ == "__main__":
    generate(root_dir / "docs" / "sources")
