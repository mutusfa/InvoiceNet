import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name="invoice-net-mutusfa",
                 version="0.0.2",
                 author="Julius Juodagalvis",
                 author_email="juodagalvis@protonmail.com",
                 description="Models for invoice scanning",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/mutusfa/InvoiceNet",
                 package_dir={'': 'src'},
                 packages=['invoice_net'],
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 install_requires=[
                     "datefinder",
                     "gensim",
                     "Keras",
                     "matplotlib",
                     "numpy",
                     "nltk",
                     "pandas",
                     "PyYAML",
                     "sklearn",
                     "tensorflow",
                     "tqdm",
                     "fasttext @ https://github.com/facebookresearch/fastText/tarball/master#egg=fasttext-0.9.1"
                 ])
