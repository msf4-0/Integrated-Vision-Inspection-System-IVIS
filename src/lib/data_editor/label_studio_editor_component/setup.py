import setuptools

setuptools.setup(
    name="label_studio_editor",
    version="0.1.0",
    author="Chu Zhen Hao ",
    author_email="",
    description="Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)",
    long_description=" Based on Label Studio Frontend",
    long_description_content_type="text/plain",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.6",
    install_requires=[
        # By definition, a Custom Component depends on Streamlit.
        # If your component has other Python dependencies, list
        # them here.
        "streamlit >= 0.63",
    ],
)
