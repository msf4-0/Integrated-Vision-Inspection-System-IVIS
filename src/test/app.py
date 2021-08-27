
import streamlit as st
from label_studio_editor import labelstudio_editor

config = ''' <View>
<Image name="img" value="$image"></Image>
<RectangleLabels name="tag" toName="img" fillOpacity="0.5" strokeWidth="5">
<Label value="Planet"></Label>
<Label value="Moonwalker" background="blue"></Label>
</RectangleLabels>
<KeyPointLabels name="kp" toName="img" strokeWidth="1">
<Label value="Point1"/>
<Label value="Point2"/>
</KeyPointLabels>
</View>'''


interfaces = [
    "panel",
    "update",
    "controls",
    "side-column",
    "annotations:menu",
    "annotations:add-new",
    "annotations:delete",
    "predictions:menu",
    "skip",
]

user = {
    "pk": 2,
    "firstName": "Zhen Hao",
    "lastName": "Chu",
}

task = {
    "annotations": [
        {
            "id": "1001",
            "lead_time": 15.053,
            "result": [
                {
                    "original_width": 2242,
                    "original_height": 2802,
                    "image_rotation": 0,
                    "value": {
                        "x": 30,
                        "y": 24.759871931696907,
                        "width": 12.4,
                        "height": 10.458911419423693,
                        "rotation": 0,
                        "rectanglelabels": ["Moonwalker"],
                    },
                    "id": "Dx_aB91ISN",
                    "from_name": "tag",
                    "to_name": "img",
                    "type": "rectanglelabels",
                },
                {
                    "original_width": 2242,
                    "original_height": 2802,
                    "image_rotation": 0,
                    "value": {
                        "x": 45.733333333333334,
                        "y": 22.30522945570971,
                        "width": 12.666666666666666,
                        "height": 8.858057630736393,
                        "rotation": 0,
                        "rectanglelabels": ["Planet"],
                    },
                    "id": "YdtLI2svMR",
                    "from_name": "tag",
                    "to_name": "img",
                    "type": "rectanglelabels",
                },
            ],
            "data": {
                "image":
                "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg",
            },
        },
    ],
    "predictions": [
    ],
    "data": {
        "image":
        "https://ichef.bbci.co.uk/news/976/cpsprodpb/C82E/production/_118164215_0x0-model_y_02.jpg",
    },
    "data": {
        "image":
        "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/soroush-karimi-crjPrExvShc-unsplash.jpg",
    },
}

results_raw = labelstudio_editor(
    config, interfaces, user, task, key='test')
st.write(results_raw)
