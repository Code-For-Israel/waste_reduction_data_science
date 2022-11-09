"""
This module is used to set up a Flask API service
"""
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/detect_trucks', methods=['GET'])
def detect_trucks():
    result = None
    return jsonify(data={
        'image_id': [
            "img00010000351_06_02_2022T15_16_52",
            "img00010000721_06_02_2022T15_16_56"
        ],
        'detection_results': [
            {'classes': ["uncovered", "uncovered", "covered", "other"],
             'probabilities': [0.125, 0.6343, 0.78, 0.983],
             'bboxs_cx_cy_w_h_fractional': [
                 [0.41, 0.621, 0.62, 0.78],  # (c_x, c_y, w, h) fractional
                 [0.82, 0.621, 0.62, 0.28]
             ]},
            {'classes': ["uncovered"],
             'probabilities': [0.326],
             'bboxs_cx_cy_w_h_fractional': [
                 [0.08, 0.99, 0.62, 0.41]  # (c_x, c_y, w, h) fractional
             ]}
        ]
    }
    )


if __name__ == "__main__":
    app.run()
