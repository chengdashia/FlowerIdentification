import io
from flask import Blueprint, request, jsonify, current_app, make_response
from flask_restx import Resource, Namespace
from app import api
from PIL import Image
import torch
import itertools
import numpy as np

identify_bp = Blueprint('identify', __name__)
identify_ns = Namespace('identify', description='File upload operations')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_mime_type(mime_type):
    return mime_type in ALLOWED_MIME_TYPES


def check_combination_in_label_data(label_data, combined_labels):
    split_labels = [combined_labels[i:i + 2] for i in range(0, len(combined_labels), 2)]
    return ((label_data['头状花序'] == int(split_labels[0])) &
            (label_data['花心'] == int(split_labels[1])) &
            (label_data['内侧颜色'] == int(split_labels[2])) &
            (label_data['内侧主色'] == int(split_labels[3])) &
            (label_data['内侧次色'] == int(split_labels[4]))).any()


def result(probabilities):
    # Get the label data from the app config
    label_data = current_app.config['LABEL_DATA']

    # 存储预测结果
    results = {}

    # 输出预测结果
    for i, probs in enumerate(probabilities):
        label_probs = " ".join([f"{j:02}:{prob:.4f}" for j, prob in enumerate(probs[0])])
        print(f'标签 {i + 1} 的概率: {label_probs}')

    top_labels = [np.argmax(probs[0]) for probs in probabilities]
    top_labels_str = "".join([f"{label:02}" for label in top_labels])
    results['top_labels_str'] = top_labels_str

    all_non_zero_labels = [[(j, prob) for j, prob in enumerate(probs[0]) if prob > 0] for probs in probabilities]
    combinations = list(itertools.product(*all_non_zero_labels))
    comb_probs = [(comb, np.prod([p for _, p in comb])) for comb in combinations]
    top_combinations = sorted(comb_probs, key=lambda x: x[1], reverse=True)[:8]
    top_combinations = top_combinations[1:]

    total_similarity = sum(prob for _, prob in top_combinations)
    normalized_combinations = [(comb, prob / total_similarity) for comb, prob in top_combinations]

    combined_results = []
    for combined_labels, normalized_prob in normalized_combinations:
        combined_labels_str = "".join([f"{label:02}" for label, _ in combined_labels])
        if check_combination_in_label_data(label_data, combined_labels_str):
            combined_results.append((combined_labels_str, normalized_prob, '近似品种'))
        else:
            combined_results.append((combined_labels_str, normalized_prob, '暂不确定该品种是否存在'))

    results['combined'] = combined_results
    return results


@identify_ns.route('/image', methods=['POST'])
class UploadImage(Resource):
    @identify_ns.doc(
        description='上传花卉图片并识别',
        responses={200: 'Image processed successfully', 400: 'Invalid input', 500: 'Internal server error'}
    )
    def post(self):
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"message": "File type not allowed"}), 400

        if not allowed_mime_type(file.content_type):
            return jsonify({"message": "MIME type not allowed"}), 400

        try:
            # Read the image file
            img_str = file.read()
            # Load the image and ensure it's in RGB format
            image = Image.open(io.BytesIO(img_str)).convert("RGB")

            # Get the model, transform, and label data from the app config
            model = current_app.config['MODEL']
            transform = current_app.config['TRANSFORM']

            # Transform the image to a tensor and add batch dimension
            image_tensor = transform(image).unsqueeze(0).to('cpu')

            with torch.no_grad():
                outputs = model(image_tensor)

            probabilities = [torch.softmax(output, dim=1).cpu().numpy() for output in outputs]
            result_data = result(probabilities)

            return make_response(jsonify({"message": "Image processed successfully", "result": result_data}), 200)
        except Exception as e:
            return make_response(jsonify({"message": "Failed to process image", "error": str(e)}), 500)


# Add the namespace to the api
api.add_namespace(identify_ns)
