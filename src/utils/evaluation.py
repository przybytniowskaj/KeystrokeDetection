import ast
import requests
import numpy as np
import pandas as pd
import torch
import os
import json
import random
import torchaudio

from Levenshtein import distance
from google import genai

from src.constants.evaluation import PROMPT_ALPHANUM, PROMPT_ALL, TARGET_SENTENCES, TARGET_WORDS
from src.utils.loading import AudioDataset, normalize_waveform
from src.models.coatnet import MyCoAtNet
from src.models.moat import MOAT

np.random.seed(42)


def get_prompt(result):
    if "word_predictions" in result:
        input_data = result["word_predictions"]
        prompt = PROMPT_ALPHANUM.format(input_data=input_data)
    else:
        input_data = result["predictions"]
        prompt = PROMPT_ALL.format(input_data=input_data)

    return prompt


def predict_file(model, test_dataset, file_path):
    waveform, sr = torchaudio.load(file_path)

    if getattr(test_dataset, "noise_reduction", False):
        waveform = test_dataset.apply_noise_reduction(waveform, sr)

    waveform = normalize_waveform(waveform, test_dataset.dataset).to(torch.float32)
    if waveform.shape[1] / sr > 0.5:
        spectrogram = test_dataset.transform_long(waveform)
    else:
        spectrogram = test_dataset.transform_short(waveform)

    spectrogram = spectrogram.unsqueeze(0)
    with torch.no_grad():
        output = model(spectrogram)
        return output.argmax().item()


def process_sentences_with_model(data_root, model_root, model_name):
    keys = model_name.split('_')
    model_type = keys[0]
    checkpoint_path = f"{model_root}/{model_type}/{model_name}/model_checkpoints/"
    config_path = f"{model_root}/{model_type}/{model_name}/config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name_in_config = config['model']
    model_params = config['model_params']
    if model_type == 'coatnet':
        config['image_size'] = 64
        config['model_configs'][model_type][model_params]['image_size'] = 64
    elif model_type == 'moat':
        config['img_size'] = 128
        config['model_configs'][model_type][model_params]['img_size'] = 128
    special_keys = config.get('special_keys', False)

    checkpoint_file = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    num_classes = len(config['class_encoding'])
    if model_type == 'coatnet':
        model = MyCoAtNet(num_classes=num_classes, **config['model_configs'][model_name_in_config][model_params])
    elif model_type == 'moat':
        model = MOAT(num_classes=num_classes, **config['model_configs'][model_name_in_config][model_params])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(checkpoint)
    model.eval()

    test_dataset = AudioDataset(
        root=data_root,
        dataset=config['dataset'],
        transform_aug=False,
        special_keys=special_keys,
        image_size=config['image_size'],
        exclude_few_special_keys=True,
        sample_rate=22000,
        class_idx=config['class_encoding'],
    )

    idx_to_class = {idx: cls for cls, idx in test_dataset.class_to_idx.items()}

    files_by_label = {}
    for path, label in test_dataset.file_paths:
        files_by_label.setdefault(label, []).append(path)

    results = {}

    if special_keys:
        for sentence, keypresses in TARGET_SENTENCES.items():
            selected_paths = []
            for key in keypresses:
                candidates = []
                candidates.extend(files_by_label.get(key, []))
                if candidates:
                    selected_paths.append(random.choice(candidates))

            predictions = []
            for file_path in selected_paths:
                pred_idx = predict_file(model, test_dataset, file_path)
                predictions.append(idx_to_class[pred_idx])

            lev_dist = distance(keypresses, predictions)

            results[sentence] = {
                "keypresses": keypresses,
                "predictions": predictions,
                "levenshtein_distance": lev_dist,
            }
    else:
        for sentence, words in TARGET_WORDS.items():
            word_predictions = []
            word_keypresses = []

            for word_keys in words:
                selected_paths = []
                for key in word_keys:
                    candidates = []
                    candidates.extend(files_by_label.get(key, []))
                    if candidates:
                        selected_paths.append(random.choice(candidates))

                preds = []
                for file_path in selected_paths:
                    pred_idx = predict_file(model, test_dataset, file_path)
                    preds.append(idx_to_class[pred_idx])

                word_predictions.append(preds)
                word_keypresses.append(word_keys)

            flat_keys = [k for wk in word_keypresses for k in wk]
            flat_preds = [p for wp in word_predictions for p in wp]
            lev_dist = distance(flat_keys, flat_preds)

            results[sentence] = {
                "word_keypresses": word_keypresses,
                "word_predictions": word_predictions,
                "keypresses": flat_keys,
                "predictions": word_predictions,
                "levenshtein_distance": lev_dist,
            }

    return results


def enhance_predictions_with_llm(results, model_name="gemini", temperature=0.0):
    """
    Enhance predictions using either the Gemini or Llama model based on the model_name argument.

    Args:
        results (dict): The results dictionary to enhance.
        model_name (str): The name of the model to use ("gemini" or "llama").
        temperature (float): The temperature setting for the model (default is 0.0).

    Returns:
        dict: The enhanced results with final predictions and Levenshtein distances.
    """
    if model_name == "gemini":
        client = genai.Client()
    elif model_name == "gpt-oss":
        gpt_model = "@cf/openai/gpt-oss-120b"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    for _, result in results.items():
        prompt = get_prompt(result)

        try:
            if model_name == "gemini":
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
                model_output = response.text
            elif model_name == "gpt-oss":
                input_data = {
                    "model": gpt_model,
                    "input": prompt
                }
                # print(input_data)
                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/ai/v1/responses",
                    headers={"Authorization": f"Bearer {os.getenv('CLOUDFLARE_API_KEY')}"},
                    json=input_data
                ).json()
                model_output = response.get("output", {})[-1]
                model_output = model_output.get("content", {})[0]
                model_output = model_output.get("text", "")

            if model_output.startswith("```"):
                model_output = model_output.lstrip('```json').rstrip('```')

            try:
                final_prediction = ast.literal_eval(model_output)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing response with ast.literal_eval: {e}")
                print(model_output)
                final_prediction = []

            # Calculate Levenshtein distance
            if "word_predictions" in result:
                flat_keys = [k for wk in result["word_keypresses"] for k in wk]
                flat_final_preds = [p for wp in final_prediction for p in wp]
                final_lev_dist = distance(flat_keys, flat_final_preds)
            else:
                final_lev_dist = distance(result["keypresses"], final_prediction)

            # Add results to the dictionary
            result[f"{model_name}_final_prediction"] = final_prediction
            result[f"{model_name}_levenshtein_distance"] = final_lev_dist

        except Exception as e:
            print(f"Error communicating with {model_name.capitalize()} API: {e}")
            result[f"{model_name}_final_prediction"] = None
            result[f"{model_name}_levenshtein_distance"] = None

    return results


def format_results_as_dataframe(results):
    """
    Format the results into a pandas DataFrame for better readability, including results from multiple models.
    """
    data = []
    for sentence, result in results.items():
        word_keypresses = " ".join(["".join(word) for word in result["word_keypresses"]])
        raw_keypresses = result["word_keypresses"]
        word_predictions = " ".join(["".join(word) for word in result["word_predictions"]])
        raw_predictions = result["word_predictions"]
        gemini_final_prediction = " ".join(["".join(word) for word in result.get("gemini_final_prediction", [])])
        gemini_raw_predictions = result.get("gemini_final_prediction", [])
        gpt_final_prediction = " ".join(["".join(word) for word in result.get("gpt-oss_final_prediction", [])])
        gpt_raw_prediction = result.get("gpt-oss_final_prediction", [])
        levenshtein_distance = result["levenshtein_distance"]
        gemini_levenshtein_distance = result.get("gemini_levenshtein_distance", None)
        gpt_levenshtein_distance = result.get("gpt-oss_levenshtein_distance", None)

        data.append({
            "Sentence": sentence,
            "Keypresses": word_keypresses,
            "Raw Keypresses": raw_keypresses,
            "Predictions": word_predictions,
            "Raw Predictions": raw_predictions,
            "Gemini Final Prediction": gemini_final_prediction,
            "Gemini Raw Prediction": gemini_raw_predictions,
            "GPT Final Prediction": gpt_final_prediction,
            "GPT Raw Prediction": gpt_raw_prediction,
            "Levenshtein Distance": levenshtein_distance,
            "Gemini Levenshtein Distance": gemini_levenshtein_distance,
            "GPT Levenshtein Distance": gpt_levenshtein_distance
        })

    return pd.DataFrame(data)


def process_keys(keys):
    """
    Process the keypresses or predictions to handle special keys like shift, caps, space, etc.
    """
    processed = []
    caps_lock = False
    shift_pressed = False

    for key in keys:
        if key == "caps":
            caps_lock = not caps_lock
        elif key == "lshift" or key == "rshift":
            shift_pressed = True
        elif key == "space":
            processed.append(" ")
        elif key == "comma":
            processed.append(",")
        elif key == "dot":
            processed.append(".")
        else:
            if shift_pressed:
                processed.append(key.upper())
                shift_pressed = False
            elif caps_lock:
                processed.append(key.upper())
            else:
                processed.append(key)

    return "".join(processed)


def format_sentence_results_as_dataframe(results):
    """
    Format the results into a pandas DataFrame for sentences (not words), handling special keys and including results from multiple models.
    """
    data = []
    for sentence, result in results.items():
        raw_keypresses = result['keypresses']
        keypresses = process_keys(result["keypresses"])
        raw_predictions = result.get("predictions", [])
        predictions = process_keys(result["predictions"])
        gemini_raw_predictions = result.get("gemini_final_prediction", [])
        gemini_final_prediction = process_keys(result.get("gemini_final_prediction", []))
        gpt_raw_prediction = result.get("gpt-oss_final_prediction", [])
        gpt_final_prediction = process_keys(result.get("gpt-oss_final_prediction", []))
        levenshtein_distance = result["levenshtein_distance"]
        gemini_levenshtein_distance = result.get("gemini_levenshtein_distance", None)
        gpt_levenshtein_distance = result.get("gpt-oss_levenshtein_distance", None)

        data.append({
            "Sentence": sentence,
            "Keypresses": keypresses,
            "Raw Keypresses": raw_keypresses,
            "Predictions": predictions,
            "Raw Predictions": raw_predictions,
            "Gemini Final Prediction": gemini_final_prediction,
            "Gemini Raw Prediction": gemini_raw_predictions,
            "GPT Final Prediction": gpt_final_prediction,
            "GPT Raw Prediction": gpt_raw_prediction,
            "Levenshtein Distance": levenshtein_distance,
            "Gemini Levenshtein Distance": gemini_levenshtein_distance,
            "GPT Levenshtein Distance": gpt_levenshtein_distance
        })

    return pd.DataFrame(data)


def save_results(results, model_root, filename="evaluation.csv"):
    """
    Save the results to a CSV file, redirecting to the appropriate formatting function
    based on the structure of the results.
    """
    if "word_predictions" in next(iter(results.values())):
        df = format_results_as_dataframe(results)
    else:
        df = format_sentence_results_as_dataframe(results)

    # Save the DataFrame to a CSV file
    file_path = os.path.join(model_root, filename)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

    return df


def log_predictions(results):
    for sentence, result in results.items():
        print(f"Sentence: {sentence}")
        if "word_predictions" in result:
            print(f"Word keypresses: {result['word_keypresses']}")
            print(f"Word predictions: {result['word_predictions']}")
        else:
            print(f"Keypresses: {result['keypresses']}")
            print(f"Predictions: {result['predictions']}")
        print(f"Levenshtein distance: {result['levenshtein_distance']}")