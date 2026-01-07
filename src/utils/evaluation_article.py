import ast
import numpy as np
import pandas as pd
import torch
import os
import json
import random
import torchaudio
from tqdm import tqdm_notebook
from time import sleep

from Levenshtein import distance
from google import genai

from constants.evaluation_article import (
    PROMPT_ALL,
    TARGET_SENTENCES,
    SPECIAL_KEY_MAP,
    KEY_NORMALIZATION,
)
from utils.loading import AudioDataset, normalize_waveform
from models.coatnet import MyCoAtNet
from models.moat import MOAT

from dotenv import load_dotenv
load_dotenv()

np.random.seed(42)

ROOT = os.getenv("ARTICLE_ROOT")

def get_prompt(result):
    input_data = result["predictions"]
    prompt = PROMPT_ALL.format(input_data=input_data)

    return prompt


def predict_file(model, test_dataset, file_path):
    waveform, sr = torchaudio.load(file_path)

    if getattr(test_dataset, "noise_reduction", False):
        waveform = test_dataset.apply_noise_reduction(waveform, sr)

    waveform = normalize_waveform(waveform, test_dataset.dataset, root=ROOT).to(torch.float32)
    if waveform.shape[1] / sr > 0.5:
        spectrogram = test_dataset.transform_long(waveform)
    else:
        spectrogram = test_dataset.transform_short(waveform)

    spectrogram = spectrogram.unsqueeze(0)
    with torch.no_grad():
        output = model(spectrogram)
        return output.argmax().item()


def process_sentences_with_model(data_root, model_root, model_name):
    keys = model_name.split("_")
    model_type = keys[0]
    checkpoint_path = f"{model_root}/{model_name}/model_checkpoints/"
    config_path = f"{model_root}/{model_name}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name_in_config = config["model"]
    model_params = config["model_params"]
    if model_type == "coatnet":
        config["image_size"] = 64
        config["model_configs"][model_type][model_params]["image_size"] = 64
    elif model_type == "moat":
        config["img_size"] = 64
        config["model_configs"][model_type][model_params]["img_size"] = 64

    special_keys = config.get("special_keys", False)

    checkpoint_file = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    num_classes = len(config["class_encoding"])
    if model_type == "coatnet":
        model = MyCoAtNet(
            num_classes=num_classes,
            **config["model_configs"][model_name_in_config][model_params],
        )
    elif model_type == "moat":
        model = MOAT(
            num_classes=num_classes,
            **config["model_configs"][model_name_in_config][model_params],
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(checkpoint)
    model.eval()

    dataset = config["dataset"] if config["dataset"] != "custom_noisy" else "custom_mac"
    print(f'eval dataset: {dataset}')

    test_dataset = AudioDataset(
        root=data_root,
        dataset=dataset,
        transform_aug=False,
        special_keys=special_keys,
        image_size=config["image_size"],
        exclude_few_special_keys=True,
        sample_rate=22000,
        class_idx=config["class_encoding"],
    )

    idx_to_class = {idx: cls for cls, idx in test_dataset.class_to_idx.items()}

    files_by_label = {}
    for path, label in test_dataset.file_paths:
        files_by_label.setdefault(label, []).append(path)

    results = {}
    for i in tqdm_notebook(range(5), desc="Total Progress (Seeds)", unit="seed"):
        random.seed(i)
        
        for category_name, sentences_dict in tqdm_notebook(TARGET_SENTENCES.items(), leave=False):
            
            for s_id, s_data in sentences_dict.items():
                
                target_sentence_str = s_data["sentence"]
                target_keys = s_data["keys"]

                selected_paths = []
                for key in target_keys:
                    candidates = []
                    candidates.extend(files_by_label.get(key, []))
                    if candidates:
                        selected_paths.append(random.choice(candidates))

                normalized_target_keys = []
                for key in target_keys:
                    if key in ['lshift', 'rshift']:
                        normalized_target_keys.append('shift')
                    else:
                        normalized_target_keys.append(key)

                predictions = []
                for file_path in selected_paths:
                    pred_idx = predict_file(model, test_dataset, file_path)
                    mapped_class = idx_to_class[pred_idx]
                    if mapped_class in ['lshift', 'rshift']:
                        mapped_class = 'shift'
                    predictions.append(mapped_class)

                lev_dist = distance(normalized_target_keys, predictions)
                
                unique_key = f"{category_name}_{s_id}_{i}"
                results[unique_key] = {
                    "sentence_type": category_name,
                    "sentence_id": s_id,
                    "sentence_run": i,
                    "target_text": target_sentence_str,
                    
                    "target_text": target_sentence_str,
                    "keypresses": normalized_target_keys, 
                    "predictions": predictions, 
                    
                    "levenshtein_distance": lev_dist,
                }

    return results


def enhance_predictions_with_llm(results, temperature=0.0):
    """
    Enhance predictions using either the Gemini or Llama model based on the model_name argument.

    Args:
        results (dict): The results dictionary to enhance.
        model_name (str): The name of the model to use ("gemini" or "llama").
        temperature (float): The temperature setting for the model (default is 0.0).

    Returns:
        dict: The enhanced results with final predictions and Levenshtein distances.
    """
    client = genai.Client()
    with tqdm_notebook(results.items(), desc="Gemini Correction", unit="req") as pbar:
        for unique_key, result in pbar:
            
            s_id = result.get("sentence_id", "?")
            run = result.get("sentence_run", "?")
            
            pbar.set_postfix(run=run, sentence_id=s_id)

            prompt = get_prompt(result)

            try:
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
                model_output = response.text

                if model_output.startswith("```"):
                    model_output = model_output.lstrip("```json").rstrip("```").strip()

                try:
                    final_prediction = ast.literal_eval(model_output)
                    if isinstance(final_prediction, list):
                        cleaned_prediction = []
                        for k in final_prediction:
                            k_lower = str(k).lower()
                            
                            k_normalized = KEY_NORMALIZATION.get(k_lower, k_lower)
                            
                            cleaned_prediction.append(k_normalized)
                            
                        final_prediction = cleaned_prediction

                except (ValueError, SyntaxError) as e:
                    pbar.write(f"Error parsing response (ID {s_id}, Run {run}): {e}")
                    final_prediction = []

                final_lev_dist = distance(result["keypresses"], final_prediction)

                result["gemini_final_prediction"] = final_prediction
                result["gemini_levenshtein_distance"] = final_lev_dist

            except Exception as e:
                pbar.write(f"Error communicating with Gemini API (ID {s_id}): {e}")
                result["gemini_final_prediction"] = []
                result["gemini_levenshtein_distance"] = None
            
            sleep(20) 

    return results


def process_keys(keys):
    """
    Process keypresses into a string, handling Shift modifications for 
    brackets, symbols, numbers, and capitalization.
    """

    processed = []
    caps_lock = False
    shift_pressed = False

    for key in keys:
        if key == "caps":
            caps_lock = not caps_lock
            continue
        elif key in ["lshift", "rshift"]:
            shift_pressed = True
            continue
        
        current_char = ""
        
        if key in SPECIAL_KEY_MAP:
            normal, shifted = SPECIAL_KEY_MAP[key]
            current_char = shifted if shift_pressed else normal
        else:
            if shift_pressed:
                current_char = key.upper()
            elif caps_lock:
                current_char = key.upper()
            else:
                current_char = key

        processed.append(current_char)

        if shift_pressed:
            shift_pressed = False

    return "".join(processed)



def format_sentence_results_as_dataframe(results):
    data = []
    
    for unique_key, res in results.items():
        sentence_type = res.get("sentence_type", "Unknown")
        sentence_id = res.get("sentence_id", -1)
        sentence_run = res.get("sentence_run", 0)
        target_text = res.get("target_text", "")
        # keypresses_str = process_keys(res["keypresses"])
        # predicted_str = process_keys(res.get("predictions", []))
        
        gemini_raw = res.get("gemini_final_prediction", [])
        if gemini_raw is None: gemini_raw = []
        # gemini_str = process_keys(gemini_raw)
        
        data.append({
            "Type": sentence_type,
            "ID": sentence_id,
            "Run": sentence_run,
            "Target Sentence": target_text,
    
            # "Target Keys": keypresses_str,
            # "Model Prediction": predicted_str,
            # "Gemini Prediction": gemini_str,
            
            
            
            "Raw Target Keys": str(res["keypresses"]),
            "Raw Model Keys": str(res.get("predictions", [])),
            "Raw Gemini Keys": str(gemini_raw),

            "Lev Dist": res["levenshtein_distance"],
            "Gemini Lev Dist": res.get("gemini_levenshtein_distance", None),
        })

    df = pd.DataFrame(data)
    
    if not df.empty:
        df = df.sort_values(by=["Type", "ID", "Run"])
        
    return df


def save_results(results, model_root, filename="evaluation.csv"):
    """
    Save the results to a CSV file, redirecting to the appropriate formatting function
    based on the structure of the results.
    """

    df = format_sentence_results_as_dataframe(results)

    file_path = os.path.join(model_root, filename)
    df.to_csv(file_path, index=False)

    return

def log_predictions(results):
    for unique_key, result in results.items():
        print(f"--- Run {result['sentence_run']} | ID {result['sentence_id']} ({result['sentence_type']}) ---")
        print(f"Target:      {result['target_text']}")
        print(f"Keypresses:  {process_keys(result['keypresses'])}")
        print(f"Predictions: {process_keys(result['predictions'])}")
        print(f"Lev Dist:    {result['levenshtein_distance']}")
        print("")
