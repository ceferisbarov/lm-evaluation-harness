import json


def json_answer_exact_match(references: list[str], predictions: list[str]) -> bool:
    assert len(predictions) == 1, "Currently, we don't support pass@k."
    reference = references[0]
    prediction = predictions[0]  # Since predictions is a list of strings
    try:
        prediction_json = json.loads(
            prediction.strip().strip("```").strip("json").strip()
        )
    except json.JSONDecodeError:
        return False
    return str(prediction_json["answer"]) == str(reference.split("### ")[-1].rstrip())
