"""An example custom metric: reward answers that stay short and avoid filler.

Strop calls score(output, row) once per example. Return a number from 0 to 1, or a dict
with a "score" and a short "feedback" that is shown to the optimizer.
"""

FILLER = ("as an ai", "i hope this helps", "feel free to reach out")


def score(output, row):
    text = output.lower()
    words = len(output.split())
    if any(phrase in text for phrase in FILLER):
        return {"score": 0.5, "feedback": "contains filler such as 'I hope this helps'"}
    if words > 120:
        return {"score": 0.5, "feedback": f"{words} words; aim for under 120"}
    return 1.0
