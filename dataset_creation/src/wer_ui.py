import os
from flask import Flask, request
from flask import render_template

import jiwer 
import jiwer.transforms as tr

import srt

app = Flask(__name__)

def parse_file(path):
    file = open(path, "r")
    srt_text = file.read()
    if ".srt" in path:
        txt = ""
        srt_parsed = srt.parse(srt_text)
        for i in srt_parsed:
            txt += i.content + " "
        return txt
    elif ".txt" in path:
        return srt_text


@app.route('/')
def compare():
    predicted_file = "h_large_0.txt"
    true_file = "captions/h_0.srt"

    text_predcited = parse_file(predicted_file)
    text_true = parse_file(true_file)

    transformation = tr.Compose([
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToSingleSentence(),
        tr.ReduceToListOfListOfWords(),
    ]) 

    out = jiwer.process_words(
        [text_true],
        [text_predcited],
        reference_transform=transformation,
        hypothesis_transform=transformation
    )

    text_hypotheses = [str(i) for i in out.hypotheses[0]]
    text_references = [str(i) for i in out.references[0]]
    
    for i, alignment in enumerate(out.alignments[0]):
        if alignment.type == "equal":
            text_hypotheses[alignment.hyp_start_idx] = f"<span class='green compare_{i}'>" + text_hypotheses[alignment.hyp_start_idx]
            text_hypotheses[alignment.hyp_end_idx-1] = text_hypotheses[alignment.hyp_end_idx-1] + "</span>"


            text_references[alignment.ref_start_idx] = f"<span class='green compare_{i}'>" + text_references[alignment.ref_start_idx]
            text_references[alignment.ref_end_idx-1] = text_references[alignment.ref_end_idx-1] + "</span>"
    text_hypotheses = " ".join(text_hypotheses)
    text_references = " ".join(text_references)

    return render_template('compare.html', error=out.wer, total=len(out.references[0]), hits=out.hits, acc=int(out.hits/len(out.references[0])*100), text_predcited=text_hypotheses, text_true=text_references) 


if __name__ == '__main__':
    app.run(debug = True)