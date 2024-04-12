import whisper
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

whisperModel = whisper.load_model("base")

def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(whisperModel.device)

    _, probs = whisperModel.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisperModel, mel, options)

    return result.text, lang


@app.route('/voice_to_text', methods=['POST'])
def handle_request():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        # Save the file to a temporary location
        file_path = 'temp.wav'
        file.save(file_path)

        # Extract emotion from the sound file
        raw_text2, lang = inference('temp.wav')

        # Delete the temporary file
        os.remove(file_path)

        return jsonify({'text': raw_text2, 'lang': lang}), 200
    except Exception as e:
        return jsonify({"error is": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
