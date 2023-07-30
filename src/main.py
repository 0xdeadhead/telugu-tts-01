from indicnlp.tokenize import sentence_tokenize
from ai4bharat.transliteration import XlitEngine
from argparse import ArgumentParser
from TTS.api import TTS
from pydub import AudioSegment
import sys
import os


def ms_to_hhmmssms(milliseconds):
    # Calculate hours, minutes, seconds, and milliseconds
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)

    # Format the time as HH:MM:SS,ms
    time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return time_format


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    transliteration_engine = XlitEngine("te", beam_width=10, rescore=True)
    argument_parser.add_argument('-i', "--input_file")
    arguments = argument_parser.parse_args()
    input_file = arguments.input_file
    try:
        with open(input_file, 'r', encoding='utf-8') as file_data:
            sentences = sentence_tokenize.sentence_split(file_data.read(),
                                                         lang='te')
    except FileNotFoundError:
        print("File Not Found")
        sys.exit(1)

    sentences = list(
        map(lambda x: x.replace("\n", "").replace('"', "").replace("'", ""),
            sentences))
    sentences = map(
        lambda sentence: transliteration_engine.translit_sentence(sentence),
        sentences)
    sentences = map(lambda sentence: sentence['te'], sentences)
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    tts = TTS(model_path=f'{script_directory}/te/fastpitch/best_model.pth',
              config_path=f"{script_directory}/te/fastpitch/config.json",
              vocoder_path=f'{script_directory}/te/hifigan/best_model.pth',
              vocoder_config_path=f'{script_directory}/te/hifigan/config.json',
              progress_bar=True)
    final_clip = AudioSegment.empty()
    srt_string = ""
    for index, sentence in enumerate(sentences):
        tts.tts_to_file(text=sentence,
                        file_path=f"{index}.wav",
                        speaker=tts.speakers[0])
        dur_before_concatenate = ms_to_hhmmssms(len(final_clip) + 1)
        final_clip += AudioSegment.from_wav(file=f"{index}.wav")
        dur_after_concatenate = ms_to_hhmmssms(len(final_clip))
        srt_string += f"{index+1}\n{dur_before_concatenate} --> {dur_after_concatenate}\n{sentence}\n\n"

    with open("subtitles.srt", "w") as final_subtitles:
        final_subtitles.write(srt_string)
        final_clip.export("final_out.wav", format='wav')
