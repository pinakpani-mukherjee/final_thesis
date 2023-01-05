from pydub import AudioSegment
import torch, torchaudio
import os
import math


def combine_waves(paths):
    filenames = []
    for path in paths:
        filenames.append(path[:-4].split("/")[-1])

    combined = AudioSegment.from_wav(paths[0])
    for i in range(1, len(paths)):
        combined = combined.overlay(AudioSegment.from_wav(paths[i]))

    output_dir = f"{'/'.join(paths[0].split('/')[:-2])}/combined"

    final_path = f"{output_dir}/{'_'.join(filenames)}_combined_sound.wav"
    combined.export(final_path)
    return final_path


def sum_spl(spls):
    intensity = 0
    for spl in spls:
        intensity = intensity + (10 ** ((spl / 10) - 12))

    return 10 * math.log((intensity * (10 ** 12)), 10)


def get_file(wav_name):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'
    return waveform, sr


def make_features(waveform, sr, mel_bins, target_length=1024):
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def decrease_volume(wav_file_path, decrease_value):
    raw_path = wav_file_path[:-4]
    folder_path = raw_path.split("/")[:-1]
    file_name = raw_path.split("/")[-1]
    if not os.path.isdir(f"{raw_path}"):
        os.mkdir(f"{raw_path}")
    output_path = f"{'/'.join(folder_path)}/{file_name}/{file_name}_quieter_{str(decrease_value)[:7].replace('.', '_')}.wav"
    if not os.path.isfile(output_path):
        sound = AudioSegment.from_wav(wav_file_path)
        new_sound = sound - decrease_value
        new_sound.export(output_path, "wav")
    return output_path


class Point:
    def __init__(self, x_pos=None, y_pos=None, signal_path=None, signal_features_data=None, spl=None,
                 output_status=None):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.signal_path = signal_path
        self.signal_features_data = signal_features_data
        self.spl = spl
        self.output_status = output_status


# define a mesh

class Mesh:
    def __init__(self, x_length=10, y_length=10, x_tick=1, y_tick=1):
        self.x_length = x_length
        self.y_length = y_length
        self.x_tick = x_tick
        self.y_tick = y_tick
        self.mesh = self.initialize_mesh(x_length, y_length, x_tick, y_tick)
        self.sources = []

    def initialize_mesh(self, x_length, y_length, x_tick, y_tick):
        return_array = []
        for i in range(0, int(x_length / x_tick)):
            temp_arr = []
            for j in range(0, int(y_length / y_tick)):
                temp_arr.append(Point(x_pos=i, y_pos=j))
            return_array.append(temp_arr)
        return return_array

    def set_source(self, audio_file_path, x=5, y=5, initial_spl_at_src=100):
        source_dict = {}
        source_dict["source_x"] = x
        source_dict["source_y"] = y
        source_dict["src_audio_file"] = audio_file_path
        source_dict["src_spl"] = initial_spl_at_src
        self.sources.append(source_dict)
        # self.mesh[self.source_x][self.source_y].spl = initial_spl_at_src
        # self.mesh[self.source_x][self.source_y].signal_path = audio_file_path
        # waveform_src, sr_src = get_file(audio_file_path)
        # self.mesh[self.source_x][self.source_y].signal_features_data = make_features(waveform_src, sr_src,
        #                                                                             mel_bins=128)
        return self

    def create_spls(self):
        if self.sources == []:
            assert "Cannot calculate Sound Pressure Levels as no sources are provided."

        for i in range(0, int(self.x_length / self.x_tick)):
            for j in range(0, int(self.y_length / self.y_tick)):
                signals = []
                spls = []

                for source in self.sources:
                    if i == source["source_x"] and j == source["source_y"]:
                        signals.append(source["src_audio_file"])
                        spls.append(source["src_spl"])
                    else:
                        euclidian_distance = math.sqrt((source["source_x"] - i) ** 2 + (source["source_y"] - j) ** 2)
                        change_in_spl = 20 * math.log((euclidian_distance / 0.1), 10)
                        decreased_volume_file_path = decrease_volume(source["src_audio_file"], change_in_spl)
                        signals.append(decreased_volume_file_path)
                        spls.append(source["src_spl"] - change_in_spl)

                self.mesh[i][j].signal_path = combine_waves(signals)
                self.mesh[i][j].spl = sum_spl(spls)
                waveform_src, sr_src = get_file(self.mesh[i][j].signal_path)
                self.mesh[i][j].signal_features_data = make_features(waveform_src, sr_src, mel_bins=128)


if __name__ == "__main__":
    kappa = Mesh(10, 10, 1, 1)
    print(kappa.mesh[0][1].x_pos)
    print(kappa.mesh[0][1].y_pos)
    kappa.set_source(x=5, y=5)
    print(kappa.source_x)
    print(kappa.source_y)
