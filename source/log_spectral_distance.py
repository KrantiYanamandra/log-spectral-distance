import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def get_log_spectral_distance(audio_buffer_1, audio_buffer_2):

    # Compute FFT
    fft_1 = np.fft.fft(audio_buffer_1)
    fft_2 = np.fft.fft(audio_buffer_2)

    # Compute power spectra
    power_spectrum_1 = np.abs(fft_1) ** 2
    power_spectrum_2 = np.abs(fft_2) ** 2

    # Compute LSD
    log_spectral_distance = np.sqrt(np.mean(np.power(10 * np.log10(power_spectrum_1 / power_spectrum_2), 2)))

    return log_spectral_distance, power_spectrum_1, power_spectrum_2


def get_mfcc_euclidean_distance(audio_buffer_1, audio_buffer_2, sr):

    mfcc_1 = librosa.feature.mfcc(y=audio_buffer_1, sr=sr, n_mfcc=13)
    mfcc_2 = librosa.feature.mfcc(y=audio_buffer_2, sr=sr, n_mfcc=13)

    euclidean_distance = np.linalg.norm(np.subtract(mfcc_1, mfcc_2), axis=1)

    return np.mean(euclidean_distance)


def stft_and_spectral_centroid(audio_buffer_1, audio_buffer_2, file_name_1, file_name_2, frame_size, hop_size, sr):

    # Compute STFT
    stft_1 = librosa.stft(audio_buffer_1, n_fft=frame_size, hop_length=hop_size)
    stft_2 = librosa.stft(audio_buffer_2, n_fft=frame_size, hop_length=hop_size)

    # Convert to power spectrogram
    power_spectrogram_1 = np.power(np.abs(stft_1), 2)
    power_spectrogram_2 = np.power(np.abs(stft_2), 2)

    # Convert to log-amplitude spectrogram
    log_power_spectrogram_1 = librosa.power_to_db(power_spectrogram_1, ref=np.max)
    log_power_spectrogram_2 = librosa.power_to_db(power_spectrogram_2, ref=np.max)

    # Compute Spectral Centroid
    sc_1 = librosa.feature.spectral_centroid(y=audio_buffer_1, sr=sr, n_fft=frame_size, hop_length=hop_size)
    sc_2 = librosa.feature.spectral_centroid(y=audio_buffer_2, sr=sr, n_fft=frame_size, hop_length=hop_size)
    sc_1[sc_1 == 0] = 0.0000001
    sc_2[sc_2 == 0] = 0.0000001
    spectral_centroid_distance = np.mean(np.power(10 * np.log10(sc_1 / sc_2), 2))

    times = librosa.times_like(sc_1, sr=sr, hop_length=hop_size)

    # Visualise STFTs and Spectral Centroids
    fig1, ax1 = plt.subplots(2, 1, sharex='all', figsize=(14, 7))
    fig1.suptitle(f'Log-scaled STFTs')

    ax1[0].set(title=f'{file_name_1}')
    ax1[0].label_outer()
    img = librosa.display.specshow(log_power_spectrogram_1, sr=sr, n_fft=frame_size, hop_length=hop_size,
                                   x_axis='time',
                                   y_axis='log',
                                   ax=ax1[0])
    ax1[0].plot(times, sc_1.T, label='Spectral Centroid', color='w', linewidth=1)
    ax1[0].legend(loc='upper right')

    ax1[1].set(title=f'{file_name_2}')
    librosa.display.specshow(log_power_spectrogram_2, sr=sr, n_fft=frame_size, hop_length=hop_size,
                             x_axis='time',
                             y_axis='log',
                             ax=ax1[1])
    ax1[1].plot(times, sc_2.T, label='Spectral Centroid', color='w', linewidth=1)
    ax1[1].legend(loc='upper right')

    fig1.colorbar(img, ax=ax1, format="%+2.f dB", aspect=50)

    return spectral_centroid_distance


def power_spectra(power_spectrum_1, power_spectrum_2, sr):

    fig2, ax2 = plt.subplots(2, 1, sharex='all', figsize=(14, 7))
    fig2.suptitle(f'Log Power spectra')

    num_samples = len(power_spectrum_1)
    sampling_period = 1.0 / sr
    frequencies = np.linspace(0.0, 1.0 / (2.0 * sampling_period), num_samples // 2)

    ax2[0].set(title=f'{file_name_1}')
    ax2[0].semilogx(frequencies, 2.0 / num_samples * power_spectrum_1[:num_samples // 2], linewidth=1)
    ax2[0].set_xlim([1, sr // 2])
    ax2[0].grid()
    ax2[0].set_ylabel('Power (dB / Hz)')

    ax2[1].set(title=f'{file_name_2}')
    ax2[1].semilogx(frequencies, 2.0 / num_samples * power_spectrum_2[:num_samples // 2], linewidth=1)
    ax2[1].set_xlim([1, sr // 2])
    ax2[1].grid()
    ax2[1].set_xlabel('Frequency (Hz)')
    ax2[1].set_ylabel('Power (dB / Hz)')


def coherence(audio_buffer_1, audio_buffer_2, frame_size, hop_size, sr):

    plt.figure()
    plt.grid()
    coherence, _ = plt.cohere(audio_buffer_1, audio_buffer_2, linewidth=1, Fs=sr, NFFT=frame_size, noverlap=hop_size)
    plt.xlabel('Frequency (Hz)')
    plt.title(f'Coherence')
    plt.show()


def band_energy_ratio(audio_buffer_1, audio_buffer_2, file_name_1, file_name_2, frame_size, hop_size, sr):

    split_at_frequency = 200

    stft_1 = librosa.stft(audio_buffer_1, n_fft=frame_size, hop_length=hop_size)
    stft_2 = librosa.stft(audio_buffer_2, n_fft=frame_size, hop_length=hop_size)

    frequency_range = sr / 2
    frequency_bin_size = frequency_range / stft_1.shape[0]

    split_at_bin = int(np.floor(split_at_frequency / frequency_bin_size))

    ber_1 = get_band_energy_ratio(stft_1, split_at_bin)
    ber_2 = get_band_energy_ratio(stft_2, split_at_bin)

    ber_distance = np.mean(np.power(10 * np.log10(ber_1 / ber_2), 2))

    frames = range(len(ber_1))
    time_axis = librosa.frames_to_time(frames, hop_length=hop_size, sr=sr)

    plt.figure()
    plt.title('Evolution of Band Energy Ratio over time')
    plt.plot(time_axis, ber_1, label=f'{file_name_1}', linewidth=1)
    plt.plot(time_axis, ber_2, label=f'{file_name_2}', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Band Energy Ratio')
    plt.legend()
    plt.grid()
    plt.show()

    return ber_distance


def get_band_energy_ratio(stft, split_at_bin):

    power_spectrogram = np.power(np.abs(stft), 2)
    power_spectrogram = power_spectrogram.T

    band_energy_ratio = []

    for freq in power_spectrogram:
        ber_current_frame = np.sum(freq[:split_at_bin]) / np.sum(freq[split_at_bin:])
        band_energy_ratio.append(ber_current_frame)

    return np.array(band_energy_ratio)


def visualise_spectra(audio_buffer_1, audio_buffer_2, file_name_1, file_name_2, power_spectrum_1, power_spectrum_2,
                      frame_size, hop_size, sr):

    spectral_centroid_distance = stft_and_spectral_centroid(audio_buffer_1,
                                                            audio_buffer_2,
                                                            file_name_1,
                                                            file_name_2,
                                                            frame_size, hop_size, sr)

    power_spectra(power_spectrum_1, power_spectrum_2, sr)

    coherence(audio_buffer_1, audio_buffer_2, frame_size, hop_size, sr)

    ber_distance = band_energy_ratio(audio_buffer_1, audio_buffer_2, file_name_1, file_name_2, frame_size, hop_size, sr)

    mfcc_euclidean_distance = get_mfcc_euclidean_distance(audio_buffer_1, audio_buffer_2, sr)

    return mfcc_euclidean_distance, spectral_centroid_distance, ber_distance


if __name__ == '__main__':

    # Parsing command line arguments
    parser = ArgumentParser(description='Computes log-spectral distance between two audio files. Optionally computes '
                                        'and visualises other distance measures based on extracting audio spectral '
                                        'features. Note: Requires Python 3')
    parser.add_argument('--audio_path_1', help='Absolute path to 1st audio file', type=str, required=True)
    parser.add_argument('--audio_path_2', help='Absolute path to 2nd audio file', type=str, required=True)
    parser.add_argument('--other_measures', help='If provided, computes additional distance measures and visualises '
                                                 'spectra', type=str, required=False)

    args = parser.parse_args()
    audio_path_1 = args.audio_path_1
    audio_path_2 = args.audio_path_2
    file_name_1 = audio_path_1.split('/')[-1]
    file_name_2 = audio_path_2.split('/')[-1]

    # Load audio files into buffers
    audio_buffer_1, sr = librosa.load(audio_path_1, sr=44100, mono=True)
    audio_buffer_2, _ = librosa.load(audio_path_2, sr=44100, mono=True)

    # Parameters for computing and visualising spectra
    frame_size = 4096
    hop_size = frame_size // 4

    # Check if audio buffer sizes match
    if audio_buffer_1.shape[0] != audio_buffer_2.shape[0]:

        num_samples_to_load = min(len(audio_buffer_1), len(audio_buffer_2))

        audio_buffer_1 = audio_buffer_1[0:num_samples_to_load]
        audio_buffer_2 = audio_buffer_2[0:num_samples_to_load]

        print(f'\nWarning: Audio buffer sizes do not match, only loading first {num_samples_to_load / sr} seconds '
              f'from both files')

    # Call the LSD function
    log_spectral_distance, power_spectrum_1, power_spectrum_2 = get_log_spectral_distance(audio_buffer_1, audio_buffer_2)

    # Print Log Spectral Distance
    print(f'\nLog Spectral Distance = {log_spectral_distance} \n')

    # Generate spectral features and distances and visualise them
    if args.other_measures == 'y':

        mfcc_euclidean_distance, spectral_centroid_distance, ber_distance = visualise_spectra(audio_buffer_1,
                                                                                              audio_buffer_2,
                                                                                              file_name_1,
                                                                                              file_name_2,
                                                                                              power_spectrum_1,
                                                                                              power_spectrum_2,
                                                                                              frame_size, hop_size, sr)

        # Print other distances
        print(f'MFCC Euclidean Distance = {mfcc_euclidean_distance} \n')
        print(f'Spectral Centroid Distance = {spectral_centroid_distance} \n')
        print(f'Band Energy Ratio Distance = {ber_distance} \n')
