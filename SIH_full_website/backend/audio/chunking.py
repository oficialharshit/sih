######## 4th step ########

'''
this will extract the chunks 
max duration is 10 sec
that is chunks of 2 sec at max

this function will return 
if the len of the audio is 10 or more 
if len is less it will so accordingly
[
    2-second chunk,
    2-second chunk,
    2-second chunk,
    2-second chunk,
    2-second chunk
]
'''

MAX_SPEECH_DURATION = 10


def create_chunks(speech_segments, sample_rate, chunk_duration=2):

    chunk_size = int(sample_rate * chunk_duration)
    max_samples = int(sample_rate * MAX_SPEECH_DURATION)

    chunks = []
    total_samples = 0

    for segment in speech_segments:

        for start in range(0, len(segment), chunk_size):

            if total_samples >= max_samples:
                return chunks

            end = start + chunk_size
            chunk = segment[start:end]

            # Don't exceed 10 seconds total
            remaining = max_samples - total_samples
            chunk = chunk[:remaining]

            # Require minimum 0.75 seconds of audio to prevent zero-padding artifacts
            if len(chunk) >= int(sample_rate * 0.75):
                chunks.append(chunk)
                total_samples += len(chunk)

    return chunks