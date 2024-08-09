import os
from time import time
from datasets import util
import apache_beam as beam
import numpy as np
import tensorflow as tf
def get_dir_size(path='.',unit=None):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    if unit=='MB':
        total=round(total/(10**6))
    else:
        total=total
    return total
if __name__ == '__main__':
    start = time()
    _INPUT_DIR = r'\\10.99.68.178\Saurabh\film_lung_train\lung_train_skip_3\sequences'
    _INTPUT_TRIPLET_LIST_FILEPATH = r'\\10.99.68.178\Saurabh\film_lung_train\lung_train_skip_3\sequences.txt'
    _OUTPUT_TFRECORD_FILEPATH = r'\\10.99.68.178\Saurabh\film_lung_train\lung_train_skip_3\sequences.tfrecord'

    datasetsize = get_dir_size(_INPUT_DIR, 'MB')

    shardsize = 200

    _NUM_SHARDS = datasetsize // shardsize + 1

    print('num shards ', _NUM_SHARDS)

    _INTERPOLATOR_IMAGES_MAP = {
        'frame_0': 'im1.png',
        'frame_1': 'im2.png',
        'frame_2': 'im3.png',
    }
    with tf.io.gfile.GFile(_INTPUT_TRIPLET_LIST_FILEPATH, 'r') as fid:
        triplets_list = np.loadtxt(fid, dtype=str)
    triplet_dicts = []
    for triplet in triplets_list:
        triplet_dict = {
            image_key: os.path.join(_INPUT_DIR, triplet, image_basename)
            for image_key, image_basename in _INTERPOLATOR_IMAGES_MAP.items()
        }
        triplet_dicts.append(triplet_dict)
    print('len triplet_dicts', len(triplet_dicts))
    p = beam.Pipeline('DirectRunner')
    (p | 'ReadInputTripletDicts' >> beam.Create(triplet_dicts)  # pylint: disable=expression-not-assigned
     | 'GenerateSingleExample' >> beam.ParDo(
                util.ExampleGenerator(_INTERPOLATOR_IMAGES_MAP))
     | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=_OUTPUT_TFRECORD_FILEPATH,
                num_shards=_NUM_SHARDS,
                coder=beam.coders.BytesCoder()))
    result = p.run()
    result.wait_until_finish()
    print('Succeeded in creating the output TFRecord file: {},{}'.format(_OUTPUT_TFRECORD_FILEPATH, str(_NUM_SHARDS)))
    print(round(time() - start), 'sec elapsed')