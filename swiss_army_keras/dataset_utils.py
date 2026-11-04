from codecs import ignore_errors
from swiss_army_keras import __path__ as mypath
from PIL import ImageFont
from PIL import ImageDraw
import pathlib
import os
from base64 import b64encode
import cv2
from functools import partial

import albumentations as albu

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread

from tqdm import tqdm

from multiprocessing import Process, cpu_count

import threading

from IronDomo import IDPBroker, IDPAsyncClient, IDPWorker

import logging

import dill

from PIL import Image

import time

# helper function for data visualization


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class BrokerProcess(Process):
    class Autorizer(object):
        db = None

        def __init__(self):
            db = None

        def callback(self, domain, key):
            logging.warning('Autorizing: {0}, {1}'.format(domain, key))
            return True

    def __init__(self, clear_url='tcp://127.0.0.1:4445', curve_url='tcp://127.0.0.1:4446'):
        super(BrokerProcess, self).__init__()
        self.clear_url = clear_url
        self.curve_url = curve_url

    def run(self):
        autorizer = BrokerProcess.Autorizer()
        server_public = b"P+S690P{iVPfx<aFJwxfSY^ugFzjuWOnaIh!o7J<"
        server_secret = b":$80ST.hxA5xL7c+@$3YTEohOR^GhrJ2$qzN@bR^"
        self.broker = IDPBroker.IronDomoBroker(self.clear_url, self.curve_url, publisher_connection_string=None, verbose=False, credentials=(
            server_public, server_secret), credentialsCallback=autorizer)
        self.broker.bind()
        self.broker.mediate()


class WorkerProcess(Process):
    class Workload(object):
        pre = None

        def __init__(self, count, augmentations_serialized, width, height, output_width, output_height, mode='segmentation'):
            self.count = count
            self.augmentations = {}
            self.augmentations['train'] = dill.loads(
                augmentations_serialized['train'])
            self.augmentations['val'] = dill.loads(
                augmentations_serialized['val'])
            self.augmentations['test'] = dill.loads(
                augmentations_serialized['test'])
            self.width = width
            self.height = height
            self.output_width = output_width
            self.output_height = output_height

            self.mode = mode

        def do(self, request):

            dset = request[0].decode()
            img = np.frombuffer(request[1], dtype=np.uint8).reshape(
                (self.width, self.height, 3))
            # `idx` tags which in-flight request this is, set by the client
            # (see `aug_function_parallel`) and echoed back unchanged here.
            # The client sends a whole batch's worth of requests before
            # receiving any replies, and replies can come back out of order
            # (different workers finish at different times) - without a tag
            # to match a reply to its request, the client has to assume
            # replies arrive in send order, which is not guaranteed and both
            # silently mismatches image/label pairs and (rarely, under a
            # vendored ZMQ router identity-matching race) corrupts a reply's
            # buffer length, crashing `np.frombuffer(...)` downstream with
            # `ValueError: setting an array element with a sequence` (seen
            # crashing real notebook runs; plan.md sec.10/sec.13 Phase 8).
            idx = request[3]

            if self.mode == 'segmentation':

                lbl = np.frombuffer(request[2], dtype=np.uint8).reshape(
                    (self.output_width, self.output_height))
                try:
                    data = {'image': img, 'mask': lbl}

                    aug_data = self.augmentations[dset](**data)
                    return [aug_data['image'].data, aug_data['mask'].data, idx]
                except Exception as e:
                    logging.error(f'Augmentation error: {e}')
                    return [img.data, lbl.data, idx]
            elif self.mode == 'classification':

                data = {'image': img}
                label = request[2]

                try:
                    aug_data = self.augmentations[dset](**data)
                    return [aug_data['image'].data, label, idx]
                except Exception as e:
                    logging.error(f'Augmentation error: {e}')
                    return [img.data, label, idx]

    def __init__(self, count, augmentations_serialized, width, height, output_width, output_height, clear_url='tcp://127.0.0.1:4445', service='augmentation', mode='segmentation'):
        self.count = count
        self.augmentations_serialized = augmentations_serialized
        self.width = width
        self.height = height
        self.output_width = output_width
        self.output_height = output_height
        self.clear_url = clear_url
        self.service = service
        self.mode = mode
        super(WorkerProcess, self).__init__()

    def run(self):
        import random as ra
        import numpy as np
        seed = (os.getpid() * int(time.time())) % 123456789
        ra.seed(seed)
        np.random.seed(seed)

        workload = WorkerProcess.Workload(
            self.count, self.augmentations_serialized, self.width, self.height, self.output_width, self.output_height, mode=self.mode)
        self.worker = IDPWorker.IronDomoWorker(
            self.clear_url, self.service.encode(), False, workload=workload, identity=f'{self.service}_{self.count}')
        self.worker.loop()


class SegmentationAlbumentationsDataLoader:

    def __init__(self, dataset_path, precache=False, train_augmentations=None, val_augmentations=None, test_augmentations=None, images_dir='images', masks_dir='annotations', resize=True, width=512, height=512, batch_size=16, num_classes=2, mask_downsample=1, train_val_test_split=[0.8, 0.1, 0.1], buffer_size=4, label_shift=0, normalization=(0, 1), dinamic_range=255, ignore_errors=False):

        self.ids = os.listdir(os.path.join(dataset_path, images_dir))
        self.num_classes = num_classes
        # self.mask_ids = os.listdir(masks_dir)
        images_frames = [os.path.join(dataset_path, images_dir, image_id)
                         for image_id in self.ids]
        labels_frames = [os.path.join(dataset_path, masks_dir, image_id.split(
            image_id.split('.')[-1])[0]+'png') for image_id in self.ids]

        self.precache = precache

        self.resize = resize

        self.configs = {}
        self.images = images_frames
        self.labels = labels_frames
        self.width = width
        self.height = height
        self.output_width = int(width/mask_downsample)
        self.output_height = int(height/mask_downsample)

        self.buffer_size = buffer_size

        self.train_val_test_split = []

        for val in train_val_test_split:
            self.train_val_test_split.append(int(len(self.images)*val))

        self.batch_size = batch_size

        self.label_shift = label_shift

        self.mask_downsample = mask_downsample

        self.augmentations = {}

        self.augmentations['train'] = train_augmentations if train_augmentations else self.get_default_augmentation()
        self.augmentations['val'] = val_augmentations if val_augmentations else self.get_default_augmentation()
        self.augmentations['test'] = test_augmentations if test_augmentations else self.get_default_augmentation()

        self.augmentations_serialized = {}
        self.augmentations_serialized['train'] = dill.dumps(
            self.augmentations['train'])
        self.augmentations_serialized['val'] = dill.dumps(
            self.augmentations['val'])
        self.augmentations_serialized['test'] = dill.dumps(
            self.augmentations['test'])

        self.datasets = {}

        self.datasets['train'] = None
        self.datasets['val'] = None
        self.datasets['test'] = None

        self.normalization = normalization
        self.dinamic_range = dinamic_range

        if (isinstance(self.normalization, tuple)):
            logging.warn(f'Normalizing in range: {self.normalization}')
            self.normalized_dynamic_range = self.normalization
        else:
            # Probe with a real 3-channel (RGB) pixel at each endpoint, not
            # a single (1,1,1,2) array whose last axis stood in for "the
            # two probe points" - that shape only worked for a
            # channel-count-agnostic elementwise `normalization` (e.g. the
            # old `x/255`-style placeholders). A real per-channel
            # `(x-mean)/std` normalization (RepViT/FastViT/MobileOne/
            # MobileNetV4's real ImageNet preprocessing, plan.md §12) needs
            # exactly 3 channels and raised `ValueError: operands could not
            # be broadcast together with shapes (1,1,1,2) (3,)` here.
            # `.mean()` collapses the (up to) 3 slightly different
            # per-channel outputs to the single scalar this attribute is
            # actually used as everywhere downstream (display rescaling).
            mn = self.normalization(np.zeros((1, 1, 1, 3))).flatten().mean()
            mx = self.normalization(
                np.full((1, 1, 1, 3), self.dinamic_range, dtype=np.float64)).flatten().mean()
            logging.warn(f'Normalizing in range: {(mn, mx)}')
            self.normalized_dynamic_range = (mn, mx)

        self.assert_dataset()

        randpart = b64encode(os.urandom(5), altchars=b'01').decode()

        self.socket_clear = f'ipc:///tmp/swiss_army_keras_segmentation_clear_socket_{randpart}'
        self.socket_curve = f'ipc:///tmp/swiss_army_keras_segmentation_curve_socket_{randpart}'

        self.broker = BrokerProcess(
            clear_url=self.socket_clear, curve_url=self.socket_curve)
        self.broker.daemon = True
        self.broker.start()

        workers = []
        # for i in range(4):
        for i in range(int(cpu_count()/2)):
            p = WorkerProcess(i, self.augmentations_serialized, self.width,
                              self.height, self.output_width, self.output_height,
                              clear_url=self.socket_clear)
            p.daemon = True
            p.start()

            workers.append(p)

        self.client = IDPAsyncClient.IronDomoAsyncClient(
            self.socket_clear, False, identity="DatasetLoader")
        # `self.client` is one ZMQ DEALER socket shared by the train/val/test
        # `tf.data` pipelines built from this loader. Each pipeline keeps its
        # own background prefetch thread alive for the lifetime of the
        # iterator (not just while actively consumed), so e.g. train's
        # prefetch thread can still be mid-round-trip when validation starts
        # pulling from val's pipeline on another thread - both call
        # `aug_function_parallel` concurrently against the same socket, which
        # is not safe for concurrent send/recv from multiple threads. That
        # interleaving corrupted replies (wrong buffer picked up by the wrong
        # thread), crashing `np.frombuffer(...)` downstream with `ValueError:
        # setting an array element with a sequence` (plan.md §10/§13 Phase 8).
        # Serializing each full batch round-trip fixes it.
        self._client_lock = threading.Lock()

        self.ignore_errors = ignore_errors

        self._warm_up_workers()

    def _warm_up_workers(self, attempts=15, delay=1.0):
        """The broker + worker pool spawned above takes a moment to finish
        starting; the very first real requests can race a worker/broker that
        hasn't finished connecting yet, coming back with a corrupted-length
        reply (`ValueError: setting an array element with a sequence` deep in
        `aug_function_parallel`) - rare but reproducible, seen crashing real
        notebook runs (plan.md sec.10/sec.13 Phase 8). A single probe request
        isn't enough: the broker round-robins across the whole worker pool, so
        one lucky reply from an already-ready worker doesn't prove the other
        (still-starting) workers are ready too. Probe the whole pool at once
        each attempt (one request per worker), before this loader is ever
        handed to a `tf.data` pipeline, retrying every worker with a fresh
        client connection until a full round is clean.
        """
        num_workers = max(1, int(cpu_count() / 2))
        dummy_img = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        dummy_mask = np.zeros(
            (self.output_width, self.output_height), dtype=np.uint8)
        for attempt in range(attempts):
            try:
                for i in range(num_workers):
                    idx_buf = np.array([i], dtype=np.int32)
                    self.client.send(b'augmentation', [
                                     b'val', dummy_img.data, dummy_mask.data, idx_buf.data])
                for _ in range(num_workers):
                    reply = self.client.recv()
                    if reply is None or len(reply) < 3:
                        raise RuntimeError(
                            'worker pool not ready yet (no/short reply)')
                    got_img = np.frombuffer(reply[0], dtype=np.uint8)
                    got_mask = np.frombuffer(reply[1], dtype=np.uint8)
                    got_idx = np.frombuffer(reply[2], dtype=np.int32)
                    if got_img.size != dummy_img.size or got_mask.size != dummy_mask.size or got_idx.size != 1:
                        raise RuntimeError(
                            'worker pool not ready yet (malformed reply)')
                return
            except Exception:
                self.client.reconnect_to_broker()
                time.sleep(delay)
        raise RuntimeError(
            'Data loader worker pool did not become ready in time')

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_class_weights(self):
        h = None
        for f in tqdm(self.labels):
            m = Image.open(f).resize(
                (self.width, self.height), resample=Image.NEAREST)
            if h is None:
                h, _ = np.histogram(m, self.num_classes, (0, self.num_classes))
            else:
                hist, _ = np.histogram(
                    m, self.num_classes, (0, self.num_classes))
                h = hist + h
        h = 1/h
        h = h/sum(h)
        return h

    def get_default_augmentation(self):
        # transform = [albu.Resize(
        #    self.width, self.height, cv2.INTER_CUBIC, p=1), albu.HorizontalFlip(p=0.5), ]
        #transform = albu.Compose(transform)
        transform = None
        return transform

    def assert_dataset(self):
        assert len(self.images) == len(self.labels)
        print('Train Images are good to go')

    def __len__(self):
        return len(self.images)

    def aug_function(self, image, mask, dset):

        data = {"image": image, 'mask': mask}

        aug_data = self.augmentations[dset.decode()](**data)

        aug_img = aug_data["image"]

        aug_msk = aug_data["mask"]  # [:, :, 0]

        return aug_img, aug_msk.astype(np.uint8)

    def aug_function_parallel(self, images, masks, dset):

        n = images.shape[0]
        img_size = self.width * self.height * 3
        mask_size = self.output_width * self.output_height

        with self._client_lock:
            for i in range(n):
                idx_buf = np.array([i], dtype=np.int32)
                self.client.send(b'augmentation', [
                                 dset,  images[i].data, masks[i].data, idx_buf.data])

            aug_img = np.empty_like(images)
            aug_msk = np.empty_like(masks)

            # Replies aren't guaranteed to arrive in send order (different
            # workers finish at different times), and can rarely come back
            # corrupted (see the note in `WorkerProcess.Workload.do`). Each
            # reply carries back the index of the request it answers, so
            # match replies to slots by that index instead of by arrival
            # order, and validate every reply's shape before trusting it -
            # silently retrying a malformed/misdirected one instead of
            # crashing the whole training run.
            pending = set(range(n))
            budget = n * 4
            while pending and budget > 0:
                budget -= 1
                aug_data = self.client.recv()
                if aug_data is None or len(aug_data) < 3:
                    continue
                try:
                    idx_arr = np.frombuffer(aug_data[2], dtype=np.int32)
                    if idx_arr.size != 1:
                        continue
                    idx = int(idx_arr[0])
                    if idx not in pending:
                        continue
                    got_img = np.frombuffer(aug_data[0], dtype=np.uint8)
                    got_msk = np.frombuffer(aug_data[1], dtype=np.uint8)
                    if got_img.size != img_size or got_msk.size != mask_size:
                        continue
                except Exception:
                    continue
                aug_img[idx] = got_img.reshape(self.width, self.height, 3)
                aug_msk[idx] = got_msk.reshape(
                    self.output_width, self.output_height)
                pending.discard(idx)

            if pending:
                raise RuntimeError(
                    f'aug_function_parallel: {len(pending)} sample(s) never '
                    f'got a valid reply: {sorted(pending)}')

        return aug_img, aug_msk.astype(np.uint8)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.uint8), tf.TensorSpec(None, tf.uint8), tf.TensorSpec(None, tf.string)])
    def augment_data(self, image, label, dset):

        aug_img, aug_msk = tf.numpy_function(func=self.aug_function_parallel, inp=[
                                             image, label, dset], Tout=[tf.uint8, tf.uint8])
        # tf.cast(image, np.uint8), tf.cast(label, np.uint8), dset], Tout=[tf.float32, tf.uint8])

        return aug_img, aug_msk

    def open_images(self, image, label):
        img = tf.image.decode_jpeg(tf.io.read_file(image), channels=3)

        lbl = tf.image.decode_png(tf.io.read_file(label), channels=1)
        
        def fmaker(m):
            def f(): 
                # tf.print(f'Resizing: {m}')
                return tf.cast(tf.image.resize(
                    img, [self.height, self.width], method=m), np.uint8)
            return f


        img = tf.switch_case(
            tf.random.uniform(shape=[], dtype=tf.int32, maxval=4), 
            {
                0: fmaker('nearest'),
                1: fmaker('bilinear'),
                2: fmaker('bicubic'),
                3: fmaker('area'),
            }, 
            default=None, 
            name='switch_case'
        )


        lbl = tf.cast(tf.image.resize(
            lbl, [int(self.height/self.mask_downsample), int(self.width/self.mask_downsample)], antialias=False, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), np.uint8)[:, :, 0]

        return img, lbl

    def set_shapes(self, img, label):

        img.set_shape((self.width, self.height, 3))

        label.set_shape((int(self.width/self.mask_downsample),
                        int(self.height/self.mask_downsample)))

        return img, label

    def normalize(self, img, label):
        img = tf.cast(img, tf.float32)

        if (isinstance(self.normalization, tuple)):
            img = (self.normalization[1] - self.normalization[0]
                   )*img/self.dinamic_range + self.normalization[0]
        else:
            img = self.normalization(img)

        img.set_shape((self.batch_size, self.width, self.height, 3))

        label = tf.cast(tf.one_hot(tf.cast(label-self.label_shift, tf.uint8),
                                   self.num_classes), tf.float32)

        label.set_shape((self.batch_size,  int(self.width/self.mask_downsample),
                        int(self.height/self.mask_downsample), self.num_classes))
        return img, label

    def prepare_dataset(self, dataset, dset):

        # Open and resize images
        dataset = dataset.map(
            self.open_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if(self.ignore_errors):
            dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        if self.precache:
            dataset = dataset.cache()
            for _ in tqdm(dataset):
                pass
        # augment data
        dataset = dataset.map(
            self.set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(
            self.batch_size*self.buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
        if self.augmentations[dset] is not None:
            dataset = dataset.map(partial(self.augment_data, dset=dset),
                                  num_parallel_calls=1)  # tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.images, self.labels)
        )
        dataset = dataset.shuffle(
            len(self.images), reshuffle_each_iteration=False)
        train_dataset = dataset.take(self.train_val_test_split[0])
        val_dataset = dataset.skip(self.train_val_test_split[0]).take(
            self.train_val_test_split[1])
        test_dataset = dataset.skip(self.train_val_test_split[0]).skip(
            self.train_val_test_split[1])

        train_dataset = self.prepare_dataset(train_dataset, 'train')
        val_dataset = self.prepare_dataset(val_dataset, 'val')
        test_dataset = self.prepare_dataset(test_dataset, 'test')

        self.datasets['train'] = train_dataset
        self.datasets['val'] = val_dataset
        self.datasets['test'] = test_dataset

        return train_dataset, val_dataset, test_dataset

    def show_images(self,  num_images=4, dset='train',):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = (res[0] - self.normalized_dynamic_range[0]) * \
            255.0 / \
            (self.normalized_dynamic_range[1] -
             self.normalized_dynamic_range[0])

        label = res[1]

        for i in range(num_images):

            visualize(
                image=tf.cast(images[i], tf.uint8),
                mask=np.argmax(label[i], axis=-1)*255,
            )

    def show_results(self, model, num_images=4, dset='test', output=None):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = res[0]
        labels = res[1]

        preds = model.predict([images])

        if output is not None:
            preds = preds[output]

        for i in range(num_images):
            image = (images[i] - self.normalized_dynamic_range[0]) * \
                255.0 / \
                (self.normalized_dynamic_range[1] -
                 self.normalized_dynamic_range[0])
            visualize(
                image=tf.cast(image, tf.uint8),
                predicted_mask=np.argmax(preds[i], axis=-1)*255,
                reference_mask=np.argmax(labels[i], axis=-1)*255,
            )


class ClassificationAlbumentationsDataLoader:

    def __init__(self, dataset_path, precache=False, train_augmentations=None, val_augmentations=None, test_augmentations=None, resize=True, width=512, height=512, batch_size=16, train_val_test_split=[0.8, 0.1, 0.1], buffer_size=4, normalization=(0, 1), dinamic_range=255, ignore_errors=False):

        self.data_root = pathlib.Path(dataset_path)

        self.path_parts = len(dataset_path.split(os.sep))

        self.resize = resize

        self.subfolders = [f.path.split(
            '/')[-1] for f in os.scandir(str(self.data_root)) if f.is_dir()]
        self.subfolders.sort()

        filelist = []
        for sf in self.subfolders:
            for dirpath, dnames, fnames in os.walk(f'{dataset_path}/{sf}'):
                for f in fnames:
                    filelist.append(os.path.join(dirpath, f))

        self.files_ds = tf.data.Dataset.from_tensor_slices(filelist)

        self.classes_map = {}
        self.reverse_classes_map = {}

        cnt = 0
        for f in self.subfolders:
            self.classes_map[f] = cnt
            self.reverse_classes_map[cnt] = f
            print(f'Class label {f} will have id: {cnt}')
            cnt = cnt + 1

        self.num_classes = len(self.classes_map.keys())

        # build a lookup table
        self.lookup_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self.classes_map.keys()),
                values=list(self.classes_map.values()),
            ),
            default_value=tf.constant(-1),
            name="class_weight"
        )

        # build a lookup table
        self.reverse_lookup_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self.classes_map.values()),
                values=list(self.classes_map.keys()),
            ),
            default_value=tf.constant("unknown"),
            name="class_weight"
        )

        self.precache = precache

        self.configs = {}

        self.width = width
        self.height = height

        self.buffer_size = buffer_size

        self.train_val_test_split = []

        for val in train_val_test_split:
            self.train_val_test_split.append(int(len(self.files_ds)*val))

        print(f'train_val_test_split: {self.train_val_test_split}')

        self.batch_size = batch_size

        self.augmentations = {}

        self.augmentations['train'] = train_augmentations if train_augmentations else self.get_default_augmentation()
        self.augmentations['val'] = val_augmentations if val_augmentations else self.get_default_augmentation()
        self.augmentations['test'] = test_augmentations if test_augmentations else self.get_default_augmentation()

        self.augmentations_serialized = {}
        self.augmentations_serialized['train'] = dill.dumps(
            self.augmentations['train'])
        self.augmentations_serialized['val'] = dill.dumps(
            self.augmentations['val'])
        self.augmentations_serialized['test'] = dill.dumps(
            self.augmentations['test'])

        self.datasets = {}

        self.datasets['train'] = None
        self.datasets['val'] = None
        self.datasets['test'] = None

        self.normalization = normalization
        self.dinamic_range = dinamic_range

        if (isinstance(self.normalization, tuple)):
            logging.warn(f'Normalizing in range: {self.normalization}')
            self.normalized_dynamic_range = self.normalization
        else:
            # See the identical fix/comment in
            # `SegmentationAlbumentationsDataLoader.__init__` - a real
            # 3-channel probe, not a (1,1,1,2) array, since a real
            # per-channel `normalization` (RepViT/FastViT/MobileOne/
            # MobileNetV4, plan.md §12) needs exactly 3 channels.
            mn = self.normalization(np.zeros((1, 1, 1, 3))).flatten().mean()
            mx = self.normalization(
                np.full((1, 1, 1, 3), self.dinamic_range, dtype=np.float64)).flatten().mean()
            logging.warn(f'Normalizing in range: {(mn, mx)}')
            self.normalized_dynamic_range = (mn, mx)

        randpart = b64encode(os.urandom(5), altchars=b'01').decode()

        self.socket_clear = f'ipc:///tmp/swiss_army_keras_classification_clear_socket_{randpart}'
        self.socket_curve = f'ipc:///tmp/swiss_army_keras_classification_curve_socket_{randpart}'

        self.broker = BrokerProcess(
            clear_url=self.socket_clear, curve_url=self.socket_curve)
        self.broker.daemon = True
        self.broker.start()

        workers = []
        # for i in range(4):
        for i in range(int(cpu_count()/2)):
            p = WorkerProcess(i, self.augmentations_serialized,
                              self.width, self.height,
                              0, 0,
                              clear_url=self.socket_clear,
                              mode='classification')
            p.daemon = True
            p.start()

            workers.append(p)

        self.client = IDPAsyncClient.IronDomoAsyncClient(
            self.socket_clear, False, identity="DatasetLoader")
        # `self.client` is one ZMQ DEALER socket shared by the train/val/test
        # `tf.data` pipelines built from this loader. Each pipeline keeps its
        # own background prefetch thread alive for the lifetime of the
        # iterator (not just while actively consumed), so e.g. train's
        # prefetch thread can still be mid-round-trip when validation starts
        # pulling from val's pipeline on another thread - both call
        # `aug_function_parallel` concurrently against the same socket, which
        # is not safe for concurrent send/recv from multiple threads. That
        # interleaving corrupted replies (wrong buffer picked up by the wrong
        # thread), crashing `np.frombuffer(...)` downstream with `ValueError:
        # setting an array element with a sequence` (plan.md §10/§13 Phase 8).
        # Serializing each full batch round-trip fixes it.
        self._client_lock = threading.Lock()

        self.ignore_errors = ignore_errors

        self._warm_up_workers()

    def _warm_up_workers(self, attempts=15, delay=1.0):
        """The broker + worker pool spawned above takes a moment to finish
        starting; the very first real requests can race a worker/broker that
        hasn't finished connecting yet, coming back with a corrupted-length
        reply (`ValueError: setting an array element with a sequence` deep in
        `aug_function_parallel`) - rare but reproducible, seen crashing real
        notebook runs (plan.md sec.10/sec.13 Phase 8). A single probe request
        isn't enough: the broker round-robins across the whole worker pool, so
        one lucky reply from an already-ready worker doesn't prove the other
        (still-starting) workers are ready too. Probe the whole pool at once
        each attempt (one request per worker), before this loader is ever
        handed to a `tf.data` pipeline, retrying every worker with a fresh
        client connection until a full round is clean.
        """
        num_workers = max(1, int(cpu_count() / 2))
        dummy_img = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        dummy_label = np.zeros(1, dtype=np.int32)
        for attempt in range(attempts):
            try:
                for i in range(num_workers):
                    idx_buf = np.array([i], dtype=np.int32)
                    self.client.send(b'augmentation', [
                                     b'val', dummy_img.data, dummy_label.data, idx_buf.data])
                for _ in range(num_workers):
                    reply = self.client.recv()
                    if reply is None or len(reply) < 3:
                        raise RuntimeError(
                            'worker pool not ready yet (no/short reply)')
                    got_img = np.frombuffer(reply[0], dtype=np.uint8)
                    got_label = np.frombuffer(reply[1], dtype=np.int32)
                    got_idx = np.frombuffer(reply[2], dtype=np.int32)
                    if got_img.size != dummy_img.size or got_label.size != 1 or got_idx.size != 1:
                        raise RuntimeError(
                            'worker pool not ready yet (malformed reply)')
                return
            except Exception:
                self.client.reconnect_to_broker()
                time.sleep(delay)
        raise RuntimeError(
            'Data loader worker pool did not become ready in time')

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_class_weights(self):
        h = None
        for f in tqdm(self.labels):
            m = Image.open(f).resize(
                (self.width, self.height), resample=Image.NEAREST)
            if h is None:
                h, _ = np.histogram(m, self.num_classes, (0, self.num_classes))
            else:
                hist, _ = np.histogram(
                    m, self.num_classes, (0, self.num_classes))
                h = hist + h
        h = 1/h
        h = h/sum(h)
        return h

    def get_default_augmentation(self):
        # transform = [albu.Resize(
        #    self.width, self.height, cv2.INTER_CUBIC, p=1), albu.HorizontalFlip(p=0.5), ]
        #transform = albu.Compose(transform)
        transform = None
        return transform

    def __len__(self):
        return len(self.files_ds)

    def aug_function_parallel(self, images, labels, dset):

        # `labels[i]` is a 0-dimensional numpy scalar (labels is shape
        # (batch,)); its `.data` memoryview has shape () and does not
        # behave like a normal fixed-length buffer across the
        # multiprocess/ZMQ round-trip (it doesn't even support `len()`
        # directly) - slicing to a proper 1-d, 1-element array first gives
        # a well-defined 4-byte buffer (plan.md sec.10/sec.13 Phase 8).
        n = images.shape[0]
        img_size = self.width * self.height * 3

        with self._client_lock:
            for i in range(n):
                idx_buf = np.array([i], dtype=np.int32)
                self.client.send(b'augmentation', [
                                 dset, images[i].data, labels[i:i+1].data, idx_buf.data])

            aug_img = np.empty_like(images)
            aug_label = np.empty_like(labels)

            # Replies aren't guaranteed to arrive in send order (different
            # workers finish at different times), and can rarely come back
            # corrupted (see the note in `WorkerProcess.Workload.do`). Each
            # reply carries back the index of the request it answers, so
            # match replies to slots by that index instead of by arrival
            # order, and validate every reply's shape before trusting it -
            # silently retrying a malformed/misdirected one instead of
            # crashing the whole training run.
            pending = set(range(n))
            budget = n * 4
            while pending and budget > 0:
                budget -= 1
                aug_data = self.client.recv()
                if aug_data is None or len(aug_data) < 3:
                    continue
                try:
                    idx_arr = np.frombuffer(aug_data[2], dtype=np.int32)
                    if idx_arr.size != 1:
                        continue
                    idx = int(idx_arr[0])
                    if idx not in pending:
                        continue
                    got_img = np.frombuffer(aug_data[0], dtype=np.uint8)
                    got_label = np.frombuffer(aug_data[1], dtype=np.int32)
                    if got_img.size != img_size or got_label.size != 1:
                        continue
                except Exception:
                    continue
                aug_img[idx] = got_img.reshape(self.width, self.height, 3)
                # `got_label` is shape (1,) (one int32 read back from a
                # 4-byte buffer); `aug_label[idx]` is a scalar (0-d) slot.
                # Assigning a shape-(1,) array directly into a 0-d slot
                # raises `ValueError: setting an array element with a
                # sequence` in this numpy version, which is what was
                # actually crashing every notebook run - not corruption on
                # the wire (plan.md sec.10/sec.13 Phase 8). Extract the
                # scalar explicitly instead of relying on broadcasting.
                aug_label[idx] = got_label.item()
                pending.discard(idx)

            if pending:
                raise RuntimeError(
                    f'aug_function_parallel: {len(pending)} sample(s) never '
                    f'got a valid reply: {sorted(pending)}')

        return aug_img, aug_label

    @tf.function(input_signature=[tf.TensorSpec(None, tf.uint8), tf.TensorSpec(None, tf.int32), tf.TensorSpec(None, tf.string)])
    def augment_data(self, image, label, dset):

        aug_img, aug_label = tf.numpy_function(func=self.aug_function_parallel, inp=[
            image, label, dset], Tout=[tf.uint8, tf.int32])

        return aug_img, aug_label

    def open_images(self, image, label):
        img = tf.image.decode_jpeg(tf.io.read_file(image), channels=3)

        def fmaker(m):
            def f(): 
                # tf.print(f'Resizing: {m}')
                return tf.cast(tf.image.resize(
                    img, [self.height, self.width], method=m), np.uint8)
            return f


        img = tf.switch_case(
            tf.random.uniform(shape=[], dtype=tf.int32, maxval=4), 
            {
                0: fmaker('nearest'),
                1: fmaker('bilinear'),
                2: fmaker('bicubic'),
                3: fmaker('area'),
            }, 
            default=None, 
            name='switch_case'
        )

        return img, label

    def set_shapes(self, img, label):

        img.set_shape((self.width, self.height, 3))

        return img, label

    def normalize(self, img, label):
        img = tf.cast(img, tf.float32)

        if (isinstance(self.normalization, tuple)):
            img = (self.normalization[1] - self.normalization[0]
                   )*img/self.dinamic_range + self.normalization[0]
        else:
            img = self.normalization(img)

        img.set_shape((self.batch_size, self.width, self.height, 3))

        label = tf.cast(tf.one_hot(tf.cast(label, tf.uint8),
                                   self.num_classes), tf.float32)

        label.set_shape((self.batch_size, self.num_classes))

        return img, label

    def prepare_dataset(self, dataset, dset):

        # Open and resize images
        dataset = dataset.map(
            self.open_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if(self.ignore_errors):
            dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        if self.precache:
            dataset = dataset.cache()
            for _ in tqdm(dataset):
                pass
        # augment data
        dataset = dataset.map(
            self.set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(
            self.batch_size*self.buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
        if self.augmentations[dset] is not None:
            dataset = dataset.map(partial(self.augment_data, dset=dset),
                                  num_parallel_calls=1)  # tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_datasets(self):

        dataset = self.files_ds

        def process_path(file_path):
            intermedio = tf.strings.split(file_path, os.sep)[self.path_parts]
            #lbl = tf.cast(self.classes_map[intermedio], tf.float32)
            lbl = tf.cast(self.lookup_table.lookup(intermedio), tf.int32)
            #lbl = tf.strings.split(file_path, os.sep)[-2]
            return file_path, lbl

        dataset = dataset.map(process_path)

        dataset = dataset.shuffle(
            len(self.files_ds), reshuffle_each_iteration=False)
        train_dataset = dataset.take(self.train_val_test_split[0])
        val_dataset = dataset.skip(self.train_val_test_split[0]).take(
            self.train_val_test_split[1])
        test_dataset = dataset.skip(self.train_val_test_split[0]).skip(
            self.train_val_test_split[1])

        train_dataset = self.prepare_dataset(train_dataset, 'train')
        val_dataset = self.prepare_dataset(val_dataset, 'val')
        test_dataset = self.prepare_dataset(test_dataset, 'test')

        self.datasets['train'] = train_dataset
        self.datasets['val'] = val_dataset
        self.datasets['test'] = test_dataset

        return train_dataset, val_dataset, test_dataset

    def show_images(self,  num_images=4, dset='train',):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = (res[0] - self.normalized_dynamic_range[0]) * \
            255.0 / \
            (self.normalized_dynamic_range[1] -
             self.normalized_dynamic_range[0])

        for i in range(num_images):
            image = tf.cast(images[i], tf.uint8)
            pimg = Image.fromarray(image.numpy())
            draw = ImageDraw.Draw(pimg)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype(
                f'{mypath}/DejaVuSansMono.ttf', int(self.height/20))
            # draw.text((x, y),"Sample Text",(r,g,b))

            shape = [(0, 0), (int(self.width/2), int(1.25*self.height/20))]
            draw.rectangle(shape, (0, 0, 0))
            #draw.text((0, 0), str(res[1][i].numpy()),(255,255,255), font=font)
            draw.text((0, 0), self.reverse_classes_map[np.argmax(
                res[1][i].numpy(), axis=-1)], (255, 255, 255), font=font)
            # print(label[i])
            visualize(
                image=pimg,
                #mask=np.argmax(label[i], axis=-1)*255,
            )

    def show_results(self, model, num_images=4, dset='test', output=None):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = res[0]
        labels = res[1]

        preds = model.predict([images])

        if output is not None:
            preds = preds[output]

        fig = plt.figure(figsize=(22, 22))
        for i in range(num_images):
            image = (images[i] - self.normalized_dynamic_range[0]) * \
                255.0 / \
                (self.normalized_dynamic_range[1] -
                 self.normalized_dynamic_range[0])
            image = tf.cast(image, tf.uint8)
            predimg = Image.fromarray(image.numpy())
            refimg = Image.fromarray(image.numpy())
            preddraw = ImageDraw.Draw(predimg)
            refdraw = ImageDraw.Draw(refimg)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype(
                f'{mypath}/DejaVuSansMono.ttf', int(self.height/20))
            # draw.text((x, y),"Sample Text",(r,g,b))

            shape = [(0, 0), (int(self.width/2), int(1.25*self.height/20))]
            preddraw.rectangle(shape, (0, 0, 0))
            refdraw.rectangle(shape, (0, 0, 0))
            #draw.text((0, 0), str(res[1][i].numpy()),(255,255,255), font=font)
            preddraw.text((0, 0), self.reverse_classes_map[np.argmax(
                preds[i], axis=-1)], (255, 255, 255), font=font)
            refdraw.text((0, 0), self.reverse_classes_map[np.argmax(
                labels[i], axis=-1)], (255, 255, 255), font=font)
            # print(label[i])

            visualize(
                predicted=predimg,
                reference=refimg,
            )
