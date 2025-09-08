import os
import pickle
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.cchessAI.parameters import SHOULD_FLIP
from utils.tools import move_action2move_id, flip, move_id2move_action
from threading import Lock


def zip_array(array, data=0.0):
    """
    将数组压缩为稀疏数组格式

    参数:
        array: 二维numpy数组
        data: 需要压缩的值, 默认为0.

    返回:
        压缩后的列表
    """
    rows, cols = array.shape
    zip_res = [[rows, cols]]

    for i in range(rows):
        for j in range(cols):
            if array[i, j] != data:
                zip_res.append([i, j, array[i, j]])

    return zip_res  # 直接返回列表，不转换为numpy数组


def recovery_array(array, data=0.0):
    """
    从稀疏数组恢复为二维数组

    参数:
        array: 压缩后的列表或numpy数组
        data: 填充的默认值, 默认为0.

    返回:
        恢复后的二维numpy数组
    """
    # 将array转换为列表进行操作，确保兼容性
    array_list = array.tolist() if isinstance(array, np.ndarray) else array

    rows, cols = array_list[0]
    recovery_res = np.full((int(rows), int(cols)), data)

    for i in range(1, len(array_list)):
        row_idx = int(array_list[i][0])
        col_idx = int(array_list[i][1])
        recovery_res[row_idx, col_idx] = array_list[i][2]

    return recovery_res



def mirror_data(play_data):
    """
    左右对称变换，扩充数据集一倍，压缩加速训练速度。

    Args:
        play_data (list): 包含(state, mcts_prob, winner)的元组列表

    Returns:
        list: 经过镜像翻转后扩充的数据
    """
    mirror_data = []
    for state, mcts_prob, winner in play_data:
        # 不管是否翻转，原始数据都要压缩
        original_compressed = CompressionUtil.zip_state_mcts_prob((state, mcts_prob, winner))
        mirror_data.append(original_compressed)

        # 如果启用翻转，则生成翻转数据并压缩后加入
        if SHOULD_FLIP:
            state_flip = state.transpose([1, 2, 0])[:, ::-1, :].transpose([2, 0, 1])
            mcts_prob_flip = [
                mcts_prob[move_action2move_id[flip(move_id2move_action[i])]]
                for i in range(len(mcts_prob))
            ]
            flipped_compressed = CompressionUtil.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner))
            mirror_data.append(flipped_compressed)

    return mirror_data





class CompressionUtil:
    """
    pkl 压缩/解压工具类
    """

    @staticmethod
    def zip_state_mcts_prob(tuple):
        state, mcts_prob, winner = tuple
        state = state.reshape((15, -1))
        mcts_prob = mcts_prob.reshape((2, -1))
        state = zip_array(state)
        mcts_prob = zip_array(mcts_prob)
        return state, mcts_prob, winner
    @staticmethod
    def zip_states_mcts_prob(play_data):
        mirror_data = []
        for state, mcts_prob, winner in play_data:
            original_compressed = CompressionUtil.zip_state_mcts_prob((state, mcts_prob, winner))
            mirror_data.append(original_compressed)
        return mirror_data

    @staticmethod
    def recovery_state_mcts_prob(tuple):
        state, mcts_prob, winner = tuple
        state = recovery_array(state)
        mcts_prob = recovery_array(mcts_prob)
        state = state.reshape((15, 10, 9))
        mcts_prob = mcts_prob.reshape(2086)
        return state, mcts_prob, winner
    @staticmethod
    def is_compressed(data_item):
        """
        判断一个数据样本是否是经过 zip_state_mcts_prob 压缩过的格式。

        Args:
            data_item (tuple): 单个训练样本，格式为 (state, mcts_prob, winner)

        Returns:
            bool: 如果是压缩数据返回 True，否则返回 False
        """
        state, mcts_prob, winner = data_item

        if isinstance(state, list) or (isinstance(state, np.ndarray) and state.shape == (15, 90)):
            return True
        elif isinstance(mcts_prob, list) or (isinstance(mcts_prob, np.ndarray) and mcts_prob.shape == (2, 1043)):
            return True
        else:
            return False
class AtomicInteger:
    def __init__(self, initial=0):
        self.value = initial
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value
class DataWriter:
    """
    数据写入工具类，支持多种格式自动识别和多线程写入加速。
    支持格式: HDF5 (.h5/.hdf5), Pickle (.pkl)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data_format = self._detect_format(file_path)
        self.locked = False
        self.max_iters = 0

    def _detect_format(self, path):
        """自动检测目标文件格式"""
        ext = os.path.splitext(path)[-1].lower()
        if ext in ('.h5', '.hdf5'):
            return 'hdf5'
        elif ext == '.pkl':
            return 'pkl'
        else:
            raise ValueError(f"不支持的文件格式：{ext}")

    def _initialize_file_h5(self, compress=True):
        """初始化文件结构（如HDF5）"""
        if compress:
            with h5py.File(self.file_path, 'a') as hf:
                if 'states' not in hf:
                    hf.create_dataset('states', shape=(0, 15, 10, 9),
                                      maxshape=(None, 15, 10, 9), dtype=np.float16,
                                      compression="gzip", compression_opts=3)
                    hf.create_dataset('mcts_probs', shape=(0, 2086),
                                      maxshape=(None, 2086), dtype=np.float16,
                                      compression="gzip", compression_opts=3)
                    hf.create_dataset('winners', shape=(0,),
                                      maxshape=(None,), dtype=np.float16,
                                      compression="gzip", compression_opts=3)
                    hf.create_dataset('iters', shape=(1,), dtype=np.int64)
        else:
            with h5py.File(self.file_path, 'a') as hf:
                if 'states' not in hf:
                    hf.create_dataset('states', shape=(0, 15, 10, 9),
                                      maxshape=(None, 15, 10, 9), dtype=np.float32)
                    hf.create_dataset('mcts_probs', shape=(0, 2086),
                                      maxshape=(None, 2086), dtype=np.float32)
                    hf.create_dataset('winners', shape=(0,),
                                      maxshape=(None,), dtype=np.float32)

                    hf.create_dataset('iters', shape=(1,), dtype=np.int64)
    def write(self, data_list, iters=0, compress=True):
        """
        写入数据，根据格式选择不同的实现
        Args:
            data_list (list): 包含(states, mcts_probs, winner_z)的列表
            compress (bool): 是否压缩数据，默认True
        """
        if self.locked:
            raise RuntimeError("该实例已被锁定写入，请使用新的实例")

        if self.data_format == 'hdf5':
            self._initialize_file_h5(compress=compress)
            self._write_hdf5(data_list, iters)
        elif self.data_format == 'pkl':
            self._write_pickle(data_list, iters, compress=compress)

    def _write_hdf5(self, data_list, iters=0):
        """批量写入 HDF5 格式文件，并自动解压压缩数据"""
        states = []
        probs = []
        winners = []

        for item in data_list:
            if not CompressionUtil.is_compressed(item):
                state, prob, winner = item
            else:
                # 如果是压缩格式，则解压
                state, prob, winner = CompressionUtil.recovery_state_mcts_prob(item)
            states.append(state)
            probs.append(prob)
            winners.append(winner)

        states = np.array(states).astype("float16")
        probs = np.array(probs).astype("float16")
        winners = np.array(winners).astype("float16")

        with h5py.File(self.file_path, 'a') as hf:
            ds_states = hf['states']
            ds_probs = hf['mcts_probs']
            ds_winners = hf['winners']

            # 扩展尺寸
            batch_len = len(states)
            ds_states.resize(ds_states.shape[0] + batch_len, axis=0)
            ds_probs.resize(ds_probs.shape[0] + batch_len, axis=0)
            ds_winners.resize(ds_winners.shape[0] + batch_len, axis=0)

            # 写入数据
            ds_states[-batch_len:] = states
            ds_probs[-batch_len:] = probs
            ds_winners[-batch_len:] = winners

            if 'iters' not in hf:
                hf.create_dataset('iters', shape=(1,), dtype='int64')
            hf['iters'][0] = iters

        print(f"[INFO] 已写入 {batch_len} 条数据至 HDF5 文件")

    def _write_pickle(self, data_buffer, iters=0, compress=True,mode='ab'):
        """追加写入 Pickle 格式文件"""
        if compress:
            data_buffer = CompressionUtil.zip_states_mcts_prob(data_buffer)
        with open(self.file_path, mode) as f:
            pickle.dump({
                "data_buffer": data_buffer,
                "iters": iters
            }, f)
        print(f"[INFO] 已写入 iter={iters}, {len(data_buffer)} 条数据至 Pickle 文件")

    def bulk_write_parallel(self, all_data, num_workers=4):
        """
        多线程并行写入数据，并自动分配 iters。
        Args:
            all_data (list): 所有要写入的数据列表
            num_workers (int): 并发线程数
        """
        chunk_size = len(all_data) // num_workers
        chunks = [
            all_data[i * chunk_size:(i + 1) * chunk_size]
            for i in range(num_workers)
        ]
        # 补充最后一个 chunk
        if len(all_data) % num_workers != 0:
            chunks[-1].extend(all_data[num_workers * chunk_size:])

        iter_counter = AtomicInteger(0)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.write, chunk, iter_counter.increment())
                for chunk in chunks
            ]
            for future in futures:
                future.result()

        print("[INFO] 多线程写入完成")


class DataReader:
    """
    数据读取工具类，支持多种格式自动识别。
    支持格式: HDF5 (.h5/.hdf5), Pickle (.pkl)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data_format = self._detect_format(file_path)

    def _detect_format(self, path):
        """自动检测目标文件格式"""
        ext = os.path.splitext(path)[-1].lower()
        if ext in ('.h5', '.hdf5'):
            return 'hdf5'
        elif ext == '.pkl':
            return 'pkl'
        else:
            raise ValueError(f"不支持的文件格式：{ext}")

    def read_all(self):
        """一次性读取全部数据"""
        if self.data_format == 'hdf5':
            return self._read_hdf5_all()
        elif self.data_format == 'pkl':
            return self._read_pickle_all()

    def read_in_chunks(self, chunk_size=1000):
        """按指定大小分块读取数据"""
        if self.data_format == 'hdf5':
            return self._read_hdf5_in_chunks(chunk_size)
        elif self.data_format == 'pkl':
            return self._read_pickle_in_chunks(chunk_size)

    def _read_hdf5_all(self):
        """读取整个 HDF5 文件内容"""
        with h5py.File(self.file_path, 'r') as hf:
            states = np.array(hf['states'])
            probs = np.array(hf['mcts_probs'])
            winners = np.array(hf['winners'])
            iters = hf['iters'][0]  # 新增读取 iters

        print(f"[INFO] 从 HDF5 读取 {len(states)} 条数据，当前迭代次数: {iters}")
        return list(zip(states, probs, winners)), iters  # 返回数据和 iters


    def _read_hdf5_in_chunks(self, chunk_size=1000):
        """分块读取 HDF5 文件内容"""
        with h5py.File(self.file_path, 'r') as hf:
            total = hf['states'].shape[0]
            iters = hf['iters'][0]  # 获取全局 iters
            for i in range(0, total, chunk_size):
                end = min(i + chunk_size, total)
                states = hf['states'][i:end]
                probs = hf['mcts_probs'][i:end]
                winners = hf['winners'][i:end]
                yield list(zip(states, probs, winners)), iters  # 返回数据块和 iters


    def _read_pickle_all(self):
        """一次性读取整个 Pickle 文件"""
        data_list = []
        with open(self.file_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    data_list.extend(data.get("data_buffer", []))
                except EOFError:
                    break
        print(f"[INFO] 从 Pickle 读取 {len(data_list)} 条数据")
        return data_list,data.get("iters",0)

    def _read_pickle_in_chunks(self, chunk_size=1000):
        """分块读取 Pickle 文件内容"""
        with open(self.file_path, 'rb') as f:
            buffer = []
            while True:
                try:
                    data = pickle.load(f)
                    buffer.extend(data.get("data_buffer", []))
                    if len(buffer) >= chunk_size:
                        yield buffer[:chunk_size],data.get("iters",0)
                        buffer = buffer[chunk_size:]
                except EOFError:
                    if buffer:
                        yield buffer,data.get("iters",0)
                    break

    def parallel_read_chunks(self, chunk_size=1000, num_workers=4):
        """并行读取多个 pickle 文件或分段读取大文件"""
        file_size = os.path.getsize(self.file_path)
        if self.data_format != 'pkl':
            raise NotImplementedError("仅支持 .pkl 格式并行读取")

        # 获取文件偏移位置
        ranges = self._get_file_ranges(file_size, chunk_size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._read_pickle_chunk, start, end) for start, end in ranges]
            for future in futures:
                yield from future.result()

    def _get_file_ranges(self, file_size, chunk_size):
        """获取 pickle 文件中每个 chunk 的起始偏移量"""
        ranges = []
        with open(self.file_path, 'rb') as f:
            offsets = [0]
            while f.tell() < file_size:
                try:
                    f.readline()  # 对于文本pickle可使用
                    if pickle.format_version >= "4.0":
                        pass
                    offsets.append(f.tell())
                except:
                    break
            for i in range(len(offsets) - 1):
                ranges.append((offsets[i], offsets[i + 1]))
            if len(offsets) > 0 and offsets[-1] < file_size:
                ranges.append((offsets[-1], None))
        return ranges

    def _read_pickle_chunk(self, start, end):
        """读取指定范围内的 pickle 数据"""
        chunk_data = []
        with open(self.file_path, 'rb') as f:
            f.seek(start)
            if end is not None:
                data = f.read(end - start)
                buf = pickle.Unpickler(f)
            else:
                buf = pickle.Unpickler(f)
            while True:
                try:
                    chunk_data.append(buf.load())
                except EOFError:
                    break
        return chunk_data
