from multiprocessing import Pool
import math
from tqdm import tqdm


def parallel_apply(
    function, dataframe, column: str = None, n_jobs=16, chunk_size=32, **kwargs
):
    """
    Parallelize a function that can be applied to a pandas dataframe.
    """
    out, pool = [], Pool
    try:
        with Pool(n_jobs) as pool:
            n_chunks = math.ceil(len(dataframe) / chunk_size)
            subdivisions = [
                dataframe.iloc[i * chunk_size : (i + 1) * chunk_size]
                for i in range(n_chunks)
            ]
            for _, chunk in tqdm(enumerate(subdivisions), total=n_chunks, unit="chunk"):
                subdiv = chunk.tolist() if column is None else chunk[column].tolist()
                std_smiles = pool.map(function, subdiv)
                out.extend(std_smiles)
    except KeyboardInterrupt:
        print("Interrupted by user.")

    pool.terminate()
    pool.join()

    return out
