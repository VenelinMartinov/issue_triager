import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from generate_embeddings import ALL_COLUMNS

EMBEDDINGS_FILE = "embeddings.csv"


# https://stackoverflow.com/a/38884051
def largest_indices(ary, n) -> tuple:
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def test():
    df = pd.read_csv(EMBEDDINGS_FILE)
    df.columns = ALL_COLUMNS + ["combined", "n_tokens", "embedding"]

    df = df[
        df.TYPE.str.startswith("issue")
        & (df.ORG.str.startswith("pulumi"))
        & ~(df.TITLE.str.startswith("PATCH"))
        & ~(df.REPOSITORY_NAME.str.startswith("home"))
        & ~(df.REPOSITORY_NAME.str.startswith("devrel-team"))
        & ~(df.REPOSITORY_NAME.str.startswith("Revenue-Operations"))
        & ~(df.REPOSITORY_NAME.str.startswith("pulumi-hugo"))
        & ~(df.REPOSITORY_NAME.str.startswith("business-development"))
        # Remove closed issues for now - too much noise.
        & (df.STATE.str.startswith("open"))
    ]

    # TODO: filter out very short issues - these tend to be mostly noise

    # the vectors are saved as strings, need to convert them back.
    df.embedding = df.embedding.str.strip("[]")
    df.embedding = df.embedding.str.split(",")
    df.embedding = df.embedding.apply(lambda x: np.array(x, dtype=np.float32))

    arr = np.array(df.embedding.values.tolist())

    similarities = cosine_similarity(arr)
    # get the upper triangular part since the rest is repeated.
    similarities = np.triu(similarities, 1)

    largest = largest_indices(similarities, 1000)

    for index in range(len(largest[0])):
        # Filter out consecutive issues in the same repo - it's mostly noise
        if (
            df.iloc[largest[0][index]].REPOSITORY_NAME
            == df.iloc[largest[1][index]].REPOSITORY_NAME
            and abs(
                int(df.iloc[largest[0][index]].ISSUE_NUMBER)
                - int(df.iloc[largest[1][index]].ISSUE_NUMBER),
            )
            == 1
        ):
            continue

        # Filter out pairs where both issues are in pulumi
        if (
            df.iloc[largest[0][index]].REPOSITORY_NAME
            == df.iloc[largest[1][index]].REPOSITORY_NAME
            == "pulumi"
        ):
            continue

        print(similarities[largest[0][index], largest[1][index]])
        print(df.iloc[largest[0][index]].TITLE)
        print(df.iloc[largest[0][index]].ISSUE_URL)
        print(df.iloc[largest[1][index]].TITLE)
        print(df.iloc[largest[1][index]].ISSUE_URL)
        print()


def main():
    test()


if __name__ == "__main__":
    main()
