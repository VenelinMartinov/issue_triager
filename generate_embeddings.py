from pprint import pprint
import pandas as pd
from openai import OpenAI
import tiktoken

# embedding model parameters
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002
MAX_TOKENS = 8000  # the maximum for text-embedding-ada-002 is 8191

client = OpenAI()


def get_embeddings(text: list[str], model=EMBEDDING_MODEL) -> list[list[float]]:
    text = [subtext.replace("\n", " ") for subtext in text]
    resp = client.embeddings.create(input=text, model=model)
    return [x.embedding for x in resp.data]


if __name__ == "__main__":
    file_path = "all_github_issues_prs.csv"
    df = pd.read_csv(file_path)
    columns = [
        "ISSUE_HK",
        "REPOSITORY_HK",
        "AUTHOR_GITHUB_LOGIN_HK",
        "REPOSITORY_ID",
        "REPOSITORY_NAME",
        "ISSUE_ID",
        "ISSUE_API_ID",
        "REPO_ID",
        "ASSIGNEES_OBJECT",
        "ASSIGNEES",
        "AUTHOR_ASSOCIATION",
        "BODY",
        "CLOSED_AT",
        "COMMENTS",
        "CREATED_AT",
        "UPDATED_AT",
        "ISSUE_URL",
        "LOCKED",
        "FIRST_MILESTONE",
        "MILESTONE",
        "NODE_ID",
        "ISSUE_NUMBER",
        "ORG",
        "LABELS_OBJECT",
        "LABELS",
        "ISSUE_KIND",
        "REACTIONS_TOTAL",
        "REACTIONS_CONFUSED",
        "REACTIONS_EYES",
        "REACTIONS_HEART",
        "REACTIONS_HOORAY",
        "REACTIONS_LAUGH",
        "REACTIONS_ROCKET",
        "REACTIONS_PLUS_ONE",
        "REACTIONS_MINUS_ONE",
        "STATE",
        "TITLE",
        "TYPE",
        "API_URL",
        "PULL_REQUEST_HK",
        "ESTIMATED_TRIAGE_HOURS",
    ]
    filter_columns = [
        "REPOSITORY_NAME",
        "TITLE",
        "ISSUE_ID",
        "BODY",
        "LABELS",
    ]
    # get all unique values of repository_name

    df.columns = columns
    df = df[filter_columns]

    df = df[df.REPOSITORY_NAME.notnull() & df.TITLE.notnull() & df.BODY.notnull()]
    df = df[
        ~(df.TITLE.str.startswith("Upgrade terraform-provider", na=False))
        & ~(df.TITLE.str.startswith("Upgrade pulumi-terraform-bridge"))
        & ~(df.TITLE.str.startswith("Workflow failure:"))
        & ~(df.TITLE.str.startswith("Update GitHub Actions workflows"))
        & ~(df.TITLE.str.startswith("Combined dependencies PR"))
        & ~(df.TITLE.str.startswith("Bump"))  # Bump is used for dependabot PRs
        & ~(df.BODY.str.startswith("*Automated PR*"))
    ]
    df = df[~(df.REPOSITORY_NAME.str.startswith("pulumi-service"))]
    df["combined"] = (
        "Repository: "
        + df.REPOSITORY_NAME.str.strip()
        + "; Title: "
        + df.TITLE.str.strip()
        + "; Content: "
        + df.BODY.str.strip()
    )
    df.combined = df.combined.str.slice(0, MAX_TOKENS)
    df = df[df.combined.notnull()]

    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= MAX_TOKENS]

    # OpenAI allows up to 2000 elements in the API input
    chunks = []
    for chunk_num in range(len(df) // 2000 + 1):
        df_chunk = df.iloc[chunk_num * 2000 : (chunk_num + 1) * 2000]
        chunks += get_embeddings(df_chunk.combined)
    df["embedding"] = chunks
    df.to_csv("embeddings.csv", index=False)
