import argparse
import logging
import pandas as pd
import numpy as np
import requests
from urllib.parse import urlparse
from generate_embeddings import get_embeddings, EMBEDDINGS_FILE
from openai import OpenAI

client = OpenAI()
pd.set_option("display.max_colwidth", None)

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_combined_text(*, repo: str, title: str, created_at: str, body: str) -> str:
    return (
        "Repository: "
        + repo.strip()
        + "; Title: "
        + title.strip()
        + "; Created at: "
        + created_at.strip()
        + "; Content: "
        + body.strip()
    )


def get_embedding_related_issues(prompt: str, top_n: int = 10) -> pd.DataFrame:
    df = pd.read_csv(EMBEDDINGS_FILE)
    # the vectors are saved as strings, need to convert them back.
    df.embedding = df.embedding.str.strip("[]")
    df.embedding = df.embedding.str.split(",")
    df.embedding = df.embedding.apply(lambda x: np.array(x, dtype=np.float32))

    prompt_embedding = get_embeddings([prompt])[0]

    df["similarities"] = df.embedding.apply(
        lambda x: cosine_similarity(x, prompt_embedding)
    )
    df.sort_values(by="similarities", ascending=False, inplace=True)

    return df.head(top_n)


def llm_prompt(issue_one: str, issue_two: str) -> int:
    logging.debug("Prompts:\n%s\n%s", issue_one, issue_two)
    messages = [
        {
            "role": "system",
            "content": "You are an expert at triaging github issues for pulumi repositories. You are especially good at recognising if two issues are actually caused by the same underlying problem.\nHere is a bit of background on the repositories:\nPulumi is an infrastructure as code tool which allows users to provision cloud resources for various cloud providers. Each provider has a separate repository called pulumi-<provider-name>. Most of these are built on top of an upstream terraform provider through pulumi-terraform-bridge. For example, pulumi-aws is built on terraform-provider-aws through pulumi-terraform-bridge. There are also native providers which do not depend on terraform-bridge or the terraform provider, like pulumi-azure-native.\nYou'll be given the descriptions of the issues and you should output a number between 1 and 100 to mean how likely the two issues are to be caused by the same problem. This should either mean that two pulumi provider issues are caused by the same problem in pulumi, that a provider issue is caused by an issue in terraform-bridge or that a pulumi provider issue is caused by an upstream terraform provider issue. 1 means extremely unlikely, 100 means almost certainly.\nExplain your reasoning before outputting the number. Make sure to end with the numeric score.\nPay particular attention to the messages under diagnostics and the stack traces - if the error messages and stack traces are very similar then the two issues are very likely to be related, even if they relate to different resources. The code for handling resources in the pulumi providers is generated so problems often affect multiple resource in the same way.",
        },
        {
            "role": "user",
            "content": f"Are these issues related?\nIssue 1: {issue_one}\nIssue 2: {issue_two}\n",
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=4095,
    )

    messages += [
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content,
        },
        {
            "role": "user",
            "content": "Output just the numeric score.",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=4095,
    )

    logging.info(response.choices[0].message.content)


def get_github_issue_prompt(url: str) -> str:
    path = urlparse(url).path.split("/")
    issue_number = path[-1]
    repo = path[-3]
    owner = path[-4]
    resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    )
    resp.raise_for_status()
    data = resp.json()

    title = data["title"]
    body = data["body"]
    created_at = data["created_at"]
    return get_combined_text(repo=repo, title=title, created_at=created_at, body=body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    input_prompt = get_github_issue_prompt(args.url)
    df = get_embedding_related_issues(input_prompt, top_n=10)

    for i in range(len(df)):
        logging.info("Comparing %s", df.iloc[i].ISSUE_URL)
        other_issue = get_combined_text(
            repo=df.iloc[i].REPOSITORY_NAME,
            title=df.iloc[i].TITLE,
            created_at=df.iloc[i].CREATED_AT,
            body=df.iloc[i].BODY,
        )
        llm_prompt(input_prompt, other_issue)


if __name__ == "__main__":
    main()
