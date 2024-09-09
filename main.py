import argparse
import os
import dotenv
import openai
from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: List[Message]


class DenseSummary(BaseModel):
    denser_summary: str
    missing_entities: List[str]


class DenserSummaryCollection(BaseModel):
    system_prompt: str
    prompt: str
    summaries: List[DenseSummary]


def main():
    parser = argparse.ArgumentParser(description="Legacy Script Converter")
    parser.add_argument("--file", required=True, help="Pathto file to summarize")
    parser.add_argument("--num-passes", type=int, default=5, help="Number of passes to summarize")
    parser.add_argument("--length-in-words", type=int, default=80, help="Length of summary in words")
    parser.add_argument("--num-entities", type=int, default=3, help="Number of entities to identify in each summary")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model to use for summarization")

    args = parser.parse_args()

    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Access the API key from the environment variables
    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    with open(args.file, "r") as f:
        article = f.read()

    length_in_words = args.length_in_words
    num_entities = args.num_entities
    num_passes = args.num_passes

    chain_of_density_system_prompt = "You are an expert in writing rich and dense summaries in broad domains."
    chain_of_density_prompt = f"""
      Article:
      {article}
      
      ----

      You will generate increasingly concise, entity-dense summaries of the
      above Article.

      You must repeat the following 2 steps {num_passes} times.

      - Step 1: Identify 1-{num_entities} informative entities from the Article 
      which are missing from the previously generated summary and are the most
      relevant.

      - Step 2: Write a new, denser summary of identical length which covers
      every entity and detail from the previous summary plus the missing
      entities.

      A Missing Entity is:

      - Relevant: to the main story
      - Specific: descriptive yet concise (5 words or fewer)
      - Novel: not in the previous summary
      - Faithful: present in the Article
      - Anywhere: located anywhere in the Article

      Guidelines:
      - The first summary should be long (4-5 sentences, approx. 80 words) yet
      highly non-specific, containing little information beyond the entities
      marked as missing.

      - Use overly verbose language and fillers (e.g. "this article discusses")
      to reach approximately {length_in_words} words.

      - Make every word count: re-write the previous summary to improve flow and
      make space for additional entities.

      - Make space with fusion, compression, and removal of uninformative
      phrases like "the article discusses"

      - The summaries should become highly dense and concise yet
      self-contained, e.g., easily understood without the Article.

      - Missing entities can appear anywhere in the new summary.

      - Never drop entities from the previous summary. If space cannot be made,
      add fewer new entities.

      > Remember to use the exact same number of words for each summary.
      > Write the missing entities in missing_entities
      > Write the summary in denser_summary
      > Repeat the steps {num_passes} times per instructions above
      """
    conversation = Conversation(
        messages=[
            Message(role="system", content=chain_of_density_system_prompt),
            Message(role="user", content=chain_of_density_prompt),
        ]
    )

    openai_resp = client.beta.chat.completions.parse(model=args.model,
                                                     messages=conversation.dict()['messages'],
                                                     response_format=DenserSummaryCollection)

    dense_summary_collection = openai_resp.choices[0].message.parsed
    for summary in dense_summary_collection.summaries:
        print(summary.denser_summary)
        print(summary.missing_entities)
        print("---")


if __name__ == "__main__":
    main()
