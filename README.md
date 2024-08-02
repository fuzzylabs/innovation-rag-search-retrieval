# RAG Search Retrieval

In this project, we will experiment with search and retrieval approaches for an RAG application.

# &#127939; How do I get started?

If you haven't already done so, please read [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on how to set up your virtual environment using Poetry.

## Run Locally

> Note: Add a pdf dataset to the [data](rag_search_retrieval/data) folder.

```bash
poetry shell
poetry install
poetry run jupyter notebook
```

Once the notebook is up, make sure you update the `FILE_PATH` parameter value. Once the correct file path is set, click `Run -> Run all cells` option.

It takes about 5 mins for everything to get completed if you have a Nvidia GPU.

Jump to the `Comparison` cell and toggle between different dropdown options to compare the results from various approaches.
