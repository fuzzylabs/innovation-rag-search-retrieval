"""FastHTML application."""

import math
import os
from operator import itemgetter
from typing import Any

import pandas as pd
from fasthtml.common import (
    Div,
    Html,
    Label,
    Main,
    Option,
    P,
    Select,
    Style,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Titled,
    Tr,
    fast_app,
    serve,
)
from logger import get_logger

# Highlight the row in table that matches this target text
TARGET_TEXT = "This microcontroller is manufactured by STMicroelectronics"
OUTPUT_PATH = "rag_search_retrieval/data/milvus_results.csv"

# Load the results data
if not os.path.exists(OUTPUT_PATH):
    raise ValueError(
        "Looks like output file is not generated. "
        "Run the notebooks to generate the output file."
    )
data = pd.read_csv(OUTPUT_PATH)
cols = list(data.columns)
cols.remove("Retrieved_Text")
approaches = [c.replace("Score_", "") for c in cols]


logger = get_logger("fasthtml", "INFO")
app, rt = fast_app()


@app.get("/create_table")  # type: ignore
def create_table(approach: str) -> Any:
    """Create a table."""
    logger.info(f"Option selected: {approach}")

    column_name = f"Score_{approach}"
    rows_data = []
    for _, row in data.iterrows():
        score = row[column_name]
        if math.isnan(score):
            continue
        retrieved_text = row["Retrieved_Text"]
        row_class = "highlight" if TARGET_TEXT.lower() in retrieved_text.lower() else ""
        rows_data.append((retrieved_text, score, row_class))

    # Sort rows by score in descending order of scores
    rows_data.sort(key=itemgetter(1), reverse=True)

    rows = [
        Tr(
            Th(retrieved_text, scope="row"),
            Td(f"{score:.4f}"),
            cls=row_class,
        )
        for retrieved_text, score, row_class in rows_data
    ]
    data_table = Table(
        Thead(
            Tr(
                Th("Retrieved Document", scope="col"),
                Th("Score", scope="col"),
            ),
        ),
        Tbody(*rows),
        id="data-table",
    )
    return data_table


@app.get("/clear_table")  # type: ignore
def clear_table() -> Any:
    """Clear the contents of the table."""
    return P("Nothing selected", id="data-table")


@app.get("/")  # type: ignore
def render_content() -> Any:
    """Create application using FastHTML."""
    style = Style(
        """
        .highlight td {
            background-color: #DEFC85;
        }
        .highlight th {
            background-color: #DEFC85;
        }
    """
    )

    approach1_drop_down = Select(
        Option(
            "--Choose Approach--",
            selected="",
            value="",
            get="clear_table",
            hx_target="#data-table",
        ),
        *[
            Option(
                approach,
                value=approach,
                get=f"create_table?approach={approach}",
                hx_target="#data-table",
            )
            for approach in approaches
        ],
        "false",
        name="select",
        aria_label="Select",
        required="",
    )
    approach2_drop_down = Select(
        Option(
            "--Choose Approach--",
            selected="",
            value="",
            get="clear_table",
            hx_target="#data-table1",
        ),
        *[
            Option(
                approach,
                value=approach,
                get=f"create_table?approach={approach}",
                hx_target="#data-table1",
            )
            for approach in approaches
        ],
        "false",
        name="select",
        aria_label="Select",
        required="",
    )
    content = Main(
        Html(data_theme="light"),
        style,
        Titled(
            "RAG Retrieval Comparison",
            Div(
                Div(Label("Approach 1:", for_="approach1"), approach1_drop_down),
                Div(Label("Approach 2:", for_="approach2"), approach2_drop_down),
                cls="grid",
            ),
            Div(
                Div(id="data-table", cls="table-container", style="width: 100%;"),
                Div(id="data-table1", cls="table-container", style="width: 100%;"),
                cls="grid",
            ),
        ),
    )
    return content


serve()
