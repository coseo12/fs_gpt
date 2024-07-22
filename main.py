from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="CO Maximus Quote Giver",
    description="Get a real quote said by CO Maximus himself.",
    servers=[{"url": "https://az-reliance-rug-strip.trycloudflare.com"}],
)


class Quote(BaseModel):
    quote: str = Field(..., description="The quote that CO Maximus said.")
    year: int = Field(..., description="The year when CO Maximus.")


@app.get(
    "/quote",
    summary="Returns a random quote by CO Maximus",
    description="Upon receiving a GET request this endpoint will return a real quiote said by CO Maximus himself.",
    response_description="A Quote object that contains the quote said by CO Maximus and the date when the quote was said.",
    response_model=Quote,
)
def get_quote():
    return {"quote": "Life is short so eat it all.", "year": 2024}
