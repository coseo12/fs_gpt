from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
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
    openapi_extra={
        "x-openai-isConsequential": True,
    },
)
def get_quote(request: Request):
    print(request.headers["authorization"])
    return {"quote": "Life is short so eat it all.", "year": 2024}


user_token_db = {"token": "hUCpjCx79e"}


@app.get("/authorize", response_class=HTMLResponse)
def handle_authorize(
    response_type: str,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
):

    return f"""
    <html>
        <head>
            <title>Authorization</title>
        </head>
        <body>
            <h1>Log Into CO Maximus</h1>
            <a href="{redirect_uri}?code=token&state={state}">Authorize CO Maximus GPT</a>
        </body>
    </html>
    """


@app.post("/token")
def handle_token(code=Form(...)):
    if user_token_db[code]:
        return {"access_token": user_token_db[code], "token_type": "bearer"}
    else:
        return {"error": "invalid_grant"}
