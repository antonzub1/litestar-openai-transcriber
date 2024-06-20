import os

from typing import Annotated

from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@post("/transcribe", media_type=RequestEncodingType.MULTI_PART)
async def transcribe(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)]
) -> dict[str, str]:
    content = await data.read()
    transcription = client.audio.translations.create(model="whisper-1", file=content)

    return {"text": transcription.text}


app = Litestar([transcribe])
