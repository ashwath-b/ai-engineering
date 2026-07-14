from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
import os

from app.schemas import ChatRequest
from app.core.sessions import conversation_store, get_or_create_session

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

@router.post("")
async def chat(request: ChatRequest):
    history = get_or_create_session(request.session_id)

    # Build full messages array with system prompt prepended
    full_messages = [
        {"role": "system", "content": request.system_prompt}
    ] + history + [
        {"role": "user", "content": request.message}
    ]

    async def stream_and_store():
        full_response = []

        stream = await client.chat.completions.create(
            model=request.model,
            messages=full_messages,
            temperature=request.temperature,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                full_response.append(content)
                yield content

        # Save assistant response to history after streaming completes
        history.append({"role": "user",      "content": request.message})
        history.append({
            "role": "assistant",
            "content": "".join(full_response)
        })

    return StreamingResponse(stream_and_store(), media_type="text/plain")
