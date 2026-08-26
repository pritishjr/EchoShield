import asyncio
import websockets

async def main() -> None:
    #fastapi websocket url as defined:
    url = "ws://127.0.0.1:8080/ws/audio"
    async with websockets.connect(url) as ws:
        #sending audio chunks (as bytes)
        await ws.send(b"fake-audio-bytes")
        
        #printing the result
        print(await ws.recv())
        
asyncio.run(main())
