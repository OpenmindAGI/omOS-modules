import asyncio
import logging
from websockets.asyncio.server import serve
from websockets import WebSocketClientProtocol, ConnectionClosed
from websockets.exceptions import InvalidUpgrade, InvalidHandshake
from queue import Empty, Queue
import uuid
import threading
from typing import Dict, Callable, Optional

logger = logging.getLogger(__name__)

logging.getLogger('websockets.server').setLevel(logging.WARNING)

class Server:
    def __init__(self, host: str = "localhost", port: int = 6789):
        self.host = host
        self.port = port
        self.running: bool = True
        self.connections: Dict[str, WebSocketClientProtocol] = {}
        self.queues: Dict[str, Queue[str | bytes]] = {}
        # This queue is used to send messages to all connections
        self.global_queue: Queue[str | bytes] = Queue()
        self.connection_callback: Optional[Callable] = None
        self.message_callbacks: Dict[str, Optional[Callable]] = {}

    def register_connection_callback(self, callback: Callable):
        self.connection_callback = callback
        logger.info("Connection callback registered")

    def register_message_callback(self, connection_id: str, callback: Callable):
        self.message_callbacks[connection_id] = callback
        logger.info(f"Registered message callback {connection_id}")

    async def process_global_messages(self):
        while self.running:
            try:
                message = self.global_queue.get_nowait()
                for connection_id in self.connections:
                    await self.connections[connection_id].send(message)
                self.global_queue.task_done()
            except Empty:
                await asyncio.sleep(0.05)
            except ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f'Error sending global message: {e}')
                pass

    async def process_connection_messages(self, connection_id: str):
        while self.running and connection_id in self.connections:
            try:
                queue = self.queues[connection_id]
                message = queue.get_nowait()
                websocket = self.connections[connection_id]
                await websocket.send(message)
                queue.task_done()
            except Empty:
                await asyncio.sleep(0.05)
            except ConnectionClosed:
                break
            except Exception as e:
                logger.error(f'Error sending message to connection {connection_id}: {e}')
                break

    async def receive_messages(self, websocket: WebSocketClientProtocol, connection_id: str):
        try:
            async for message in websocket:
                if connection_id in self.message_callbacks:
                    self.message_callbacks[connection_id](connection_id, message)
        except ConnectionClosed:
            logger.info(f'Connection closed: {connection_id}')
            pass
        except Exception as e:
            logger.error(f'Error receiving message from connection {connection_id}: {e}')

    async def handle_connection(self, websocket: WebSocketClientProtocol):
        connection_id = str(uuid.uuid4())
        try:
            # Store connection and create message queue
            self.connections[connection_id] = websocket
            self.queues[connection_id] = Queue()
            logger.info(f'New connection established: {connection_id}')

            # Notify about new connection if callback is registered
            if self.connection_callback:
                self.connection_callback('connect', connection_id)

            # Start tasks for sending and receiving messages
            send_global_task = asyncio.create_task(self.process_global_messages())
            send_task = asyncio.create_task(self.process_connection_messages(connection_id))
            receive_task = asyncio.create_task(self.receive_messages(websocket, connection_id))

            try:
                _, pending = await asyncio.wait([send_global_task, send_task, receive_task], return_when=asyncio.FIRST_COMPLETED)

                for task in pending:
                    task.cancel()

            except InvalidUpgrade as e:
                logger.warning(f"Invalid upgrade attempt: {e}")
                return
            except InvalidHandshake as e:
                logger.warning(f"Invalid handshake attempt: {e}")
                return
            except ConnectionClosed:
                logger.info(f"Connection closed normally: {connection_id}")
            except Exception as e:
                logger.error(f"Unexpected error in connection handler: {e}", exc_info=True)

        finally:
            # Notify about connection closure if callback is registered
            if self.connection_callback:
                self.connection_callback('disconnect', connection_id)

            # Clean up connection resources
            if connection_id in self.queues:
                del self.queues[connection_id]
            if connection_id in self.connections:
                del self.connections[connection_id]
                logger.info(f'Connection closed: {connection_id}')

    def handle_response(self, connection_id: str, msg: str | bytes):
        if connection_id in self.queues and self.running:
            self.queues[connection_id].put(msg)
            logger.info(
                f"Connection {connection_id} - Message queued: {self.format_message(msg)}"
            )

    def handle_global_response(self, msg: str | bytes):
        """This method is used to send a message to all connected clients"""
        if self.running and self.connections:
            self.global_queue.put(msg)
            logger.info(f"Global message queued: {self.format_message(msg)}")

    async def start_server(self):
        server = await serve(self.handle_connection, self.host, self.port)
        logger.info(f"WebSocket server started on port {self.port}")
        try:
            while self.running:
                await asyncio.sleep(1)
        finally:
            server.close()
            await server.wait_closed()

    def _run_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_server())

    def start(self):
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logger.info("WebSocket server thread started")

    def format_message(self, msg: str | bytes, max_length: int = 200):
        try:
            if len(msg) <= max_length:
                return msg

            preview_size = max_length // 2 - 20
            return f"{msg[:preview_size]}...{msg[-preview_size:]}"
        except Exception as e:
            return f"<Error formatting message: {e}>"

    def has_connections(self):
        return bool(self.connections)

    def stop(self):
        self.running = False