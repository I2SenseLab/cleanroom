import asyncio
import threading
from time import time as _time

from .muse_bleak import MuseBleak


class Muse:
    """
    Legacy Muse interface (from NeuroTechX/bci-workshop) re-implemented
    on top of Bleak.  Provides the same synchronous API:
      m = Muse(address, callback=fn, eeg=True, backend='auto', interface=None)
      m.connect()
      m.start()
      m.stop()
      m.disconnect()

    Under the hood it runs an asyncio loop in a background thread
    so Bleak notifications can be processed continuously.
    """

    def __init__(
        self,
        address=None,
        callback=None,
        eeg=True,
        accelero=False,
        giro=False,
        backend='auto',
        interface=None,
        time_func=_time,
        name=None
    ):
        # Initialize the Bleak-based Muse
        self._inner = MuseBleak(
            address=address,
            callback=callback,
            eeg=eeg,
            time_func=time_func
        )
        self.name = name

        # Create a dedicated asyncio loop and thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )
        self._thread.start()

    def _run_loop(self):
        """Background thread: run the asyncio event loop forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def connect(self, interface=None, backend='auto'):
        """
        Scan (if no address) and connect to the Muse.
        Blocks until connection and subscription complete.
        """
        fut = asyncio.run_coroutine_threadsafe(
            self._inner.connect(name=self.name),
            self._loop
        )
        return fut.result()

    def start(self):
        """
        Begin EEG streaming (0x02,0x64,0x0A).
        Blocks until the command is sent.
        """
        fut = asyncio.run_coroutine_threadsafe(
            self._inner.start(),
            self._loop
        )
        return fut.result()

    def stop(self):
        """
        Stop EEG streaming (0x02,0x68,0x0A).
        Blocks until the command is sent.
        """
        fut = asyncio.run_coroutine_threadsafe(
            self._inner.stop(),
            self._loop
        )
        return fut.result()

    def disconnect(self):
        """
        Stop streaming, disconnect, and shut down the background loop.
        """
        # Ensure streaming is stopped
        try:
            self.stop()
        except Exception:
            pass

        # Disconnect BLE
        fut = asyncio.run_coroutine_threadsafe(
            self._inner.disconnect(),
            self._loop
        )
        fut.result()

        # Tear down the loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
