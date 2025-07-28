import asyncio
import bitstring
import numpy as np
from time import time
from bleak import BleakClient, BleakScanner

class MuseBleak:
    """Muse 2016 headband via Bleak (CoreBluetooth on macOS)."""

    # UUIDs for the EEG data channels
    #
    # Original versions of this project expected five EEG channels, however
    # modern Muse headsets only expose four.  The final UUID (ending in
    # ``...7``) no longer produces notifications which meant
    # ``_handle_eeg`` never saw the last channel and therefore never invoked
    # the callback that feeds data to the web socket.  Removing the unused UUID
    # ensures we receive all expected packets and can forward samples to the
    # front-end.
    EEG_UUIDS = [
        '273e0003-4c4d-454d-96be-f03bac821358',
        '273e0004-4c4d-454d-96be-f03bac821358',
        '273e0005-4c4d-454d-96be-f03bac821358',
        '273e0006-4c4d-454d-96be-f03bac821358',
    ]
    # Control characteristic UUID for start/stop commands
    CONTROL_UUID = '273e0001-4c4d-454d-96be-f03bac821358'

    def __init__(self, address=None, callback=None, eeg=True, time_func=time):
        """
        address: BLE address (UUID on macOS) or None to scan
        callback: function(data: np.ndarray, timestamps: np.ndarray)
        eeg: whether to subscribe to EEG channels
        """
        self.address   = address
        self.callback  = callback
        self.eeg       = eeg
        self.time_func = time_func
        self.client    = None
        self._init_sample()

    def _init_sample(self):
        """Reset sample buffers for the number of subscribed EEG channels."""
        n_chan = len(self.EEG_UUIDS)
        self.timestamps = np.zeros(n_chan)
        self.data = np.zeros((n_chan, 12))
        self.last_tm    = 0

    @staticmethod
    async def find_address(name=None, timeout=10.0):
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            if name and d.name == name:
                return d.address
            if not name and d.name and 'Muse' in d.name:
                return d.address
        return None

    async def connect(self, name=None):
        if self.address is None:
            self.address = await self.find_address(name)
            if not self.address:
                raise RuntimeError("Unable to find Muse device")
        self.client = BleakClient(self.address)
        await self.client.connect()
        if self.eeg:
            for uuid in self.EEG_UUIDS:
                await self.client.start_notify(uuid, self._handle_eeg)

    async def start(self):
        """Start EEG streaming via the control UUID."""
        self._init_sample()
        await self.client.write_gatt_char(
            self.CONTROL_UUID,
            bytearray([0x02, 0x64, 0x0A]),
            response=False
        )

    async def stop(self):
        """Stop EEG streaming via the control UUID."""
        await self.client.write_gatt_char(
            self.CONTROL_UUID,
            bytearray([0x02, 0x68, 0x0A]),
            response=False
        )

    async def disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()

    def _unpack_eeg_channel(self, packet: bytearray):
        aa = bitstring.Bits(bytes=bytes(packet))
        pattern = "uint:16," + ",".join(["uint:12"]*12)
        vals = aa.unpack(pattern)
        tm, samples = vals[0], np.array(vals[1:])
        samples = 0.48828125 * (samples - 2048)
        return tm, samples

    def _handle_eeg(self, sender, packet: bytearray):
        """
        Notification callback: sender is a BleakGATTCharacteristic.
        Use sender.uuid to identify the characteristic.
        """
        # Debug: show raw notification
        try:
            uuid_str = sender.uuid
        except AttributeError:
            uuid_str = str(sender)
        print(f"🔔 notification from {uuid_str}, Packet: {packet}")

        raw_uuid = uuid_str.lower()
        matches = [i for i, u in enumerate(self.EEG_UUIDS) if u.lower() == raw_uuid]
        if not matches:
            print(f"⚠️ Received data from unknown UUID: {uuid_str}")
            return
        idx = matches[0]

        try:
            tm, samples = self._unpack_eeg_channel(packet)
        except Exception as e:
            print("❌ Error unpacking EEG packet:", e)
            return

        if self.last_tm == 0:
            self.last_tm = tm - 1

        self.data[idx]       = samples
        self.timestamps[idx] = self.time_func()

        if idx == len(self.EEG_UUIDS) - 1:
            if tm != self.last_tm + 1:
                print(f"missing sample {tm} (last {self.last_tm})")
            self.last_tm = tm

            rel = np.arange(-12, 0) / 256.0
            rel += np.min(self.timestamps[self.timestamps != 0])

            if self.callback:
                try:
                    self.callback(self.data.copy(), rel.copy())
                except Exception as e:
                    print("❌ Error in user callback:", e)

            self._init_sample()
