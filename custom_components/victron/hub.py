"""Support for Victron Energy devices."""

from collections import OrderedDict
import errno
import logging
import socket
import threading

from packaging import version
import pymodbus
from pymodbus.client import ModbusTcpClient

from homeassistant.exceptions import HomeAssistantError

from .const import (
    DEFAULT_MODBUS_RETRIES,
    INT16,
    INT32,
    INT64,
    STRING,
    UINT16,
    UINT32,
    UINT64,
    register_info_dict,
    valid_unit_ids,
)

_LOGGER = logging.getLogger(__name__)

# TCP / transport failures where closing the socket and connecting again is appropriate.
_RECOVERABLE_ERRNOS = frozenset(
    {
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ENOTCONN,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
    }
)


def _is_recoverable_transport_error(exc: Exception) -> bool:
    """Return True if the exception may be resolved by reconnecting the Modbus client."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in _RECOVERABLE_ERRNOS
    return False


class VictronHub:
    """Victron Hub."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        tcp_keepalive: bool = False,
        modbus_retries: int = DEFAULT_MODBUS_RETRIES,
    ) -> None:
        """Initialize."""
        self.host = host
        self.port = port
        self._keepalive = tcp_keepalive
        self._modbus_retries = max(0, int(modbus_retries))
        self._client = ModbusTcpClient(host=self.host, port=self.port)
        self._lock = threading.Lock()

    def is_still_connected(self):
        """Check if the connection is still open."""
        return self._client.is_socket_open()

    def convert_string_from_register(self, segment, string_encoding="ascii"):
        """Convert from registers to the appropriate data type."""
        if (
            version.parse("3.8.0")
            <= version.parse(pymodbus.__version__)
            <= version.parse("3.8.4")
        ):
            return self._client.convert_from_registers(
                segment, self._client.DATATYPE.STRING
            ).split("\x00")[0]
        return self._client.convert_from_registers(
            segment, self._client.DATATYPE.STRING, string_encoding=string_encoding
        ).split("\x00")[0]

    def convert_number_from_register(self, segment, dataType):
        """Convert from registers to the appropriate data type."""
        if dataType == UINT16:
            raw = self._client.convert_from_registers(
                segment, data_type=self._client.DATATYPE.UINT16
            )
        elif dataType == INT16:
            raw = self._client.convert_from_registers(
                segment, data_type=self._client.DATATYPE.INT16
            )
        elif dataType == UINT32:
            raw = self._client.convert_from_registers(
                segment, data_type=self._client.DATATYPE.UINT32
            )
        elif dataType == INT32:
            raw = self._client.convert_from_registers(
                segment, data_type=self._client.DATATYPE.INT32
            )
        return raw

    def _apply_tcp_keepalive(self) -> None:
        """Enable SO_KEEPALIVE on the underlying TCP socket. Caller must hold _lock."""
        if not self._keepalive:
            return
        sock = getattr(self._client, "socket", None)
        if sock is None:
            return
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError as err:
            _LOGGER.warning("Could not enable TCP keepalive: %s", err)

    def connect(self):
        """Connect to the Modbus TCP server."""
        with self._lock:
            ok = self._client.connect()
            if ok:
                self._apply_tcp_keepalive()
            return ok

    def disconnect(self):
        """Disconnect from the Modbus TCP server."""
        with self._lock:
            if self._client.is_socket_open():
                return self._client.close()
        return None

    def _reconnect(self) -> None:
        """Close the socket and open a new Modbus TCP session. Caller must hold _lock."""
        try:
            self._client.close()
        except OSError:
            pass
        if not self._client.connect():
            raise ConnectionError(
                f"Could not reconnect to Modbus TCP at {self.host}:{self.port}"
            )
        self._apply_tcp_keepalive()

    def _ensure_connected(self) -> None:
        """Connect if the client has no open socket. Caller must hold _lock."""
        if not self._client.is_socket_open():
            if not self._client.connect():
                raise ConnectionError(
                    f"Could not connect to Modbus TCP at {self.host}:{self.port}"
                )
            self._apply_tcp_keepalive()

    def _transport_attempts(self) -> int:
        """Number of Modbus operations to attempt (first try + retries)."""
        return self._modbus_retries + 1

    def write_register(self, unit, address, value):
        """Write a register."""
        slave = int(unit) if unit else 1
        max_attempts = self._transport_attempts()
        with self._lock:
            for attempt in range(max_attempts):
                try:
                    self._ensure_connected()
                    return self._client.write_register(
                        address=address, value=value, device_id=slave
                    )
                except Exception as err:
                    if (
                        attempt < max_attempts - 1
                        and _is_recoverable_transport_error(err)
                    ):
                        _LOGGER.warning(
                            "Modbus write failed (%s), reconnecting", err
                        )
                        self._reconnect()
                        continue
                    raise

    def read_holding_registers(self, unit, address, count):
        """Read holding registers."""
        slave = int(unit) if unit else 1
        max_attempts = self._transport_attempts()
        _LOGGER.debug("Reading unit %s address %s count %s", unit, address, count)
        with self._lock:
            for attempt in range(max_attempts):
                try:
                    self._ensure_connected()
                    return self._client.read_holding_registers(
                        address=address, count=count, device_id=slave
                    )
                except Exception as err:
                    if (
                        attempt < max_attempts - 1
                        and _is_recoverable_transport_error(err)
                    ):
                        _LOGGER.warning(
                            "Modbus read failed (%s), reconnecting", err
                        )
                        self._reconnect()
                        continue
                    raise

    def calculate_register_count(self, registerInfoDict: OrderedDict):
        """Calculate the number of registers to read."""
        first_key = next(iter(registerInfoDict))
        last_key = next(reversed(registerInfoDict))
        end_correction = 1
        if registerInfoDict[last_key].dataType in (INT32, UINT32):
            end_correction = 2
        elif registerInfoDict[last_key].dataType in (INT64, UINT64):
            end_correction = 4
        elif isinstance(registerInfoDict[last_key].dataType, STRING):
            end_correction = registerInfoDict[last_key].dataType.length

        return (
            registerInfoDict[last_key].register - registerInfoDict[first_key].register
        ) + end_correction

    def get_first_register_id(self, registerInfoDict: OrderedDict):
        """Return first register id."""
        first_register = next(iter(registerInfoDict))
        return registerInfoDict[first_register].register

    def determine_present_devices(self):
        """Determine which devices are present."""
        valid_devices = {}

        _LOGGER.debug("Determining present devices")

        for unit in valid_unit_ids:
            working_registers = []
            for key, register_definition in register_info_dict.items():
                _LOGGER.debug("Checking unit %s for register set %s", unit, key)
                # VE.CAN device zero is present under unit 100. This seperates non system / settings entities into the seperate can device
                if unit == 100 and not key.startswith(("settings", "system")):
                    continue

                try:
                    address = self.get_first_register_id(register_definition)
                    count = self.calculate_register_count(register_definition)
                    result = self.read_holding_registers(unit, address, count)
                    if result.isError():
                        _LOGGER.debug(
                            "result is error for unit: %s address: %s count: %s",
                            unit,
                            address,
                            count,
                        )
                    else:
                        working_registers.append(key)
                except (HomeAssistantError, OSError) as e:
                    _LOGGER.error("Device scan read failed: %s", e)

            if len(working_registers) > 0:
                valid_devices[unit] = working_registers
            else:
                _LOGGER.debug("no registers found for unit: %s", unit)

        return valid_devices
