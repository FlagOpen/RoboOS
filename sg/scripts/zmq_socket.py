import zmq
import numpy as np
import pickle
import zlib

class ZmqClientSocket:
    def __init__(self, ip, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")

    def send_dict(self, data: dict):
        """Send a dictionary as JSON."""
        # self.socket.send_json(data)
        byte_data = pickle.dumps(data)
        compressed_data = zlib.compress(byte_data)
        self.socket.send(compressed_data)

    def recv_dict(self) -> dict:
        """Receive a dictionary as JSON."""
        # return self.socket.recv_json()
        compressed_data = self.socket.recv()
        decompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(decompressed_data)
        return data

    # def send_dict(self, data: dict):
    #     """Send a dictionary as JSON."""
    #     self.socket.send_json(data)

    # def recv_dict(self) -> dict:
    #     """Receive a dictionary as JSON."""
    #     return self.socket.recv_json()

    def close(self):
        """Close the socket."""
        self.socket.close()
        self.context.term()


class ZmqSocket:
    def __init__(self, ip="192.168.1.50", port="9999"):
        self.client = ZmqClientSocket(ip, port)
    
    def send_info(self, info, type):  # type are include in [paths, pose, frame, over]
        try:
            message = {
                "type": f"{type}",
                f"{type}": info
            }
            self.client.send_dict(message)
        except Exception as e:
            print(f"Error: {e}")

    def received(self):
        recv_info = self.client.recv_dict()
        return recv_info
    
    def close(self):
        self.client.close()

if __name__ == '__main__':
    zmqsocket = ZmqSocket(ip="192.168.1.54", port="9999")
    try:
        paths = [
            [0.0,          0.0,        3.14],
            # [-0.49747512, -0.12629389, 3.52209903],
            # [-0.99747512, -0.32629389, 4.76477098]
        ]
        zmqsocket.send_info(info=paths, type="set_paths")
        print(zmqsocket.received())
        zmqsocket.send_info(info=[110, 220], type="get_frame")
        info = zmqsocket.received()
        print(np.array(info["rgb"]).shape)
        
    finally:
        zmqsocket.send_info(info="", type="close")
        print(zmqsocket.received())
        zmqsocket.close()
