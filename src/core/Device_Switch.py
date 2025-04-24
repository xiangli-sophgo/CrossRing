from collections import deque

class DeviceSwitch:
    def __init__(self, ip_ids):
        """
        ip_ids: 一个可轮询的标识列表，元素用 (ip_type, ip_index) 表示
        """
        self.ip_ids = list(ip_ids)
        self.queues = {ip: deque() for ip in self.ip_ids}
        self._rr_index = 0

    def enqueue(self, ip_id, packet):
        if ip_id not in self.queues:
            raise KeyError(f"Unknown IP id: {ip_id}")
        self.queues[ip_id].append(packet)

    def has_pending(self):
        return any(self.queues[ip] for ip in self.ip_ids)

    def next(self):
        n = len(self.ip_ids)
        for _ in range(n):
            cur = self.ip_ids[self._rr_index]
            if self.queues[cur]:
                pkt = self.queues[cur].popleft()
                self._rr_index = (self._rr_index + 1) % n
                return cur, pkt
            self._rr_index = (self._rr_index + 1) % n
        return None, None
