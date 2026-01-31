class RepetitionTable:

    def __init__(self):
        self.table = {}
    def increment(self, zobrist_key):
        entry = self.table.get(zobrist_key)
        if entry:
            instances = entry.get("instances", 0)
            entry["instances"] = instances + 1
        else:
            self.table[zobrist_key] = {
                "instances" : 1
            }
    def decrement(self, zobrist_key):
        entry = self.table.get(zobrist_key)
        instances = entry["instances"]
        entry["instances"] = instances -1

    def lookup(self, zobrist_key):
        entry = self.table.get(zobrist_key)
        if entry:
            return entry["instances"]
        else:
            return 0